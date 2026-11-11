"""Lychee-FD's own training targets (text, speech tokens, control) on a multi-party timeline.

Read from HITsz-TMG/Lychee-FD @7bb1068, Training_package/Code/DataLoaders/LycheeFDDataset.py
(``interleaved_tokenizer``, ``_append_initial_user_turn``, ``_append_ai_response_turn``,
``mask_control_label``) and Models/LycheeFD.py (``clamp_loss_fn`` ranges); token ids from
the released config.json / added_tokens.json.

Lychee's sequence is one 40 ms slot per position; a control chunk is 10 slots (0.4 s,
one audio window). Each slot sums four channels: text, speech token (stoken), control,
audio. Lychee's own data is a two-party turn sequence; here one participant of a
multi-party recording is the agent and the others are the "user" (their tracks summed):

  listening chunk  text  text_pad x10          stoken  stoken_pad x10
                   control [sleep x8][detect][keep_listening | start_speaking]
  agent turn       text    text ids + <|EOT|>, then <tts_pad> up to the stoken length
                   stoken  <stoken_delay> x delay, <tts_start>, stokens + 151696, <tts_end>
                   then text_pad / stoken_pad to the end of the turn's last chunk
                   control [sleep x8][detect][keep_speaking | start_listening (last chunk)]
  audio            [<audio_patch>, audio_pad] x5 every chunk, always the users' real audio

The one deliberate change from Lychee's data (docs/SPK_HEAD_LYCHEE.md): Lychee feeds
silence as user audio while the AI speaks; here the other participants' real audio
stays in, because in a meeting they keep talking.

Labels sit at the same position as the input token and the model shifts them
(``clamp_loss_fn``: logits[i] predicts labels[i + 1]):
  text    the text ids, <|EOT|> included; <tts_pad> and padding ignored
  stoken  the stokens and <tts_end>; <stoken_delay> and <tts_start> ignored
  control the last slot of every chunk, thinned by ``mask_control_label`` exactly as
          Lychee does (all start_* and the ends of every keep run, plus a random
          min(20 %, 3 x specials) of the keep interiors)

Placement. A turn whose agent audio starts at ``s`` seconds begins at chunk
round((s - (delay + 1) * 0.04) / 0.4), so its first speech token lands near ``s``; it
takes ceil((len(stoken seq) + 1) / 10) chunks, like Lychee. A turn never starts before
one listening chunk has followed the previous one (that chunk carries start_speaking);
if it would, it is pushed later and counted in ``placement_stats``.

Hard deps: numpy
"""

from __future__ import annotations

import math
import random

import numpy as np

IGNORE = -100
SLOT_S = 0.04
CHUNK = 10
CHUNK_S = CHUNK * SLOT_S
AUDIO_TOKEN_OFFSET = 151696          # '<audio_0>': stoken k is id 151696 + k

TOK = {
    # control (config.json)
    "start_speaking": 158352, "start_listening": 158353, "keep_listening": 158354,
    "keep_speaking": 158355, "detect": 158356, "sleep": 158357, "start_bc": 158362,
    # channel pads (config.json)
    "text_pad": 158358, "stoken_pad": 158359, "audio_pad": 158360, "stoken_delay": 158361,
    # tokenizer specials (added_tokens.json)
    "tts_start": 151693, "tts_end": 151694, "tts_pad": 151695, "eot": 151665,
    "audio_patch": 151690,
}
# clamp_loss_fn ranges [lo, hi) in LycheeFD.forward
STOKEN_RANGE = (151694, 158352)      # config stoken_token_ids_min / _max
CONTROL_RANGE = (158352, 158363)     # control_token_ids_min .. start_bc_token_id + 1
STOKEN_DELAY = 10                    # config.json stoken_delay_num (the example script says 0;
                                     # the released model was trained with the config value)


def turn_tokens(text_ids, stokens, delay=STOKEN_DELAY, tok=TOK):
    """``interleaved_tokenizer`` (adding_text_hiddenstates False): equal-length text and stoken
    id lists for one agent turn. Text longer than the stoken sequence is cut, as in Lychee."""
    st = [tok["stoken_delay"]] * delay + [tok["tts_start"]] + [int(s) + AUDIO_TOKEN_OFFSET for s in stokens]
    st = st + [tok["tts_end"]]
    tx = list(text_ids) + [tok["eot"]]
    tx = tx[:len(st)]
    tx = tx + [tok["tts_pad"]] * (len(st) - len(tx))
    return tx, st


def turn_chunks(n_tokens):
    """Chunks a turn of ``n_tokens`` slots occupies: ceil((n + 1) / 10), so start_listening
    never overwrites a token (Lychee's '+1')."""
    return int(math.ceil((n_tokens + 1) / CHUNK))


def place_turns(turns, delay=STOKEN_DELAY):
    """Turns [{start, end, stoken (or n_stoken), ...}] (seconds) -> list of (c_start, c_end, turn) on the
    recording's chunk grid, and stats. c_end is exclusive; the next turn starts at c_end + 1 or later."""
    out, prev_end, pushed, push_chunks = [], -10 ** 9, 0, 0
    lead = (delay + 1) * SLOT_S
    for t in sorted(turns, key=lambda t: float(t["start"])):
        want = max(1, int(round((float(t["start"]) - lead) / CHUNK_S)))
        c0 = max(want, prev_end + 1)
        if c0 > want:
            pushed += 1
            push_chunks += c0 - want
        n = turn_chunks(delay + 2 + int(t.get("n_stoken", len(t.get("stoken", [])))))
        out.append((c0, c0 + n, t))
        prev_end = c0 + n
    return out, {"turns": len(out), "pushed": pushed, "pushed_chunks": push_chunks}


def snap_start(c0, placed):
    """Move a crop start off an agent turn: a crop must start on a listening chunk."""
    for a, b, _ in placed:
        if a <= c0 < b:
            return b
    return c0


def build_sequence(n_chunks, c0, placed, text_fn, delay=STOKEN_DELAY, tok=TOK):
    """Suffix channels and labels for chunks [c0, c0 + n_chunks) of a recording.

    ``placed`` from ``place_turns``; turns starting inside (c0, c0 + n_chunks) are used, the
    last one cut at the crop end. ``c0`` must be a listening chunk (``snap_start``).
    Returns a dict of int64 arrays of length 10 * n_chunks: text, stoken, control,
    text_label, stoken_label, control_label (unmasked), and speaking [n_chunks] bool."""
    S = CHUNK * n_chunks
    text = np.full(S, tok["text_pad"], np.int64)
    stoken = np.full(S, tok["stoken_pad"], np.int64)
    text_label = np.full(S, IGNORE, np.int64)
    stoken_label = np.full(S, IGNORE, np.int64)
    speaking = np.zeros(n_chunks, bool)
    last_chunk = np.zeros(n_chunks, bool)
    starts = np.zeros(n_chunks, bool)           # chunk is the first of a turn
    n_used = 0
    for a, b, t in placed:
        if a <= c0 < b:
            raise ValueError(f"crop start {c0} is inside an agent turn [{a}, {b})")
        if not c0 < a < c0 + n_chunks:
            continue
        n_used += 1
        tx, st = turn_tokens(text_fn(t["text"]), t["stoken"], delay, tok)
        lo = (a - c0) * CHUNK
        hi = min(S, lo + len(st))
        k = hi - lo
        text[lo:hi] = tx[:k]
        stoken[lo:hi] = st[:k]
        tl = np.array(tx[:k], np.int64)
        tl[tl == tok["tts_pad"]] = IGNORE
        sl = np.array(st[:k], np.int64)
        sl[(sl == tok["tts_start"]) | (sl == tok["stoken_delay"])] = IGNORE
        text_label[lo:hi] = tl
        stoken_label[lo:hi] = sl
        ca, cb = a - c0, min(n_chunks, b - c0)
        speaking[ca:cb] = True
        starts[ca] = True
        if b - c0 <= n_chunks:
            last_chunk[b - c0 - 1] = True
    control = np.full(S, tok["sleep"], np.int64)
    control_label = np.full(S, IGNORE, np.int64)
    for c in range(n_chunks):
        if speaking[c]:
            x = tok["start_listening"] if last_chunk[c] else tok["keep_speaking"]
        else:
            x = tok["start_speaking"] if c + 1 < n_chunks and starts[c + 1] else tok["keep_listening"]
        control[c * CHUNK + CHUNK - 2] = tok["detect"]
        control[c * CHUNK + CHUNK - 1] = x
        control_label[c * CHUNK + CHUNK - 1] = x
    return {"text": text, "stoken": stoken, "control": control, "text_label": text_label,
            "stoken_label": stoken_label, "control_label": control_label, "speaking": speaking,
            "n_turns": n_used}


def mask_control_label(label, tok=TOK, normal_keep_ratio=0.2, special_multiple=3, rng=None):
    """Lychee's ``mask_control_label`` as ``apply_mask_control_label`` calls it (no AI backchannel
    runs, no forced positions): keep every start_* and both ends of every keep_* run; of the
    keep interiors keep a random min(round(0.2 n), 3 x n_special); ignore the rest."""
    keep_set = {tok["keep_listening"], tok["keep_speaking"]}
    start_set = {tok["start_speaking"], tok["start_listening"]}
    rng = rng or random
    out = np.array(label, np.int64).copy()
    runs, cur, starts = [], [], []
    for i, v in enumerate(out.tolist()):
        if v == IGNORE:
            continue
        if v in keep_set:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
                cur = []
            if v in start_set:
                starts.append(i)
    if cur:
        runs.append(cur)
    special, normal = set(starts), []
    for r in runs:
        special.add(r[0])
        special.add(r[-1])
        normal.extend(r[1:-1])
    normal = [i for i in normal if i not in special]
    n_keep = min(int(round(normal_keep_ratio * len(normal))), special_multiple * len(special))
    n_keep = max(0, min(n_keep, len(normal)))
    kept = set(rng.sample(normal, n_keep)) if n_keep else set()
    for i in normal:
        if i not in kept:
            out[i] = IGNORE
    return out


def merge_utterances(utts, max_gap):
    """One speaker's utterances [{start, end, text}] -> turns, joining pauses shorter than ``max_gap`` s."""
    turns = []
    for u in sorted(utts, key=lambda u: float(u["start"])):
        s, e, txt = float(u["start"]), float(u["end"]), str(u.get("text", "")).strip()
        if e <= s:
            continue
        if turns and s - turns[-1]["end"] < max_gap:
            turns[-1]["end"] = max(turns[-1]["end"], e)
            turns[-1]["text"] = (turns[-1]["text"] + " " + txt).strip()
        else:
            turns.append({"start": s, "end": e, "text": txt})
    return turns


def crop_sequence(turns, f0, f1, text_fn, delay=STOKEN_DELAY, rng=None, align=5, train=True):
    """One crop's agent sequence. ``turns`` = the agent's turns (lyspk.data.view), frames [f0, f1)
    on 12.5 Hz with ``align`` frames per chunk. The start moves forward off any agent turn.
    Returns (f0, seq, placement stats) or (f0, None, stats) when nothing is left.

    train: control labels thinned by ``mask_control_label``. Otherwise every chunk end is
    labelled, and ``control_label_start`` holds only the start_* positions (for eval)."""
    placed, st = place_turns(turns, delay)
    c0 = snap_start(f0 // align, placed)
    c1 = f1 // align
    if c1 - c0 < 4:
        return c0 * align, None, st
    seq = build_sequence(c1 - c0, c0, placed, text_fn, delay)
    if train:
        seq["control_label"] = mask_control_label(seq["control_label"], rng=rng)
    else:
        lab = seq["control_label"]
        seq["control_label_start"] = np.where(np.isin(lab, [TOK["start_speaking"], TOK["start_listening"]]),
                                              lab, IGNORE)
    return c0 * align, seq, st


def agent_active_fraction(agent_segments, f0, f1, fps=12.5):
    """Share of 12.5 Hz frames in [f0, f1) where the agent speaks (its own segments, rastered
    like the head's labels: floor of the start, ceil of the end)."""
    on = np.zeros(max(int(f1) - int(f0), 0), bool)
    for sgm in agent_segments:
        a = max(int(np.floor(float(sgm["start"]) * fps + 1e-6)), int(f0))   # 4.4 s * 12.5 is 55.000...01
        b = min(int(np.ceil(float(sgm["end"]) * fps - 1e-6)), int(f1))
        if b > a:
            on[a - int(f0):b - int(f0)] = True
    return float(on.mean()) if len(on) else 0.0


def agent_text_weight(frac_agent_active):
    """mpmoshi moshi_ms/p1c_labels.py agent_text_weight, unchanged: down-weight the agent's text and
    speech loss when the agent is quiet in the crop. clip(frac / 0.15, 0.1, 1.0)."""
    return float(np.clip(float(frac_agent_active) / 0.15, 0.1, 1.0))
