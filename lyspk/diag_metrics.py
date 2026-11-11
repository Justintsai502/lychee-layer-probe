"""Vendored unchanged from mpmoshi-oracle/phase1d/diag_metrics.py (this line is the only edit).

Separated from ``diag_stage1.py`` so they can be tested without a GPU, a
checkpoint, or torch: everything here takes ``[T, 3]`` multihot labels, ``[T, 3]``
probabilities and a ``[T]`` keep mask, all in audio time, and returns counts.

See ``docs/EVAL_TOOLS_INVENTORY.md`` for why these three and not others.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def ids_of(mask: np.ndarray) -> np.ndarray:
    """``[T, 3]`` multihot -> one integer per frame (bit k = slot k). 0 is silence."""
    m = np.asarray(mask).astype(bool)
    return (m[:, 0].astype(np.int64) | (m[:, 1].astype(np.int64) << 1) | (m[:, 2].astype(np.int64) << 2))


def turns_of(y: np.ndarray, keep: np.ndarray) -> list[dict]:
    """Merged turns: per slot, the maximal runs where that slot is active.

    A slot keeps its turn while another slot joins and leaves again, because the
    run is over that slot's own activity and not over the frame's id. Splitting
    on the id instead cuts a holding at every co-speaker and inflates the count
    -- on the old stack chair 1's 898 real holdings were counted as 2098.

    ``appearance`` is the 1-indexed position of this turn among that slot's
    turns, so the first time a slot ever speaks can be told from its tenth:
    nothing in any cache is about a slot that has not spoken yet.
    """
    act = np.asarray(y) > 0
    keep = np.asarray(keep).astype(bool)
    out: list[dict] = []
    for k in range(act.shape[1]):
        g = act[:, k]
        i, n, seen = 0, len(g), 0
        while i < n:
            if not g[i]:
                i += 1
                continue
            j = i
            while j < n and g[j]:
                j += 1
            if keep[i:j].any():
                seen += 1
                out.append({
                    "slot": k, "start": i, "end": j, "appearance": seen,
                    "overlap": bool((act[i:j].sum(1) > 1).any()),
                })
            i = j
    out.sort(key=lambda t: (t["start"], t["slot"]))
    return out


def per_window(prob: np.ndarray, y: np.ndarray, keep: np.ndarray, theta: float) -> dict:
    """Frame, merged-turn and change-frame counts for one window, all in audio time."""
    keep = np.asarray(keep).astype(bool)
    gold_act = np.asarray(y) > 0
    pred_act = np.asarray(prob) >= float(theta)
    gold = ids_of(gold_act)
    pred = ids_of(pred_act)

    # Frame: the quantity headline() calls single_acc, recomputed here so the
    # turn and change numbers below sit on an identical decode and denominator.
    single = keep & (gold_act.sum(1) == 1)
    frame_ok = single & (pred == gold)

    # Turn: one count per holding, so a two-second turn weighs the same as a
    # thirty-second one. Three readings of the same turns:
    #
    # "mode" is the turn-level verdict and the one to quote: what the model
    #   mostly said over the turn against what was mostly true over it. A merged
    #   turn has no single gold id -- a co-speaker joins and leaves, so the id
    #   changes inside it -- which is why both sides take a mode rather than the
    #   prediction being compared to a fixed label.
    # "exact" is frame-level dressed as turn-level: the share of frames whose
    #   whole id matched, thresholded at half. A turn the model got mostly right
    #   in pieces can pass it while never settling on the right answer.
    # "held" asks only whether the right slot was on for most of the turn, and
    #   firing more slots can only raise it -- a policy scoring higher on held
    #   and lower on mode is being promiscuous, not better. Never quote it alone.
    held = exact = mode_ok = n_turn = 0
    held_1 = n_turn_1 = 0
    for turn in turns_of(y, keep):
        a, b, k = turn["start"], turn["end"], turn["slot"]
        m = keep[a:b]
        if not m.any():
            continue
        n_turn += 1
        ok = int(pred_act[a:b, k][m].mean() >= 0.5)
        held += ok
        exact += int((pred[a:b][m] == gold[a:b][m]).mean() >= 0.5)
        # Counter keeps first-seen order, so a tie resolves to whichever id
        # appeared earliest in the turn -- deterministic, and the same rule on
        # both sides.
        mode_ok += int(Counter(pred[a:b][m].tolist()).most_common(1)[0][0]
                       == Counter(gold[a:b][m].tolist()).most_common(1)[0][0])
        if turn["appearance"] == 1:
            n_turn_1 += 1
            held_1 += ok

    # Change frames: the gold id differs from the frame before, both kept, and
    # the new id is not silence. Repeating the previous label is wrong here by
    # construction, which is the whole point of measuring it separately.
    chg = np.zeros(len(gold), dtype=bool)
    chg[1:] = (gold[1:] != gold[:-1]) & keep[1:] & keep[:-1] & (gold[1:] != 0)
    c_ok = int((chg & (pred == gold)).sum())
    c_n = int(chg.sum())

    return {
        "n_frames": int(keep.sum()), "n_single": int(single.sum()),
        "frame_ok": int(frame_ok.sum()),
        "turn_ok": held, "turn_n": n_turn,
        "turn_exact_ok": exact, "turn_mode_ok": mode_ok,
        "turn_first_ok": held_1, "turn_first_n": n_turn_1,
        "change_ok": c_ok, "change_n": c_n,
    }


def ratio(ok: int, n: int) -> float | None:
    return None if n == 0 else ok / n


def boot(pairs: list[tuple[int, int, int, int]], iters: int, seed: int = 0,
         groups: list | None = None) -> dict:
    """Paired bootstrap of the a-minus-b rate over (ok_a, n_a, ok_b, n_b) rows.

    Resampling whole windows, not frames: frames inside a window are not
    independent, and the question is whether the difference survives a different
    draw of meetings.

    ``groups`` takes that one level further. Several windows cut from one
    meeting are not independent of each other either, so when they are present
    the resample draws *sessions* and takes all of each drawn session's windows.
    Resampling windows there would narrow the interval by counting the same
    meeting as several independent observations.
    """
    rows = [(p, (groups[i] if groups is not None else i)) for i, p in enumerate(pairs)
            if p[1] > 0 and p[3] > 0]
    if not rows:
        return {"delta": None, "lo": None, "hi": None, "n_windows": 0, "n_groups": 0}
    by_group: dict = {}
    for pair, g in rows:
        by_group.setdefault(g, []).append(pair)
    keys = list(by_group)
    arr = np.asarray([p for p, _g in rows], dtype=float)
    point = float(arr[:, 0].sum() / arr[:, 1].sum() - arr[:, 2].sum() / arr[:, 3].sum())
    if len(keys) < 2:
        return {"delta": point, "lo": None, "hi": None,
                "n_windows": len(rows), "n_groups": len(keys)}
    blocks = [np.asarray(by_group[k], dtype=float) for k in keys]
    rng = np.random.default_rng(seed)
    draws = np.empty(iters, dtype=float)
    for i in range(iters):
        pick = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        draws[i] = pick[:, 0].sum() / max(pick[:, 1].sum(), 1) - pick[:, 2].sum() / max(pick[:, 3].sum(), 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": point, "lo": float(lo), "hi": float(hi),
            "n_windows": len(rows), "n_groups": len(keys)}
