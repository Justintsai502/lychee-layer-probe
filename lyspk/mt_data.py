#!/usr/bin/env python3
"""
Multi-track manifests from mpmoshi-oracle's AMI / IF / IF-self-distill manifests
(mpmoshi-oracle docs/AUDIO_AND_DATA.md describes the corpora and their files).

One output row per recording, every participant kept on its own track:

  {"id": "ami:EN2001b", "source": "ami", "duration": 3451.9,
   "tracks":   {"A": "/abs/EN2001b.Headset-0.wav", ...},      one wav per participant
   "speakers": {"A": "FEO066", ...},                          global ids (recur across recordings)
   "agents":   ["A", "B", "C", "D"],                          who may be cast as the agent
   "segments": [{"speaker": "A", "start": 5.50, "end": 6.07}, ...],        activity, by letter
   "utterances": [{"speaker": "A", "start": 5.50, "end": 6.07, "text": "Okay."}, ...]}

At training time one agent is drawn per crop: the input is the sum of the OTHER tracks,
the head labels only the others, and the agent's own turns become Lychee's text /
speech-token / control targets (lyspk.dialog). As in mpmoshi, a recording therefore
trains once per participant.

  AMI     meeting ids, agent letters and wav names from the mpmoshi manifest
          (ami_train_ads.jsonl repeats each meeting once per agent letter; rows are grouped
          back by meeting); activity from the NXT segments (transcriber_start / _end, as
          mpmoshi's prepare_ami.py), text from the words those segments point to;
          letter -> headset channel and global speaker name from meetings.xml.
  IF / SD scene dir = <root>/<scene>; tracks audio_*_<L>.wav; segments and text from
          aligned_script.jsonl (the timestamps the audio was rendered from); global ids
          from voices.json (the voice-prompt file stem), else "<source>:<scene>:<L>"
          (no cross-recording positives for those; the count is printed).

Hold-out: --exclude takes mpmoshi test manifests; their meetings / scenes are dropped
(sd4spk_if_train shares 131 scenes with if_4spk_test).

  python -m lyspk.mt_data ami --manifest $MP/data/manifests/ami_train_ads.jsonl \\
      --annotations $MP/data/ami/annotations --root $AMI_ROOT --out work/spk/mt/ami_train.jsonl
  python -m lyspk.mt_data scenes --manifest $MP/data/manifests/if_4spk_train.jsonl --source if \\
      --root $IF_ROOT --out work/spk/mt/if_train.jsonl
  python -m lyspk.mt_data scenes --manifest $MP/data/manifests/sd4spk_if_train.jsonl --source sd \\
      --root $IF_SD4_ROOT --exclude $MP/data/manifests/if_4spk_test.jsonl --out work/spk/mt/sd_train.jsonl

Hard deps: numpy   Audio deps (lazy): soundfile
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import read_jsonl, write_jsonl  # noqa: E402

LETTERS = "ABCDEFGH"
NITE_ID = "{http://nite.sourceforge.net/}id"


# ---------------------------------------------------------------- shared

def session_key(row):
    """Meeting id (AMI) or scene name (synthetic) of an mpmoshi manifest row."""
    meta = row.get("meta") or {}
    if meta.get("meeting_id"):
        return str(meta["meeting_id"])
    p = meta.get("agent_wav") or row.get("agent_audio_path") or row.get("audio_path")
    return Path(str(p)).parent.name


def excluded_keys(paths):
    keys = set()
    for p in paths or []:
        keys |= {session_key(r) for r in read_jsonl(p)}
    return keys


def env_roots(names):
    return [Path(os.environ[n]) for n in names if os.environ.get(n)]


def track_duration(tracks):
    import soundfile as sf
    return max(float(sf.info(str(p)).duration) for p in tracks.values())


# ---------------------------------------------------------------- AMI

def parse_meetings(path):
    """meetings.xml -> {meeting: {letter: {"channel": int, "global": str}}}."""
    out = {}
    for m in ET.parse(path).getroot().iter("meeting"):
        spk = {}
        for s in m.iter("speaker"):
            if s.get("nxt_agent") is None or s.get("channel") is None:
                continue
            spk[s.get("nxt_agent")] = {"channel": int(s.get("channel")), "global": s.get("global_name") or ""}
        out[m.get("observation")] = spk
    return out


def parse_words(path):
    """words.xml -> (ordered [(id, text, start, end, punc)], {id: index}). Only <w> elements."""
    words = []
    for el in ET.parse(path).getroot():
        if el.tag != "w":
            continue
        s, e = el.get("starttime"), el.get("endtime")
        words.append((el.get(NITE_ID), (el.text or "").strip(), float(s) if s else None,
                      float(e) if e else None, el.get("punc") == "true"))
    return words, {w[0]: i for i, w in enumerate(words)}


_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def join_words(ws):
    out = ""
    for _, text, _, _, punc in ws:
        if not text:
            continue
        out = out + text if (punc or not out) else out + " " + text
    return out.strip()


def ami_letter(meeting, letter, ann):
    """Segments of one participant with the text of the words they point to."""
    seg_path = ann / "segments" / f"{meeting}.{letter}.segments.xml"
    word_path = ann / "words" / f"{meeting}.{letter}.words.xml"
    if not seg_path.exists():
        return []
    words, index = parse_words(word_path) if word_path.exists() else ([], {})
    out = []
    for seg in ET.parse(seg_path).getroot().iter("segment"):
        s, e = seg.get("transcriber_start"), seg.get("transcriber_end")
        if s is None or e is None or float(e) <= float(s):
            continue
        text = ""
        for child in seg:
            m = _HREF.search(child.get("href", ""))
            if m and m.group(1) in index:
                a = index[m.group(1)]
                b = index.get(m.group(2) or m.group(1), a)
                text = (text + " " + join_words(words[a:b + 1])).strip()
        out.append({"speaker": letter, "start": round(float(s), 3), "end": round(float(e), 3), "text": text})
    return out


def find_headset(meeting, channel, roots):
    """mpmoshi's AMI lookup: roots R, R/amicorpus, R/headset, each with MEETING/audio/ or MEETING/."""
    name = f"{meeting}.Headset-{channel}.wav"
    for r in roots:
        for base in (r, r / "amicorpus", r / "headset"):
            for p in (base / meeting / "audio" / name, base / meeting / name):
                if p.exists():
                    return p.resolve()
    return None


def ami_main(argv):
    ap = argparse.ArgumentParser(prog="lyspk.mt_data ami")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True, help="mpmoshi data/ami/annotations")
    ap.add_argument("--root", type=Path, nargs="*", default=[], help="AMI audio roots (also $AMI_ROOT, $AMI_AUDIO_ROOT)")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--source", default="ami")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-missing", action="store_true", help="keep rows whose wavs are not found (dry runs)")
    cfg = ap.parse_args(argv)

    roots = list(cfg.root) + env_roots(["AMI_AUDIO_ROOT", "AMI_ROOT"])
    meetings = parse_meetings(cfg.annotations / "corpusResources" / "meetings.xml")
    skip = excluded_keys(cfg.exclude)
    by_meeting = {}
    for r in read_jsonl(cfg.manifest):
        mid = session_key(r)
        if mid in skip:
            continue
        g = by_meeting.setdefault(mid, {"agents": set(), "duration": float(r.get("duration") or 0)})
        g["agents"].add(str((r.get("meta") or {}).get("agent_letter") or r.get("agent_speaker_id")))
    rows, missing = [], []
    for mid, g in sorted(by_meeting.items()):
        spk = meetings.get(mid)
        if not spk:
            missing.append(f"{mid}: not in meetings.xml")
            continue
        tracks, speakers, utts = {}, {}, []
        for letter, info in sorted(spk.items()):
            p = find_headset(mid, info["channel"], roots)
            if p is None and not cfg.allow_missing:
                missing.append(f"{mid}.Headset-{info['channel']}.wav")
                tracks = None
                break
            tracks[letter] = str(p) if p else f"MISSING/{mid}.Headset-{info['channel']}.wav"
            speakers[letter] = info["global"] or f"{cfg.source}:{mid}:{letter}"
            utts += ami_letter(mid, letter, cfg.annotations)
        if tracks is None:
            continue
        utts.sort(key=lambda u: u["start"])
        agents = sorted(a for a in g["agents"] if a in tracks)
        rows.append({"id": f"{cfg.source}:{mid}", "source": cfg.source, "duration": g["duration"],
                     "tracks": tracks, "speakers": speakers, "agents": agents,
                     "segments": [{k: u[k] for k in ("speaker", "start", "end")} for u in utts],
                     "utterances": [u for u in utts if u["text"]]})
    write_jsonl(cfg.out, rows)
    print(f"wrote {cfg.out}: {len(rows)} meetings ({len(skip & set(by_meeting)) if skip else 0} held out by --exclude, "
          f"{len(missing)} dropped{', e.g. ' + missing[0] if missing else ''})")


# ---------------------------------------------------------------- synthetic scenes

def scene_tracks(scene_dir):
    """{letter: wav} for audio_*_<L>.wav; with several candidates per letter prefer *clone*."""
    out = {}
    for p in sorted(scene_dir.glob("audio_*_?.wav")):
        letter = p.stem[-1]
        if letter not in LETTERS:
            continue
        if letter not in out or ("clone" in p.name and "clone" not in Path(out[letter]).name):
            out[letter] = str(p.resolve())
    return out


def scene_speakers(scene_dir, letters, source, scene):
    """voices.json values are voice-prompt paths; the stem is the global speaker id."""
    f = scene_dir / "voices.json"
    ids = {}
    if f.exists():
        v = json.loads(f.read_text())
        for L in letters:
            if L in v and v[L]:
                ids[L] = Path(str(v[L] if not isinstance(v[L], dict) else v[L].get("path", ""))).stem or None
    return {L: ids.get(L) or f"{source}:{scene}:{L}" for L in letters}, bool(ids)


def scenes_main(argv):
    ap = argparse.ArgumentParser(prog="lyspk.mt_data scenes")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, nargs="*", default=[], help="scene roots (also $IF_ROOT, $IF4_ROOT, $IF_SD4_ROOT)")
    ap.add_argument("--source", required=True, help="pool name written to each row, e.g. if / sd")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    cfg = ap.parse_args(argv)

    roots = list(cfg.root) + env_roots(["IF_ROOT", "IF4_ROOT", "IF_SD4_ROOT"])
    skip = excluded_keys(cfg.exclude)
    rows, missing, no_voices, n_skip = [], [], 0, 0
    seen = set()
    for r in read_jsonl(cfg.manifest):
        scene = session_key(r)
        if scene in seen:
            continue
        seen.add(scene)
        if scene in skip:
            n_skip += 1
            continue
        d = next((root / scene for root in roots if (root / scene / "aligned_script.jsonl").exists()), None)
        if d is None:
            missing.append(f"{scene}: no <root>/{scene}/aligned_script.jsonl")
            continue
        tracks = scene_tracks(d)
        script = read_jsonl(d / "aligned_script.jsonl")
        letters = sorted({str(u["speaker"]) for u in script})
        if not set(letters) <= set(tracks):
            missing.append(f"{scene}: tracks {sorted(tracks)} vs speakers {letters}")
            continue
        tracks = {L: tracks[L] for L in letters}
        speakers, ok = scene_speakers(d, letters, cfg.source, scene)
        no_voices += not ok
        utts = sorted(({"speaker": str(u["speaker"]), "start": round(float(u["start"]), 3),
                        "end": round(float(u["end"]), 3), "text": str(u.get("text") or "").strip()}
                       for u in script if float(u["end"]) > float(u["start"])), key=lambda u: u["start"])
        rows.append({"id": f"{cfg.source}:{scene}", "source": cfg.source,
                     "duration": float(r.get("duration") or track_duration(tracks)),
                     "tracks": tracks, "speakers": speakers, "agents": letters,
                     "segments": [{k: u[k] for k in ("speaker", "start", "end")} for u in utts],
                     "utterances": [u for u in utts if u["text"]]})
    write_jsonl(cfg.out, rows)
    print(f"wrote {cfg.out}: {len(rows)} scenes ({n_skip} held out by --exclude, {len(missing)} dropped"
          f"{', e.g. ' + missing[0] if missing else ''}; {no_voices} without voices.json -> per-scene speaker ids)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ami":
        ami_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "scenes":
        scenes_main(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
