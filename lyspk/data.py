#!/usr/bin/env python3
"""
Training data for the speaker head: one JSON line per recording.

  {"id": "dlg00012", "wav": "wavs/dlg00012.wav", "duration": 41.3, "source": "synth",
   "segments": [{"speaker": "vivian", "start": 0.0, "end": 2.71}, ...]}

``wav`` is relative to the manifest's folder unless absolute (mono or multi-channel;
channels are averaged, any rate is resampled to 16 kHz). ``speaker`` should be a
GLOBAL id: the same person in two recordings must carry the same string, because the
contrastive term takes its cross-recording positives from that.

Multi-track rows (lyspk.mt_data: AMI / IF / self-distill) carry ``tracks`` {letter: wav},
``speakers`` {letter: global id}, ``agents``, letter-keyed ``segments`` and ``utterances``,
and, after lyspk.stoken, ``turns`` (a JSON of each participant's turns with speech tokens).
``view(row, agent)`` turns one into what a training item sees: the sum of the other
tracks as input and the others' segments under their global ids; the agent's turns are
the Lychee dialogue targets (lyspk.dialog).

Two producers of single-wav rows:
  python -m lyspk.build_synth ...                     synthetic dialogues from the TTS voices
  python -m lyspk.data rttm --rttm X.rttm --wav-dir D --pattern "{rec}.Mix-Headset.wav" \\
      --out work/spk/ami_train.jsonl --source ami      any RTTM corpus (AMI, ...)

Hard deps: numpy   Audio deps (lazy): soundfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import read_jsonl, resample, write_jsonl  # noqa: E402
from lyspk.labels import FPS, activity_from_segments, build_labels  # noqa: E402

SR = 16000
SAMPLES_PER_FRAME = int(SR / FPS)          # 1280


def _abs(p, base):
    p = Path(p)
    return str(p if p.is_absolute() else base / p)


def load_manifest(path):
    path = Path(path)
    rows = read_jsonl(path)
    for r in rows:
        if "tracks" in r:
            r["tracks"] = {k: _abs(v, path.parent) for k, v in r["tracks"].items()}
            if r.get("turns"):
                r["turns"] = _abs(r["turns"], path.parent)
            r.setdefault("source", path.stem)
            r["segments"] = sorted(r["segments"], key=lambda s: float(s["start"]))
            continue
        w = Path(r["wav"])
        r["wav_path"] = str(w if w.is_absolute() else path.parent / w)
        r.setdefault("source", path.stem)
        if "duration" not in r:
            import soundfile as sf
            r["duration"] = float(sf.info(r["wav_path"]).duration)
        r["segments"] = sorted(r["segments"], key=lambda s: float(s["start"]))
    return rows


def n_frames(row):
    return int(float(row["duration"]) * FPS)


def is_multitrack(row):
    return "tracks" in row


def load_turns(row):
    """{letter: [{start, end, text, stoken}]} from the row's ``turns`` JSON (lyspk.stoken), cached."""
    if not row.get("turns"):
        return {}
    if "_turns" not in row:
        import json
        row["_turns"] = json.loads(Path(row["turns"]).read_text())["turns"]
    return row["_turns"]


def view(row, agent=None):
    """What one item sees. Single-wav rows pass through. Multi-track rows: the input is the sum
    of every track except ``agent``'s; segments are the others', keyed by global speaker id
    (the agent takes no slot, as in mpmoshi); ``turns`` are the agent's (may be empty)."""
    if not is_multitrack(row):
        return row
    if agent is not None and agent not in row["tracks"]:
        raise KeyError(f"{row['id']}: no track for agent {agent}")
    spk = row["speakers"]
    others = [L for L in sorted(row["tracks"]) if L != agent]
    segs = [{"speaker": spk[s["speaker"]], "start": s["start"], "end": s["end"]}
            for s in row["segments"] if s["speaker"] != agent]
    return {"id": row["id"] if agent is None else f"{row['id']}@{agent}", "source": row["source"],
            "duration": row["duration"], "segments": segs, "wav_paths": [row["tracks"][L] for L in others],
            "agent": agent, "turns": load_turns(row).get(agent, []) if agent is not None else [],
            "agent_segments": [s for s in row["segments"] if agent is not None and s["speaker"] == agent]}


def _read_mono(path, f0, f1):
    import soundfile as sf

    info = sf.info(path)
    sr = int(info.samplerate)
    a, b = int(round(f0 / FPS * sr)), int(round(f1 / FPS * sr))
    x, _ = sf.read(path, start=min(a, info.frames), stop=min(b, info.frames), dtype="float32", always_2d=True)
    x = resample(x.mean(axis=1), sr, SR)
    want = (f1 - f0) * SAMPLES_PER_FRAME
    out = np.zeros(want, np.float32)
    out[:min(want, len(x))] = x[:want]
    return out


def load_frames(row, f0, f1):
    """Audio of frames [f0, f1) at 16 kHz, exactly (f1 - f0) * 1280 samples. A view with several
    ``wav_paths`` is their sum, scaled down only if it clips (mpmoshi's _load_ami_ihm_stereo)."""
    paths = row.get("wav_paths") or [row["wav_path"]]
    out = _read_mono(paths[0], f0, f1)
    for p in paths[1:]:
        out = out + _read_mono(p, f0, f1)
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if len(paths) > 1 and peak > 1.0:
        out = out / peak
    return out


def crop_labels(row, f0, f1, label_shift):
    """build_labels on frames [f0, f1) of the recording -> y, single, keep, slot_to_global, first_frame."""
    act, ids = activity_from_segments(row["segments"], n_frames(row))
    return build_labels(act, ids, f0, f1, label_shift=label_shift)


# ---------------------------------------------------------------- RTTM -> manifest

def read_rttm(path):
    by_rec = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        rec, start, dur, spk = p[1], float(p[3]), float(p[4]), p[7]
        by_rec.setdefault(rec, []).append({"speaker": spk, "start": start, "end": start + dur})
    return by_rec


def rttm_main(argv):
    ap = argparse.ArgumentParser(prog="lyspk.data rttm")
    ap.add_argument("--rttm", type=Path, required=True, nargs="+")
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--pattern", default="{rec}.wav", help="wav file name for recording {rec}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="rttm")
    ap.add_argument("--speaker-prefix", default="", help="prepended to every speaker id (keep ids global)")
    cfg = ap.parse_args(argv)
    import soundfile as sf

    rows, missing = [], []
    for rttm in cfg.rttm:
        for rec, segs in sorted(read_rttm(rttm).items()):
            wav = cfg.wav_dir / cfg.pattern.format(rec=rec)
            if not wav.exists():
                missing.append(str(wav))
                continue
            for s in segs:
                s["speaker"] = cfg.speaker_prefix + s["speaker"]
            rows.append({"id": rec, "wav": str(wav.resolve()), "duration": float(sf.info(str(wav)).duration),
                         "source": cfg.source, "segments": segs})
    write_jsonl(cfg.out, rows)
    print(f"wrote {cfg.out}: {len(rows)} recordings ({len(missing)} wavs missing{', e.g. ' + missing[0] if missing else ''})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rttm":
        rttm_main(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
