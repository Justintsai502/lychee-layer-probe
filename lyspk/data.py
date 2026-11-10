#!/usr/bin/env python3
"""
Training data for the speaker head: one JSON line per recording.

  {"id": "dlg00012", "wav": "wavs/dlg00012.wav", "duration": 41.3, "source": "synth",
   "segments": [{"speaker": "vivian", "start": 0.0, "end": 2.71}, ...]}

``wav`` is relative to the manifest's folder unless absolute (mono or multi-channel;
channels are averaged, any rate is resampled to 16 kHz). ``speaker`` should be a
GLOBAL id: the same person in two recordings must carry the same string, because the
contrastive term takes its cross-recording positives from that.

Two producers:
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


def load_manifest(path):
    path = Path(path)
    rows = read_jsonl(path)
    for r in rows:
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


def load_frames(row, f0, f1):
    """Audio of frames [f0, f1) at 16 kHz, exactly (f1 - f0) * 1280 samples."""
    import soundfile as sf

    info = sf.info(row["wav_path"])
    sr = int(info.samplerate)
    a, b = int(round(f0 / FPS * sr)), int(round(f1 / FPS * sr))
    x, _ = sf.read(row["wav_path"], start=a, stop=min(b, info.frames), dtype="float32", always_2d=True)
    x = resample(x.mean(axis=1), sr, SR)
    want = (f1 - f0) * SAMPLES_PER_FRAME
    out = np.zeros(want, np.float32)
    out[:min(want, len(x))] = x[:want]
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
