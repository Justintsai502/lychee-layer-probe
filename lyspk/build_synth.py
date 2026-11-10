#!/usr/bin/env python3
"""
Synthetic multi-speaker dialogues from the TTS clips already on the server
(bayling-speaker-probe: 25 voices x 40 Harvard sentences), with exact per-frame labels.

Each dialogue:
  - 2, 3 or 4 speakers (40 / 45 / 15 %); 4-speaker dialogues exercise the "keep 3
    slots, mask the 4th" rule
  - turns of 1-2 sentences by one speaker; the next speaker differs 85 % of the time
  - gap between turns uniform in [-0.6, 1.2] s (negative = overlap), plus a 2-4 s
    silence before 10 % of turns (identity has to survive silence through the cache)
  - per-speaker gain +-4 dB; length 20-60 s
  - segments = the trimmed speech span of each sentence (lyprobe.common.trim_silence)

Voices are split: --val-speakers of them (default 5, seeded) appear only in val, so
validation measures speakers the head has never seen.

  python -m lyspk.build_synth                       # -> work/spk/synth/{train,val}.jsonl + wavs/
  python -m lyspk.build_synth --n-train 50 --n-val 5 --out work/spk/synth_smoke

Hard deps: numpy   Audio deps (lazy): soundfile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, SIBLING_MANIFEST, load_audio, read_jsonl, trim_silence, write_jsonl  # noqa: E402

SR = 16000


def load_voices(manifest, root, limit_per_voice=0):
    by_spk = {}
    for r in read_jsonl(manifest):
        x, _ = load_audio(root / r["wav"], sr=SR)
        speech, _, _ = trim_silence(x, SR)
        if len(speech) < int(0.5 * SR):
            continue
        lst = by_spk.setdefault(str(r["speaker"]), [])
        if not limit_per_voice or len(lst) < limit_per_voice:
            lst.append(speech.astype(np.float32))
    return by_spk


def make_dialogue(voices, speakers, rng, min_s=20.0, max_s=60.0):
    n_spk = int(rng.choice([2, 3, 4], p=[0.40, 0.45, 0.15]))
    who = list(rng.choice(speakers, size=min(n_spk, len(speakers)), replace=False))
    gain = {s: 10 ** (rng.uniform(-4, 4) / 20) for s in who}
    target = rng.uniform(min_s, max_s)
    pieces, segs = [], []
    t, prev = 0.3, None
    while t < target:
        if prev is None or rng.random() < 0.85:
            choices = [s for s in who if s != prev] or who
            spk = choices[int(rng.integers(len(choices)))]
        else:
            spk = prev
        if prev is not None:
            gap = rng.uniform(-0.6, 1.2)
            if rng.random() < 0.10:
                gap = rng.uniform(2.0, 4.0)
            if spk == prev:
                gap = max(gap, 0.15)          # one voice does not overlap itself
            t = max(0.0, t + gap)
        for k in range(int(rng.integers(1, 3))):
            clip = voices[spk][int(rng.integers(len(voices[spk])))] * gain[spk]
            pieces.append((t, clip))
            segs.append({"speaker": spk, "start": round(t, 4), "end": round(t + len(clip) / SR, 4)})
            t += len(clip) / SR + (0.15 if k == 0 else 0.0)
        prev = spk
        t = segs[-1]["end"]
    total = int((t + 0.5) * SR)
    wave = np.zeros(total, np.float32)
    for start, clip in pieces:
        a = int(round(start * SR))
        wave[a:a + len(clip)] += clip[:max(0, total - a)]
    peak = float(np.abs(wave).max())
    if peak > 0.99:
        wave *= 0.99 / peak
    return wave, segs, who


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SIBLING_MANIFEST)
    ap.add_argument("--tts-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "work/spk/synth")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--val-speakers", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=20.0)
    ap.add_argument("--max-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    import soundfile as sf

    if not cfg.manifest.exists():
        raise SystemExit(f"manifest not found: {cfg.manifest}")
    voices = load_voices(cfg.manifest, cfg.tts_dir or cfg.manifest.parent)
    names = sorted(voices)
    rng = np.random.default_rng(cfg.seed)
    val_spk = sorted(rng.choice(names, size=cfg.val_speakers, replace=False).tolist())
    train_spk = [s for s in names if s not in val_spk]
    if len(train_spk) < 4 or len(val_spk) < 2:
        raise SystemExit(f"need >= 4 train and >= 2 val voices, have {len(train_spk)} / {len(val_spk)}")
    print(f"{len(names)} voices: train {len(train_spk)}, val {val_spk}")
    (cfg.out / "wavs").mkdir(parents=True, exist_ok=True)
    for split, n, spk in (("train", cfg.n_train, train_spk), ("val", cfg.n_val, val_spk)):
        rows = []
        for i in range(n):
            wave, segs, who = make_dialogue(voices, spk, rng, cfg.min_s, cfg.max_s)
            did = f"{split}{i:05d}"
            sf.write(str(cfg.out / "wavs" / f"{did}.wav"), wave, SR, subtype="PCM_16")
            rows.append({"id": did, "wav": f"wavs/{did}.wav", "duration": len(wave) / SR, "source": "synth",
                         "speakers": who, "segments": segs})
            if (i + 1) % 100 == 0:
                print(f"{split} {i + 1}/{n}", flush=True)
        write_jsonl(cfg.out / f"{split}.jsonl", rows)
        hours = sum(r["duration"] for r in rows) / 3600
        print(f"wrote {cfg.out / (split + '.jsonl')}: {len(rows)} dialogues, {hours:.1f} h")
    (cfg.out / "split.json").write_text(json.dumps({"train": train_spk, "val": val_spk}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
