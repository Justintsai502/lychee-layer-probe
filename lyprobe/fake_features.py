#!/usr/bin/env python3
"""
Offline stand-in for the extraction: a Lychee-shaped features file with a KNOWN answer.

Planted, per point: the voice signal fades along the trunk, the speech branch brings it
back, the control branch keeps it flat; the sentence signal does the opposite. The
analysis should recover acoustic_share falling along the trunk and rising on speech.

Hard deps: numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, setup_project_cache, utt_id  # noqa: E402
from lyprobe.extract_lychee import point_table  # noqa: E402


def planted_gains(pts):
    """(speaker gain, sentence gain) per point."""
    out = []
    for name, depth, track in pts:
        if track == "trunk":
            a, b = 1.6 * np.exp(-depth / 10), 0.4 + 1.2 * depth / 28
        elif track == "control":
            a, b = 0.6, 1.0
        else:                                   # speech, merge: the voice comes back
            k = depth - 24
            a, b = 0.5 + 0.2 * k, 1.4 - 0.1 * k
        out.append((a, b))
    return np.array(out)


def make(n_spk=8, n_sent=20, dim=48, seed=0):
    pts = point_table(28, {"control": 20, "speech": 24}, {"control": 4, "speech": 4, "merge": 4})
    gains = planted_gains(pts)
    rng = np.random.default_rng(seed)
    spk_vec, sent_vec = rng.normal(size=(n_spk, dim)), rng.normal(size=(n_sent, dim))
    feats, spk, sent, utts = [], [], [], []
    for s in range(n_spk):
        for t in range(n_sent):
            feats.append(gains[:, :1] * spk_vec[s] + gains[:, 1:] * sent_vec[t]
                         + rng.normal(size=(len(pts), dim)))
            spk.append(f"spk{s:02d}")
            sent.append(f"s{t:03d}")
            utts.append(utt_id(spk[-1], sent[-1]))
    return pts, gains, np.stack(feats).astype(np.float32), np.array(utts), np.array(spk), np.array(sent)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "work/fake")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    setup_project_cache()

    pts, _, feats, utts, spk, sent = make(seed=cfg.seed)
    cfg.out.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.out / "lychee_fake.npz", pooled=feats, utt_id=utts, speaker=spk, sentence_id=sent,
             n_speech_slots=np.full(len(utts), 60),
             point_names=np.array([p[0] for p in pts]), point_depths=np.array([p[1] for p in pts]),
             point_track=np.array([p[2] for p in pts]), meta=json.dumps({"model": "Fake (Lychee-shaped)"}))
    print(f"wrote {cfg.out}/lychee_fake.npz {feats.shape}")


if __name__ == "__main__":
    sys.exit(main())
