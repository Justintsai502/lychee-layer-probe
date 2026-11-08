#!/usr/bin/env python3
"""
Speaker-verification EER at every layer, from the audio encoder through the LLM.

Per point (and per baseline feature: ECAPA, Mimi):
  eer_cos        cosine on z-scored features, no training           (lower = more voice)
  eer_wccn       WCCN backend trained on 20 speakers, tested on 5 unseen speakers,
                 5 speaker-disjoint folds, scores pooled
  eer_*_hard     same, with hard impostors: different speakers saying the SAME sentence.
                 Above 0.5 = the layer matches words more than voices.

Trials on the complete speaker x sentence grid (25 x 30 = 750 clips):
  target     same speaker, different sentence        25 x C(30, 2) = 10875
  impostor   different speaker, different sentence   750 x 24 x 29 / 2 = 261000
  hard imp.  different speaker, same sentence        30 x C(25, 2) = 9000

Inputs: the LLM features (pooled [N, P, D] - lychee.npz, or bayling_silence.npz from
bayling-speaker-probe) and optionally the encoder features (extract_*_encoder.py). Both
are joined by utt_id, and every point is scored on the same clips.

  python lyprobe/analyze_eer.py --llm work/features/lychee.npz --encoder work/features/lychee_encoder.npz

Output: results/<name>_eer.json, work/results/<name>_eer.csv

Hard deps: numpy
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.analyze import load_baselines, load_features  # noqa: E402
from lyprobe.common import PROJECT_ROOT, RESULTS_DIR, SIBLING_BASELINES, complete_grid  # noqa: E402
from lyprobe.metrics import eer_cosine, eer_wccn  # noqa: E402


def load_encoder(path):
    d = np.load(path, allow_pickle=False)
    names = d["point_names"].tolist()
    return {"names": names, "depths": d["point_depths"].tolist(), "tracks": d["point_track"].tolist(),
            "X": [d[f"X_{k}"] for k in range(len(names))], "utt_id": d["utt_id"],
            "meta": json.loads(str(d["meta"]))}


def collect_points(llm, enc):
    """[(name, depth, track, X [N, D], utt_ids)] for encoder points then LLM points."""
    pts = []
    if enc is not None:
        for n, dp, t, X in zip(enc["names"], enc["depths"], enc["tracks"], enc["X"]):
            pts.append((n, dp, t, X, enc["utt_id"]))
    for p, (n, dp, t) in enumerate(zip(llm["names"], llm["depths"], llm["tracks"])):
        pts.append((n, dp, t, llm["pooled"][:, p], llm["utt_id"]))
    return pts


def align_grid(llm, enc, baselines):
    """Utterances shared by every input that form a complete speaker x sentence grid."""
    shared = set(llm["utt_id"].tolist())
    if enc is not None:
        shared &= set(enc["utt_id"].tolist())
    for _, u in baselines.values():
        shared &= set(u.tolist())
    keep = np.array([u in shared for u in llm["utt_id"]])
    grid, spk_set, sent_set = complete_grid(llm["speaker"][keep], llm["sentence_id"][keep])
    utts = llm["utt_id"][keep][grid]
    pos = {u: i for i, u in enumerate(llm["utt_id"].tolist())}
    idx = np.array([pos[u] for u in utts])
    return utts, llm["speaker"][idx], llm["sentence_id"][idx], spk_set, sent_set


def take(X, utt_ids, utts):
    pos = {u: i for i, u in enumerate(np.asarray(utt_ids).tolist())}
    return np.asarray(X)[np.array([pos[u] for u in utts])].astype(np.float32)


def score(X, spk, sent, folds, pca, seed):
    c, ch = eer_cosine(X, spk, sent)
    w, wh = eer_wccn(X, spk, sent, folds, pca, seed)
    return {"eer_cos": c, "eer_wccn": w, "eer_cos_hard": ch, "eer_wccn_hard": wh}


COLS = ["model", "point", "track", "depth", "dim", "eer_cos", "eer_wccn", "eer_cos_hard", "eer_wccn_hard"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, required=True, help="pooled LLM features (.npz)")
    ap.add_argument("--encoder", type=Path, default=None, help="encoder features (.npz); optional")
    ap.add_argument("--baselines", type=Path, default=SIBLING_BASELINES)
    ap.add_argument("--name", default=None, help="default: from the model (lychee, bayling)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5, help="speaker-disjoint folds for eer_wccn")
    ap.add_argument("--pca", type=int, default=200, help="PCA dims before WCCN")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    for p in (cfg.llm, cfg.encoder):
        if p is not None and not p.exists():
            raise SystemExit(f"not found: {p}")
    llm = load_features(cfg.llm)
    enc = load_encoder(cfg.encoder) if cfg.encoder else None
    name = cfg.name or llm["model"].split("-")[0].split(" ")[0].lower()
    cfg.json = cfg.json or RESULTS_DIR / f"{name}_eer.json"
    baselines = load_baselines(cfg.baselines)
    if not baselines:
        print(f"no baselines at {cfg.baselines}; continuing without")

    utts, spk, sent, spk_set, sent_set = align_grid(llm, enc, baselines)
    pts = collect_points(llm, enc)
    print(f"{llm['model']}: {len(utts)} clips | {len(spk_set)} speakers x {len(sent_set)} sentences "
          f"| {len(pts)} points ({0 if enc is None else len(enc['names'])} encoder)")
    if len(spk_set) < 6 or len(sent_set) < 3:
        print("WARNING: too few speakers/sentences for speaker-disjoint folds - pipeline check only; "
              "JSON goes to work/results/")
        if cfg.json.parent.resolve() == RESULTS_DIR.resolve():
            cfg.json = PROJECT_ROOT / "work/results" / f"{name}_eer_DEGENERATE.json"

    rows = []
    t0 = time.time()
    for i, (n, dp, t, X, u) in enumerate(pts):
        Xg = take(X, u, utts)
        r = score(Xg, spk, sent, cfg.folds, cfg.pca, cfg.seed)
        rows.append(dict(model=llm["model"], point=n, track=t, depth=dp, dim=int(Xg.shape[1]), **r))
        print(f"  {n:>11} d={dp:>4}  eer_cos {r['eer_cos']:.3f}  eer_wccn {r['eer_wccn']:.3f}  "
              f"hard {r['eer_cos_hard']:.3f}/{r['eer_wccn_hard']:.3f}  [{i + 1}/{len(pts)}, "
              f"{time.time() - t0:.0f}s]", flush=True)
    base_rows = []
    for bn, (F, u) in baselines.items():
        r = score(take(F, u, utts), spk, sent, cfg.folds, cfg.pca, cfg.seed)
        base_rows.append(dict(point=bn, dim=int(F.shape[1]), **r))
        print(f"  {bn:>11}  eer_cos {r['eer_cos']:.3f}  eer_wccn {r['eer_wccn']:.3f}")

    out_csv = PROJECT_ROOT / "work/results" / f"{name}_eer.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    res = {"model": llm["model"], "llm_features": str(cfg.llm),
           "encoder_features": str(cfg.encoder) if cfg.encoder else None,
           "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_clips": len(utts), "n_speakers": len(spk_set), "n_sentences": len(sent_set),
           "metric": "speaker-verification EER (0 = perfect, 0.5 = chance)",
           "trials": "target: same speaker, different sentence; impostor: different speaker, "
                     "different sentence; hard impostor: different speaker, same sentence",
           "eer_wccn_setup": f"PCA {cfg.pca} + WCCN, {cfg.folds} speaker-disjoint folds",
           "depth_note": "negative = audio encoder (before the LLM), 0 = LLM input embedding",
           "points": rows, "baselines": base_rows}
    cfg.json.parent.mkdir(parents=True, exist_ok=True)
    cfg.json.write_text(json.dumps(res, indent=1))
    print(f"wrote {cfg.json} and {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
