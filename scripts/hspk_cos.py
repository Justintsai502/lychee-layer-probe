"""Cosine similarity of the head's input features (hspk.npz from lyspk/diag.py), same vs different speaker.

CPU only, numpy only, seconds. Run on the server next to the diag output, push the small json:

  python scripts/hspk_cos.py work/spk/runs/mt/diag_ckpt_step001200/hspk.npz results/spk/mt/diag_step1200/hspk_cos.json

h_spk is readout(stacks) -- the per-frame features the Sortformer head and its cache see (not the
SupCon projection, which diag does not save). Units compared, all within one window (the cache
only ever compares speakers inside one recording):
  frame    single-speaker frames, up to 400 per window drawn at random
  segment  mean of 12 consecutive same-speaker frames (0.96 s, the SupCon anchor length), L2-normed
For each unit pair: same = same gold slot, diff = different slot. Reports mean cosines, their gap,
AUC (P[same > diff]) and EER, per subset and per window, plus the per-slot centroid cosine matrix.
"""
import json
import sys
from pathlib import Path

import numpy as np

SEG = 12
MAX_FRAMES = 400


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def segments(h, g):
    """Non-overlapping runs of SEG frames with one gold slot -> (embeddings, labels)."""
    out, lab = [], []
    i, n = 0, len(g)
    while i < n:
        if g[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and g[j] == g[i]:
            j += 1
        for a in range(i, j - SEG + 1, SEG):
            out.append(h[a:a + SEG].mean(0))
            lab.append(g[i])
        i = j
    if not out:
        return np.zeros((0, h.shape[1]), np.float32), np.zeros(0, int)
    return l2(np.stack(out)), np.asarray(lab)


def pair_scores(e, lab):
    if len(lab) < 2:
        return np.zeros(0), np.zeros(0)
    s = e @ e.T
    iu = np.triu_indices(len(lab), 1)
    same = lab[iu[0]] == lab[iu[1]]
    return s[iu][same], s[iu][~same]


def auc_eer(same, diff):
    if len(same) == 0 or len(diff) == 0:
        return None, None
    sc = np.concatenate([same, diff])
    y = np.concatenate([np.ones(len(same)), np.zeros(len(diff))])
    order = np.argsort(sc, kind="mergesort")
    ranks = np.empty(len(sc))
    ranks[order] = np.arange(1, len(sc) + 1)
    auc = (ranks[y == 1].sum() - len(same) * (len(same) + 1) / 2) / (len(same) * len(diff))
    th = np.unique(sc)
    if len(th) > 2000:
        th = np.quantile(sc, np.linspace(0, 1, 2000))
    frr = np.array([(same < t).mean() for t in th])
    far = np.array([(diff >= t).mean() for t in th])
    k = int(np.argmin(np.abs(frr - far)))
    return float(auc), float((frr[k] + far[k]) / 2)


def summarise(same, diff):
    auc, eer = auc_eer(same, diff)
    return {"n_same": int(len(same)), "n_diff": int(len(diff)),
            "cos_same": float(same.mean()) if len(same) else None,
            "cos_diff": float(diff.mean()) if len(diff) else None,
            "gap": float(same.mean() - diff.mean()) if len(same) and len(diff) else None,
            "auc": auc, "eer": eer}


def main(src, dst):
    z = np.load(src, allow_pickle=True)
    H, G, W, D = z["h_spk"].astype(np.float32), z["gold"], z["window"], z["dataset"].astype(str)
    rng = np.random.default_rng(0)
    pooled = {}
    per_window = []
    for w in np.unique(W):
        m = W == w
        h, g, sub = H[m], G[m], D[m][0]
        rec = {"window": int(w), "dataset": sub, "n_slots": int(len(set(g[g > 0].tolist())))}
        idx = np.flatnonzero(g > 0)
        if len(idx) > MAX_FRAMES:
            idx = np.sort(rng.choice(idx, MAX_FRAMES, replace=False))
        units = {"frame": (l2(h[idx]), g[idx]), "segment": segments(h, g)}
        for name, (e, lab) in units.items():
            s, d = pair_scores(e, lab)
            rec[name] = summarise(s, d)
            for key in (sub, "all"):
                p = pooled.setdefault(key, {}).setdefault(name, [[], []])
                p[0].append(s)
                p[1].append(d)
        cents = {int(k): l2(h[g == k].mean(0)) for k in sorted(set(g[g > 0].tolist()))}
        rec["centroid_cos"] = {f"{a}-{b}": float(cents[a] @ cents[b]) for a in cents for b in cents if a < b}
        per_window.append(rec)
    out = {"src": str(src), "dim": int(H.shape[1]), "seg_frames": SEG,
           "pooled": {k: {n: summarise(np.concatenate(v[0]), np.concatenate(v[1])) for n, v in u.items()}
                      for k, u in pooled.items()},
           "per_window": per_window}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(out, indent=1))
    for k, u in out["pooled"].items():
        for n, r in u.items():
            print(f"{k:4s} {n:8s} same {r['cos_same']:.3f}  diff {r['cos_diff']:.3f}  gap {r['gap']:.3f}  "
                  f"AUC {r['auc']:.3f}  EER {r['eer']:.3f}  (pairs {r['n_same']}/{r['n_diff']})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
