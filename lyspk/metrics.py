"""Streaming speaker-head scores on arrays (feature time = audio time here: label_shift 0).

Definitions follow mpmoshi-oracle:
  single_acc   moshi_ms/p1d_metrics.headline: frames where exactly one gold slot is on
               (and kept); right when exactly that slot is >= theta and no other.
  change_acc   phase1d/diag_metrics.per_window: the gold id (bitmask of active slots)
               differs from the previous frame, both kept, the new id is not silence;
               right when the predicted id equals it. Holding the previous speaker is
               wrong by construction, which is why it is reported separately.
  missed / false_alarm / confusion over active kept frames, and der = their sum.
  rotation_follow  RotatedGold minus GoldCache single_acc after relabelling the
               rotated run: > 0 means the head follows the cache contents (identity),
               ~0 means it ignores them (md section 9).

Hard deps: numpy
"""

from __future__ import annotations

import numpy as np

COUNTS = ("n_single", "single_ok", "change_n", "change_ok", "n_active", "missed", "false_alarm", "confusion")


def ids_of(act):
    a = np.asarray(act).astype(np.int64)
    return a[:, 0] | (a[:, 1] << 1) | (a[:, 2] << 2)


def counts(prob, y, keep, theta=0.30, relabel=None):
    """Counts for one clip. ``relabel[k]`` = which gold slot predicted slot k stands for."""
    keep = np.asarray(keep).astype(bool)
    pred = np.asarray(prob) >= float(theta)
    if relabel is not None:
        moved = np.zeros_like(pred)
        for k, g in enumerate(relabel):
            moved[:, g] = pred[:, k]
        pred = moved
    gold = np.asarray(y) > 0.5
    one = keep & (gold.sum(1) == 1)
    pred_one = np.where(pred.sum(1) == 1, pred.argmax(1), -1)
    ok = one & (pred_one == gold.argmax(1))
    gid, pid = ids_of(gold), ids_of(pred)
    chg = np.zeros(len(gid), bool)
    chg[1:] = (gid[1:] != gid[:-1]) & keep[1:] & keep[:-1] & (gid[1:] != 0)
    g_on, p_on = keep & gold.any(1), keep & pred.any(1)
    return {"n_single": int(one.sum()), "single_ok": int(ok.sum()),
            "change_n": int(chg.sum()), "change_ok": int((chg & (pid == gid)).sum()),
            "n_active": int(g_on.sum()), "missed": int((g_on & ~pred.any(1)).sum()),
            "false_alarm": int((p_on & ~gold.any(1)).sum()),
            "confusion": int((g_on & pred.any(1) & np.any(pred != gold, 1)).sum())}


def pool(rows):
    tot = {k: sum(r[k] for r in rows) for k in COUNTS}
    r = lambda a, b: None if tot[b] == 0 else tot[a] / tot[b]  # noqa: E731
    out = {"single_acc": r("single_ok", "n_single"), "change_acc": r("change_ok", "change_n"),
           "missed": r("missed", "n_active"), "false_alarm": r("false_alarm", "n_active"),
           "confusion": r("confusion", "n_active")}
    parts = [out["missed"], out["false_alarm"], out["confusion"]]
    out["der"] = None if None in parts else sum(parts)
    out.update(tot)
    return out
