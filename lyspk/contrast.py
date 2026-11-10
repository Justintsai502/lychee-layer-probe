"""Auxiliary speaker contrastive on the readout's ``con`` projection (md section 7).

The md describes GE2E; the trainer it came from (mpmoshi-oracle phase1d/train_p1d.py
``_contrastive``, used by train_p1d_fix.py) actually runs SupCon, and that is what is
ported here:

  anchors   mean of 12 frames (0.96 s) inside runs where one slot speaks alone,
            up to 4 per slot per clip, L2-normalised
  peers     the earlier micro-batches of the same optimizer step, detached: same global
            speaker id in another recording = positive (the cross-recording positives
            the md says the loss needs)
  queue     the last 1024 anchors of earlier steps, detached: negatives, except rows of
            the anchor's own speaker, which are left out (a change from mpmoshi, see supcon)
  loss      SupCon with a learned temperature tau (init 0.1, clamped to [0.03, 0.5])

Model deps (lazy): torch
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

SEG = 12          # frames per anchor, 0.96 s
PER_SLOT = 4


def segments(con, single, keep, slot_to_global, rng):
    """Anchors from one clip: (emb [n, D] with graph, global ids) or (None, [])."""
    emb, ids = [], []
    single, keep = np.asarray(single), np.asarray(keep).astype(bool)
    for k, gid in enumerate(slot_to_global):
        if not gid:
            continue
        on = np.flatnonzero((single == k) & keep)
        if len(on) < SEG:
            continue
        cuts = np.where(np.diff(on) > 1)[0]
        starts = [int(on[0])] + [int(on[i + 1]) for i in cuts]
        ends = [int(on[i]) for i in cuts] + [int(on[-1])]
        runs = [(a, b) for a, b in zip(starts, ends) if b - a + 1 >= SEG]
        for j in rng.permutation(len(runs))[:PER_SLOT]:
            a, b = runs[int(j)]
            s = int(rng.integers(a, b - SEG + 2))
            emb.append(con[s:s + SEG].mean(0))
            ids.append(str(gid))
    if not emb:
        return None, []
    return F.normalize(torch.stack(emb), dim=-1), ids


def supcon(local, local_ids, peers, queue, tau):
    """Returns (loss, stats). ``peers`` = [(emb, ids)] detached; ``queue`` = [(row, id)]."""
    if local is None:
        return tau * 0.0, {"n_pos": 0, "bank": 0}
    dev = local.device
    n = local.shape[0]
    bank, bank_ids = [local], list(local_ids)
    for e, i in peers:
        bank.append(e.detach().to(dev))
        bank_ids += list(i)
    n_before_queue = len(bank_ids)
    if queue:
        bank.append(torch.stack([r for r, _ in queue]).to(dev))
        bank_ids += [i for _, i in queue]
    bank = torch.cat(bank, 0)
    logits = local @ bank.T / tau.clamp(0.03, 0.5)
    eye = torch.zeros_like(logits, dtype=torch.bool)
    eye[:, :n] = torch.eye(n, dtype=torch.bool, device=dev)
    same = torch.tensor([[a == b for b in bank_ids] for a in local_ids], device=dev)
    # Queue rows are stale, so they are never positives (as in mpmoshi). Unlike mpmoshi, a
    # queue row of the anchor's OWN speaker is dropped instead of counted as a negative:
    # with 20 training voices the 1024-row queue is mostly same-voice rows, and pushing
    # them away fights the positives (on a 3-speaker toy run the term rose 1.4 -> 4.8).
    drop = eye.clone()
    drop[:, n_before_queue:] = same[:, n_before_queue:]
    same[:, n_before_queue:] = False
    pos = same & ~eye
    logits = logits.masked_fill(drop, -1e4)
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    has = pos.any(1)
    stats = {"n_pos": int(has.sum()), "bank": int(bank.shape[0]),
             "n_cross": int(same[:, n:n_before_queue].any(1).sum()) if n_before_queue > n else 0,
             "chance": math.log(max(int(bank.shape[0]) - 1, 1))}
    if not bool(has.any()):
        return local.sum() * 0.0, stats
    loss = (-(log_prob * pos).sum(1)[has] / pos.sum(1)[has]).mean()
    return loss, stats


def push_queue(queue, emb, ids, limit=1024):
    if emb is None:
        return
    for row, spk in zip(emb.detach().cpu(), ids):
        queue.append((row, spk))
    del queue[:max(0, len(queue) - limit)]
