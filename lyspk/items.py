"""Vendored from mpmoshi-oracle/moshi_ms/p1d_items.py; the only change is that
``sample_batch`` floors every chunk start to a multiple of ``spec.align``.

Gold-cache Sortformer items. Cache positions are context and have loss weight 0.

Every layout number comes from a ``HeadSpec``. Chunk 4 supervises ``[t0, t0+3]``
per item; chunk 1 supervises ``[t0]`` only, so frame mode samples 4x as many
items (96, 24 of them centred on first appearances) to supervise the same
number of current frames per clip, scales the FIFO weight by 1/4 (0.0625) so the
chunk:FIFO weight ratio per item stays at legacy's 4 : 12.5, and drops PIL (with one chunk frame the
best-of-6 permutation would forgive every slot error; it is still logged).
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from lyspk.head_spec import LEGACY, HeadSpec
from lyspk.sortformer_head import N_SLOTS

N_ITEMS = LEGACY.n_items  # legacy default, new code takes a HeadSpec
N_NEW = LEGACY.n_new  # legacy default, new code takes a HeadSpec
PERMS = list(itertools.permutations(range(N_SLOTS)))


def frame_weight(y: np.ndarray, first_frame: list) -> np.ndarray:
    """Change ×2 for 1 s, first appearance ×3 for 2 s, product capped at 6."""
    w = np.ones(len(y), dtype=np.float32)
    if len(y) == 0:
        return w
    prev = np.concatenate([np.zeros((1, y.shape[1]), dtype=y.dtype), y[:-1]], axis=0)
    changed = np.any(np.abs(y - prev) > 0.5, axis=1)
    for t in np.flatnonzero(changed).tolist():
        w[t : t + 12] *= 2.0
    for frame in first_frame:
        if frame is None:
            continue
        w[int(frame) : int(frame) + 25] *= 3.0
    return np.minimum(w, 6.0)


def _blank(spec: HeadSpec = LEGACY) -> dict[str, np.ndarray]:
    n = spec.seq_len
    return {
        "src": np.full(n, -1, dtype=np.int64),
        "y": np.zeros((n, 3), dtype=np.float32),
        "w": np.zeros(n, dtype=np.float32),
        "seg": np.zeros(n, dtype=np.int64),
        "pad": np.ones(n, dtype=bool),
    }


def _write(block, start, src, y, w, seg) -> None:
    n = len(src)
    block["src"][start : start + n] = src
    block["y"][start : start + n] = y
    block["w"][start : start + n] = w
    block["seg"][start : start + n] = seg
    block["pad"][start : start + n] = False


def sample_item(y, single, keep, weights, rng, t0: int, young: bool | None = None, spec: HeadSpec = LEGACY):
    """Chunk ``[t0, t0+chunk)``; FIFO ``[t0-fifo, t0)`` at weight ×``spec.fifo_weight``; cache before ``t0-fifo`` at weight 0."""
    length = len(y)
    if t0 < 0 or t0 >= length:
        return None
    block = _blank(spec)
    limit = t0 - spec.fifo
    cursor = 0
    use_young = bool(rng.random() < 0.2) if young is None else bool(young)
    for slot in range(spec.n_slots):
        idx = np.flatnonzero((single[: max(limit, 0)] == slot) & keep[: max(limit, 0)])
        take = idx
        if len(idx) >= 4 and use_young:
            k = int(rng.integers(4, min(spec.cache_k, len(idx)) + 1))
            take = np.sort(rng.choice(idx, size=k, replace=False))
        elif len(idx) > spec.cache_k:
            take = np.sort(rng.choice(idx, size=spec.cache_k, replace=False))
        if len(take):
            _write(block, cursor, take, y[take], np.zeros(len(take), np.float32), 0)
        cursor += spec.cache_k
    fifo = np.arange(max(0, t0 - spec.fifo), t0)
    if len(fifo):
        pos = cursor + (spec.fifo - len(fifo))
        fw = weights[fifo] * spec.fifo_weight
        fw = np.where(keep[fifo], fw, 0.0)
        _write(block, pos, fifo, y[fifo], fw, 1)
    cursor += spec.fifo
    chunk = np.arange(t0, min(length, t0 + spec.chunk))
    if len(chunk):
        cw = np.where(keep[chunk], weights[chunk], 0.0)
        _write(block, cursor, chunk, y[chunk], cw, 2)
    if float(block["w"][block["seg"] == 2].sum()) <= 0:
        return None
    if float(block["w"][block["seg"] == 0].sum()) != 0.0:
        raise RuntimeError("cache positions were given a nonzero loss weight")
    return block


def sample_batch(y, single, keep, first_frame, rng, n_items: int | None = None, spec: HeadSpec = LEGACY):
    weights = frame_weight(y, first_frame)
    n_items = spec.n_items if n_items is None else int(n_items)
    length = len(y)
    windows = []
    for frame in first_frame:
        if frame is None:
            continue
        lo, hi = int(frame), min(length, int(frame) + 25)
        if hi > lo:
            windows.append((lo, hi))
    n_new = min(spec.n_new, n_items) if windows else 0
    items = []
    for i in range(n_items):
        if i < n_new:
            lo, hi = windows[i % len(windows)]
            t0 = int(rng.integers(lo, hi))
        elif i < n_new + int(round(0.8 * (n_items - n_new))) and length > spec.fifo + 1:
            t0 = int(rng.integers(spec.fifo, length))
        else:
            t0 = int(rng.integers(0, max(length, 1)))
        align = max(1, int(getattr(spec, "align", 1)))
        t0 = align * (t0 // align)
        block = sample_item(y, single, keep, weights, rng, t0, young=None, spec=spec)
        if block is not None:
            items.append(block)
    return items


def diar_loss(head, feat: torch.Tensor, items: list[dict], spec: HeadSpec = LEGACY) -> tuple[torch.Tensor, dict]:
    """``feat`` is ``[T, 384]`` and keeps its graph. Items only store frame indices.

    Loss = arrival-order BCE + ``spec.pil_weight`` × best-of-6 PIL on the chunk. PIL is
    always computed and logged as ``diar_pil``, also when its weight is 0.
    """
    if not items:
        zero = feat.sum() * 0.0
        return zero, {"diar_sort": 0.0, "diar_pil": 0.0, "sum_cache_weight": 0.0, "n_items": 0, "diar_w_chunk_share": None}
    shares = []
    for b in items:
        cw = float(b["w"][b["seg"] == 2].sum())
        tot = cw + float(b["w"][b["seg"] == 1].sum())
        if tot > 0:
            shares.append(cw / tot)
    device = feat.device
    src = torch.tensor(np.stack([b["src"] for b in items]), device=device)
    gathered = feat.index_select(0, src.clamp(min=0).reshape(-1)).reshape(src.shape[0], src.shape[1], -1)
    pad = torch.tensor(np.stack([b["pad"] for b in items]), device=device)
    x = gathered.masked_fill(pad.unsqueeze(-1), 0)
    y = torch.tensor(np.stack([b["y"] for b in items]), device=device)
    w = torch.tensor(np.stack([b["w"] for b in items]), device=device)
    seg = torch.tensor(np.stack([b["seg"] for b in items]), device=device)
    pad = torch.tensor(np.stack([b["pad"] for b in items]), device=device)
    logits = head(x, seg, pad)
    cache_w = float(w[:, : spec.n_slots * spec.cache_k].sum())
    if cache_w != 0.0:
        raise RuntimeError(f"sum of cache weights is {cache_w}")
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(-1)
    sort = (per * w).sum() / w.sum().clamp_min(1e-6)
    chunk = seg == 2
    pil_terms = []
    for perm in PERMS:
        target = y[:, :, list(perm)]
        err = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)
        weight = w * chunk.float()
        pil_terms.append((err * weight).sum() / weight.sum().clamp_min(1e-6))
    pil = torch.stack(pil_terms).min()
    loss = sort + float(spec.pil_weight) * pil if spec.pil_weight else sort + 0.0 * pil
    return loss, {"diar_sort": float(sort.detach()), "diar_pil": float(pil.detach()), "sum_cache_weight": cache_w, "n_items": len(items),
                  "diar_w_chunk_share": float(np.mean(shares)) if shares else None}
