"""Vendored unchanged from mpmoshi-oracle/moshi_ms/sortformer_head.py (import path only).
The trainer uses ``SortformerHead``; the legacy samplers and GE2E below are kept as-is.

Sortformer head used by phase 1c.

Sequence is [cache 3×K | FIFO F | chunk C]. Segment embeddings mark those
three regions. There is no slot-id embedding. Outputs are 3 independent sigmoids.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_CACHE = 48  # legacy default, new code takes a HeadSpec
F_FIFO = 50  # legacy default, new code takes a HeadSpec
C_CHUNK = 4  # legacy default, new code takes a HeadSpec
N_SLOTS = 3
D_MODEL = 384
N_LAYERS = 8
N_HEADS = 8
FFN = 1536
SEQ_LEN = N_SLOTS * K_CACHE + F_FIFO + C_CHUNK  # legacy default, new code takes a HeadSpec
PERMS = list(itertools.permutations(range(N_SLOTS)))
CHANGE_FRAMES = 12  # 1 second at 12.5 Hz
GE2E_WIN = 6  # 0.5 s at 12.5 Hz
GE2E_MAX_SPK = 64
GE2E_UTT = 4
N_ITEMS = 24  # legacy default, new code takes a HeadSpec
# Cache frames are grouped by slot, so their label is given away by position.
CACHE_W = 0.0


def reorder_arrival(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Permute slot columns so slot 0 is the first speaker active in this window.

    Slots that never turn on in the window stay all-zero and are ordered last.
    Returns ``(reordered [T, K], order)`` where ``order[new] = old``.
    """
    y = np.asarray(y, dtype=np.float32)
    first = []
    for k in range(y.shape[1]):
        on = np.flatnonzero(y[:, k] > 0.5)
        first.append(int(on[0]) if len(on) else 10**9)
    order = np.argsort(np.asarray(first), kind="stable")
    return y[:, order], order


def apply_label_shift(y: np.ndarray, shift: int) -> np.ndarray:
    """``out[t] = y[t - shift]``. Positive shift uses an earlier label at each feature frame."""
    y = np.asarray(y)
    shift = int(shift)
    if shift == 0:
        return y.copy()
    out = np.zeros_like(y)
    if shift > 0:
        out[shift:] = y[:-shift]
    else:
        s = -shift
        out[:-s] = y[s:]
    return out


def change_weight(y: np.ndarray) -> np.ndarray:
    """Frames within 1 s after an activity change are weighted ×2."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=np.float32)
    if len(y) == 0:
        return w
    prev = np.concatenate([np.zeros((1, y.shape[-1]), dtype=y.dtype), y[:-1]], axis=0)
    changed = np.any(y != prev, axis=-1)
    for i in np.flatnonzero(changed).tolist():
        w[i : i + CHANGE_FRAMES] = 2.0
    return w


def _blank(n: int, dim: int) -> dict[str, np.ndarray]:
    return {
        "x": np.zeros((n, dim), dtype=np.float32),
        "y": np.zeros((n, N_SLOTS), dtype=np.float32),
        "w": np.zeros(n, dtype=np.float32),
        "seg": np.zeros(n, dtype=np.int64),
        "pad": np.ones(n, dtype=bool),
        "src": np.full(n, -1, dtype=np.int64),
    }


def _write(block: dict[str, np.ndarray], start: int, x, y, w, seg: int, src) -> None:
    n = len(x)
    block["x"][start : start + n] = x
    block["y"][start : start + n] = y
    block["w"][start : start + n] = w
    block["seg"][start : start + n] = seg
    block["pad"][start : start + n] = False
    block["src"][start : start + n] = np.asarray(src, dtype=np.int64)


def sample_gold_item(
    feat: np.ndarray,
    y: np.ndarray,
    single: np.ndarray,
    rng: np.random.Generator,
    t0: int | None = None,
    force_young: bool | None = None,
) -> dict[str, np.ndarray] | None:
    """One Sortformer item. Cache is gold single-speaker frames before the FIFO.

    Cache-frame weights are multiplied by ``CACHE_W`` (default 0) so a full cache
    is not free loss. FIFO is ×0.5 and the chunk is ×1. With probability 0.2 the
    cache is young (K' drawn from 4..K) when that slot has at least 4 eligible frames.
    """
    length = int(feat.shape[0])
    if length < 2 or float(np.asarray(y).sum()) <= 0:
        return None
    dim = int(feat.shape[-1])
    t0 = int(rng.integers(0, length) if t0 is None else t0)
    if t0 < 0 or t0 >= length:
        return None
    weights = change_weight(y)
    block = _blank(SEQ_LEN, dim)
    cursor = 0
    limit = t0 - F_FIFO
    for slot in range(N_SLOTS):
        idx = np.flatnonzero(single[: max(limit, 0)] == slot)
        take = idx
        young = force_young if force_young is not None else float(rng.random()) < 0.2
        if len(idx) >= 4 and young:
            k = int(rng.integers(4, K_CACHE + 1))
            if len(idx) > k:
                take = np.sort(rng.choice(idx, size=k, replace=False))
        elif len(idx) > K_CACHE:
            take = np.sort(rng.choice(idx, size=K_CACHE, replace=False))
        if len(take) > 0:
            _write(block, cursor, feat[take], y[take], weights[take] * CACHE_W, 0, take)
        cursor += K_CACHE
    f0 = max(0, t0 - F_FIFO)
    fifo = np.arange(f0, t0)
    fifo_pos = cursor + (F_FIFO - len(fifo))
    if len(fifo):
        _write(block, fifo_pos, feat[fifo], y[fifo], weights[fifo] * 0.5, 1, fifo)
    cursor += F_FIFO
    chunk = np.arange(t0, min(length, t0 + C_CHUNK))
    if len(chunk):
        _write(block, cursor, feat[chunk], y[chunk], weights[chunk], 2, chunk)
    if float(block["w"][block["seg"] == 2].sum()) <= 0:
        return None
    block["t0"] = np.asarray(t0, dtype=np.int64)
    return block


def sample_item_batch(
    feat: np.ndarray,
    y: np.ndarray,
    single: np.ndarray,
    rng: np.random.Generator,
    n_items: int = N_ITEMS,
    ignore: np.ndarray | None = None,
) -> tuple[list[dict[str, np.ndarray]], int]:
    """``n_items`` gold-cache items. At least 80% use ``t0 >= F_FIFO`` when that is possible.

    Returns the items and how many of them have a non-empty cache.
    """
    length = int(feat.shape[0])
    if length < 2 or float(np.asarray(y).sum()) <= 0:
        return [], 0
    if ignore is not None:
        y = np.array(y, copy=True)
        single = np.array(single, copy=True)
        y[np.asarray(ignore).astype(bool)] = 0
        single[np.asarray(ignore).astype(bool)] = -1
    n_cache = int(round(0.8 * n_items))
    items: list[dict[str, np.ndarray]] = []
    nonempty = 0
    for i in range(n_items):
        if i < n_cache and length > F_FIFO + 1:
            t0 = int(rng.integers(F_FIFO, length))
        else:
            t0 = int(rng.integers(0, max(length, 1)))
        block = sample_gold_item(feat, y, single, rng, t0=t0, force_young=False)
        if block is None:
            continue
        if np.any(block["src"][: N_SLOTS * K_CACHE] >= 0):
            nonempty += 1
        items.append(block)
    return items, nonempty


class SortformerHead(nn.Module):
    """Transformer over [cache | FIFO | chunk]. ``spec`` sets the sequence length (chunk 4: 198, chunk 1: 195).

    Chunk 1 needs no causal mask: the only predicted position is the current frame and
    every other position is past (FIFO) or cache. Chunk 4 keeps full attention inside
    the chunk (the known 0-3 frame lookahead).

    ``slot_id_dim`` and ``seg_id_dim`` decide whether what a cell *is* shares the
    frame's dimensions or gets its own. At 0 each is a full-width vector added to the
    frame: the region through ``seg_emb``, and the chair not written anywhere at all,
    since cache block k holds slot k and the answer is the position -- the same
    ``pos_emb`` that orders the FIFO, one vector doing two jobs, and the reason the
    cache positions carry no loss (``CACHE_W``). Above 0 they are appended in reserved
    dimensions and ``in_proj`` narrows to match, so the width is unchanged and a read
    of those dimensions recovers the id exactly, with no dependence on the frame.
    Summed, no such read exists. ``pos_emb`` goes with the chair id, so the FIFO loses
    its ordering: that is what the trade costs.

    Cell layout, LEGACY versus the id presets::

        0            [ in_proj(x) 384 + seg_emb 384 + pos_emb 384 ]
        id4 / seg4   [ in_proj(x) 376 | slot 4 | seg 4 ]
    """

    def __init__(self, dim_in: int = 384, spec=None) -> None:
        super().__init__()
        from lyspk.head_spec import LEGACY

        self.spec = LEGACY if spec is None else spec
        self.id_dim = int(getattr(self.spec, "slot_id_dim", 0))
        self.seg_dim = int(getattr(self.spec, "seg_id_dim", 0))
        self.content_dim = D_MODEL - self.id_dim - self.seg_dim
        if self.id_dim < 0 or self.seg_dim < 0 or self.content_dim < 1:
            raise ValueError(
                f"slot_id_dim {self.id_dim} + seg_id_dim {self.seg_dim} leaves no room in {D_MODEL}"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.in_proj = nn.Linear(dim_in, self.content_dim)
        self.seg_emb = nn.Embedding(3, self.seg_dim or D_MODEL)
        if self.id_dim:
            # One row per chair plus one for "no chair": the FIFO and the chunk.
            self.id_emb = nn.Embedding(N_SLOTS + 1, self.id_dim)
            self.pos_emb = None
            self.register_buffer("slot_ids", self._slot_ids(), persistent=False)
        else:
            self.id_emb = None
            self.pos_emb = nn.Embedding(self.spec.seq_len + 8, D_MODEL)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False)
        self.out = nn.Linear(D_MODEL, N_SLOTS)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def _slot_ids(self) -> torch.Tensor:
        """``[seq_len]``: slot k on cache block k, ``N_SLOTS`` ("no chair") on FIFO and chunk."""
        spec = self.spec
        ids = torch.full((spec.seq_len,), N_SLOTS, dtype=torch.long)
        for k in range(spec.n_slots):
            ids[k * spec.cache_k : (k + 1) * spec.cache_k] = k
        return ids

    def _cells(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """``[B, T, dim_in]`` -> the encoder's input ``[B, T, D_MODEL]``."""
        bsz, length, _ = x.shape
        if self.id_dim:
            # slot_ids is this spec's layout. A sequence of another length would
            # put the FIFO and the chunk under cache ids without saying so.
            if length != len(self.slot_ids):
                raise RuntimeError(
                    f"this head's layout is {len(self.slot_ids)} cells ({self.spec.mode}), got {length}"
                )
            ids = self.slot_ids[:length].unsqueeze(0).expand(bsz, length)
            parts = [self.in_proj(x), self.id_emb(ids)]
        else:
            pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, length)
            parts = [self.in_proj(x) + self.pos_emb(pos)]
        if self.seg_dim:
            h = torch.cat(parts + [self.seg_emb(seg)], dim=-1)
        else:
            h = torch.cat(parts, dim=-1) + self.seg_emb(seg)
        return h

    def forward(self, x: torch.Tensor, seg: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self._cells(x, seg), src_key_padding_mask=pad)
        return self.out(h)


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)
    return (per * weight).sum() / weight.sum().clamp_min(1e-6)


def sortformer_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, seg: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Arrival-order BCE on supervised frames, plus 0.5 × best-of-6 PIL on the chunk."""
    sort = _masked_bce(logits, target, weight)
    chunk_w = weight * (seg == 2).to(weight.dtype)
    if float(chunk_w.sum()) <= 0:
        pil = logits.sum() * 0.0
    else:
        best = None
        for perm in PERMS:
            cand = _masked_bce(logits, target[..., list(perm)], chunk_w)
            best = cand if best is None else torch.minimum(best, cand)
        pil = best
    total = sort + 0.5 * pil
    return total, {"sort": float(sort.detach()), "pil": float(pil.detach()), "diar": float(total.detach())}


def sample_ge2e(
    clips: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    max_spk: int = GE2E_MAX_SPK,
    n_utt: int = GE2E_UTT,
    win: int = GE2E_WIN,
) -> np.ndarray | None:
    """Mean-pooled windows for GE2E. Returns [speakers, utterances, dim], or None.

    ``clips`` is a list of ``(emb [T, D], spk [T])``. ``spk < 0`` is not a
    single-speaker frame. The same speaker id in different clips is one speaker.
    """
    buckets: dict[int, list[tuple[int, int, int]]] = {}
    for clip_i, (_emb, spk) in enumerate(clips):
        spk = np.asarray(spk)
        t = 0
        n = int(spk.shape[0])
        while t < n:
            if int(spk[t]) < 0:
                t += 1
                continue
            start = t
            who = int(spk[t])
            while t < n and int(spk[t]) == who:
                t += 1
            if t - start >= win:
                buckets.setdefault(who, []).append((clip_i, start, t))
    speakers = [s for s, spans in buckets.items() if sum((e - a) // win for _i, a, e in spans) >= n_utt]
    if len(speakers) < 2:
        return None
    if len(speakers) > max_spk:
        speakers = list(rng.choice(np.asarray(speakers), size=max_spk, replace=False))
    rows = []
    for who in speakers:
        starts: list[tuple[int, int]] = []
        for clip_i, a, e in buckets[int(who)]:
            starts.extend((clip_i, s) for s in range(a, e - win + 1, win))
        pick = rng.choice(len(starts), size=n_utt, replace=False)
        pooled = []
        for j in pick:
            clip_i, s = starts[int(j)]
            pooled.append(clips[clip_i][0][s : s + win].mean(axis=0))
        rows.append(np.stack(pooled).astype(np.float32))
    return np.stack(rows)


class GE2ELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb [speakers, utterances, dim], already L2-normalised.

        The speaker's own centroid leaves out the utterance being scored.
        Other speakers use the mean of all of their utterances.
        """
        n_spk, n_utt, _ = emb.shape
        if n_spk < 2 or n_utt < 2:
            return emb.sum() * 0.0
        total = emb.sum(dim=1)
        own = F.normalize((total.unsqueeze(1) - emb) / float(n_utt - 1), dim=-1)
        sim_own = (emb * own).sum(dim=-1)
        sim_all = torch.einsum("nmd,kd->nmk", emb, F.normalize(total, dim=-1))
        idx = torch.arange(n_spk, device=emb.device)
        sim_all[idx, :, idx] = sim_own
        logits = self.w * sim_all + self.b
        target = idx.view(n_spk, 1).expand(n_spk, n_utt)
        return F.cross_entropy(logits.reshape(-1, n_spk), target.reshape(-1))
