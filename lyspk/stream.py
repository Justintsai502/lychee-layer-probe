"""Vendored unchanged from mpmoshi-oracle/moshi_ms/speaker_stream.py (import paths only).

Causal streaming speaker head: the only chunk loop in phase 1d.

The head sees ``[cache n_slots*K | FIFO F | chunk C]``. ``SpeakerStream`` feeds
feature frames one at a time; a chunk is scored when it closes:

| Mode | ``step(feat_t)`` returns | Lookahead | Audio frame ``a`` is final after step |
|---|---|---|---|
| chunk 4, shift 2 (legacy) | 4 rows every 4th step, ``flush()`` for the tail | 0-3 frames | ``4*((a+2)//4)+3`` (2-5 frames later) |
| chunk 1, shift 0 | 1 row every step | none | ``a`` |

Cache behaviour is a ``CachePolicy``: ``SelfFill`` (the predicted cache),
``GoldCache``, ``RotatedGold`` and ``EmptyCache``. ``run_streams`` runs many
(features, policy) jobs in lockstep, one head call per chunk boundary for all
of them; with one job it builds exactly the sequences the old per-clip loops
built. Probabilities come back in FEATURE time; ``to_audio_time`` is the one
conversion to audio time (row ``t`` -> ``t - label_shift``).
"""

from __future__ import annotations

import numpy as np
import torch

from lyspk.head_spec import HeadSpec


class CachePolicy:
    """Which past feature frames fill each slot's cache block."""

    observes = False

    def bind(self, spec: HeadSpec) -> "CachePolicy":
        self.spec = spec
        return self

    def cache_frames(self, slot: int, c0: int) -> list[int]:
        raise NotImplementedError

    def observe(self, frames: list[int], probs: np.ndarray) -> None:
        """Called with the final probabilities of each scored chunk."""


class SelfFill(CachePolicy):
    """Predicted cache: write a frame to slot k when exactly one slot is >= theta_act and
    its probability is >= theta_write; keep the top K by score, ordered by time."""

    observes = True

    def __init__(self, theta_act: float | None = None, theta_write: float | None = None) -> None:
        self.theta_act = theta_act
        self.theta_write = theta_write

    def bind(self, spec: HeadSpec) -> "SelfFill":
        super().bind(spec)
        self.theta_act = spec.theta_act if self.theta_act is None else float(self.theta_act)
        self.theta_write = spec.theta_write if self.theta_write is None else float(self.theta_write)
        self.cache: list[list[tuple[int, float]]] = [[] for _ in range(spec.n_slots)]
        return self

    def cache_frames(self, slot: int, c0: int) -> list[int]:
        take = sorted(self.cache[slot], key=lambda item: item[0])[-self.spec.cache_k :]
        return [frame for frame, _score in take]

    def observe(self, frames: list[int], probs: np.ndarray) -> None:
        k_max = self.spec.cache_k
        for i, frame in enumerate(frames):
            on = probs[i] >= self.theta_act
            if int(on.sum()) == 1 and float(probs[i].max()) >= self.theta_write:
                k = int(probs[i].argmax())
                others = np.delete(probs[i], k)
                score = float(probs[i, k] * np.prod(1.0 - others))
                self.cache[k].append((frame, score))
                if len(self.cache[k]) > k_max:
                    self.cache[k] = sorted(sorted(self.cache[k], key=lambda item: -item[1])[:k_max])


class GoldCache(CachePolicy):
    """The last K gold single-speaker frames of each slot before ``c0 - fifo`` (feature time)."""

    def __init__(self, y: np.ndarray | None, single: np.ndarray, keep: np.ndarray, source: tuple[int, ...] | None = None) -> None:
        self.single = np.asarray(single)
        self.keep = np.asarray(keep).astype(bool)
        self.source = source

    def cache_frames(self, slot: int, c0: int) -> list[int]:
        limit = c0 - self.spec.fifo
        if limit <= 0:
            return []
        src = slot if self.source is None else self.source[slot]
        idx = np.flatnonzero((self.single[:limit] == src) & self.keep[:limit])
        return idx[-self.spec.cache_k :].tolist()


ROTATION = (2, 0, 1)  # cache slot k gets the gold frames of slot ROTATION[k]; a 3-cycle


class RotatedGold(GoldCache):
    """Gold cache with blocks rotated: cache slot k gets the gold frames of slot ``source[k]``."""

    def __init__(self, single: np.ndarray, keep: np.ndarray, source: tuple[int, ...] = ROTATION) -> None:
        super().__init__(None, single, keep, source=tuple(source))


class EmptyCache(CachePolicy):
    def cache_frames(self, slot: int, c0: int) -> list[int]:
        return []


class SpeakerStream:
    """Causal streaming speaker head. One instance per clip.

    Puts the head in eval mode (dropout off) for its lifetime; ``close()`` or leaving a
    ``with SpeakerStream(...) as s:`` block restores the previous mode.
    """

    def __init__(self, head, spec: HeadSpec, cache_policy: CachePolicy, device=None, feats: np.ndarray | None = None) -> None:
        self.head = head
        self.spec = spec
        self.policy = cache_policy.bind(spec)
        self.device = device if device is not None else next(head.parameters()).device
        self.feats: list[np.ndarray] | np.ndarray = [] if feats is None else feats
        self.done = 0  # frames already scored
        self._was_training = bool(head.training)
        head.eval()

    def close(self) -> None:
        """Restore the head's train/eval mode from before the stream was built."""
        self.head.train(self._was_training)

    def __enter__(self) -> "SpeakerStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _build(self, c0: int, c1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spec = self.spec
        dim = int(np.asarray(self.feats[0]).shape[-1])
        seq = np.zeros((spec.seq_len, dim), dtype=np.float32)
        seg = np.zeros(spec.seq_len, dtype=np.int64)
        pad = np.ones(spec.seq_len, dtype=bool)
        for k in range(spec.n_slots):
            for i, frame in enumerate(self.policy.cache_frames(k, c0)):
                seq[k * spec.cache_k + i] = self.feats[frame]
                pad[k * spec.cache_k + i] = False
        fifo = range(max(0, c0 - spec.fifo), c0)
        base = spec.n_slots * spec.cache_k + (spec.fifo - len(fifo))
        for i, frame in enumerate(fifo):
            seq[base + i] = self.feats[frame]
            seg[base + i] = 1
            pad[base + i] = False
        start = spec.chunk_start
        for i, frame in enumerate(range(c0, c1)):
            seq[start + i] = self.feats[frame]
            seg[start + i] = 2
            pad[start + i] = False
        return seq, seg, pad

    def _finish(self, c0: int, c1: int, prob: np.ndarray) -> list[tuple[int, np.ndarray]]:
        frames = list(range(c0, c1))
        prob = prob[: len(frames)]
        self.policy.observe(frames, prob)
        self.done = c1
        return list(zip(frames, prob))

    def _score(self, c0: int, c1: int) -> list[tuple[int, np.ndarray]]:
        probs = _call_head(self.head, self.spec, [self._build(c0, c1)], self.device)
        return self._finish(c0, c1, probs[0])

    @torch.no_grad()
    def step(self, feat_t) -> list[tuple[int, np.ndarray]]:
        """Feed feature frame t. Returns the (frame, probs[3]) rows that became final now."""
        assert not self.head.training, "SpeakerStream needs the head in eval mode (dropout would change predictions)"
        if isinstance(feat_t, torch.Tensor):
            feat_t = feat_t.detach().float().cpu().numpy()
        if not isinstance(self.feats, list):
            raise RuntimeError("step() needs a stream built without preloaded features")
        self.feats.append(np.asarray(feat_t, dtype=np.float32).reshape(-1))
        t = len(self.feats) - 1
        if (t + 1) % self.spec.chunk == 0:
            return self._score(t + 1 - self.spec.chunk, t + 1)
        return []

    @torch.no_grad()
    def flush(self) -> list[tuple[int, np.ndarray]]:
        """End of clip: score the last partial chunk (legacy behaviour)."""
        if self.done < len(self.feats):
            return self._score(self.done, len(self.feats))
        return []


def _call_head(head, spec: HeadSpec, built: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device) -> np.ndarray:
    """One head call for several built sequences; returns chunk probabilities ``[B, chunk, n_slots]``."""
    seq = torch.tensor(np.stack([b[0] for b in built]), device=device)
    seg = torch.tensor(np.stack([b[1] for b in built]), device=device)
    pad = torch.tensor(np.stack([b[2] for b in built]), device=device)
    logits = head(seq, seg, pad)
    start = spec.chunk_start
    return torch.sigmoid(logits[:, start : start + spec.chunk]).float().cpu().numpy()


@torch.no_grad()
def run_streams(head, spec: HeadSpec, jobs: list[tuple[np.ndarray, CachePolicy]], max_batch: int = 256) -> list[np.ndarray]:
    """Run several (features [T, D], policy) jobs in lockstep. Returns ``[T, n_slots]`` per job, feature time."""
    device = next(head.parameters()).device
    was = head.training
    head.eval()
    streams = [SpeakerStream(head, spec, policy, device, feats=np.asarray(feats, dtype=np.float32)) for feats, policy in jobs]
    outs = [np.zeros((len(s.feats), spec.n_slots), dtype=np.float32) for s in streams]
    longest = max((len(s.feats) for s in streams), default=0)
    for c0 in range(0, longest, spec.chunk):
        live = [j for j, s in enumerate(streams) if c0 < len(s.feats)]
        for lo in range(0, len(live), max_batch):
            part = live[lo : lo + max_batch]
            bounds = [(c0, min(c0 + spec.chunk, len(streams[j].feats))) for j in part]
            probs = _call_head(head, spec, [streams[j]._build(a, b) for j, (a, b) in zip(part, bounds)], device)
            for row, j, (a, b) in zip(probs, part, bounds):
                for frame, p in streams[j]._finish(a, b, row):
                    outs[j][frame] = p
    head.train(was)
    return outs


def run_stream(head, spec: HeadSpec, feats: np.ndarray, cache_policy: CachePolicy) -> np.ndarray:
    """``[T, n_slots]`` probabilities in FEATURE time for one clip."""
    return run_streams(head, spec, [(feats, cache_policy)])[0]


def to_audio_time(probs_feat: np.ndarray, spec: HeadSpec | int, fill=0) -> np.ndarray:
    """Move feature-time row ``t`` to audio-time row ``t - label_shift``. The last ``shift`` audio
    rows have no prediction and get ``fill``. Works for probs, labels, ``single`` (fill -1) and ``keep`` (fill False)."""
    shift = int(spec.label_shift if isinstance(spec, HeadSpec) else spec)
    arr = np.asarray(probs_feat)
    out = np.full_like(arr, fill)
    if shift == 0:
        return arr.copy()
    if shift > 0:
        if len(arr) > shift:
            out[:-shift] = arr[shift:]
    else:
        s = -shift
        if len(arr) > s:
            out[s:] = arr[:-s]
    return out


def require_audio_time(meta: dict, where: str) -> None:
    """Scorers call this on every dump: probabilities must already be in audio time."""
    if meta.get("time_base") != "audio":
        raise SystemExit(
            f"refusing to score {where}: meta.json has time_base={meta.get('time_base')!r}, not 'audio'. "
            "Re-dump with lyspk/evaluate.py, which writes time_base."
        )
