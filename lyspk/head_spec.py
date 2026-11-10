"""Speaker-head configuration, stored with the weights.

Vendored from mpmoshi-oracle/moshi_ms/head_spec.py. Changes for Lychee-FD: an
``align`` field (chunk starts on a multiple of ``align`` frames in training, as they
always do when streaming) and the ``LYCHEE*`` presets at the bottom. Everything else
is unchanged.

Two head designs share every code path; a ``HeadSpec`` selects one.

| Mode | Chunk | Label shift | Lookahead in head | Prediction for an audio frame is ready after |
|---|---|---|---|---|
| ``chunk4_s2`` (legacy, s985) | 4 frames, stride 4 | 2 | 0-3 frames inside the chunk | 2-5 frames |
| ``frame_s0`` | 1 frame, stride 1 | 0 | none | 0 frames |
| ``frame_s2`` | 1 frame, stride 1 | 2 | none | 2 frames |

Time bases. The head reads backbone features; feature frame ``t`` is trained to
predict the label of audio frame ``t - label_shift``. Streams return
probabilities in *feature time*; ``lyspk.stream.to_audio_time`` is
the only place that moves them to *audio time* (row ``t`` -> ``t - shift``).

Why shift 2 was the legacy value: with Moshi's codebook delays and the initial
token, hidden position ``t`` has seen the user's semantic codebook up to about
``t - 1`` and the acoustic codebooks up to about ``t - 2``. With shift 0 the
head names the speaker of an audio frame the backbone has not fully heard yet,
which is a ~160 ms forecast at speaker changes.

Readiness. ``ready_step(spec, t)`` is the streaming step after which the
prediction for feature frame ``t`` (audio frame ``t - label_shift``) is final:
chunk 4 waits for the chunk to close, chunk 1 is final at step ``t``.

A checkpoint without ``head_spec`` is legacy (chunk 4, its stored shift).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadSpec:
    mode: str
    chunk: int
    label_shift: int
    fifo: int = 50
    cache_k: int = 48
    n_slots: int = 3
    n_items: int = 24
    n_new: int = 6
    pil_weight: float = 0.5
    theta_act: float = 0.30
    theta_write: float = 0.70
    # FIFO positions are supervised at this x the frame weight. Frame mode uses 0.25/4 so the
    # chunk:FIFO weight ratio per item stays at legacy's 4 : 12.5 (chunk ~24% of the item loss).
    fifo_weight: float = 0.25
    # 0 keeps pos_emb: every cell is content + seg + position, summed, and which chair a cache
    # cell belongs to is carried by where it sits (block k is slot k), the same position that
    # also orders the FIFO. Above 0, pos_emb goes and the last ``slot_id_dim`` dimensions of
    # each cell hold a slot id instead -- 0..n_slots-1 for the cache blocks, n_slots for the
    # FIFO and the chunk, which have no chair. in_proj narrows by that much, so the width is
    # unchanged and identity no longer shares a vector with order.
    slot_id_dim: int = 0
    # 0 keeps seg_emb as a full-width vector added to every cell, so which region a cell
    # sits in shares its dimensions with the frame. Above 0 it is appended instead, in
    # its own ``seg_id_dim`` dimensions, and in_proj narrows by that much as well.
    seg_id_dim: int = 0
    # Training items start their chunk at a multiple of this many frames. Streaming
    # always scores chunks at multiples of ``chunk`` from the clip start, so 1 (legacy:
    # any frame) trains on chunk positions the stream never uses. Lychee sets it to the
    # chunk, 5 frames = one independently encoded 0.4 s window.
    align: int = 1

    @property
    def seq_len(self) -> int:
        return self.n_slots * self.cache_k + self.fifo + self.chunk

    @property
    def chunk_start(self) -> int:
        """Sequence index of the first chunk position: after the cache blocks and the FIFO."""
        return self.n_slots * self.cache_k + self.fifo

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "HeadSpec":
        names = {f.name for f in dataclasses.fields(HeadSpec)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown HeadSpec fields {sorted(unknown)}")
        return HeadSpec(**d)


LEGACY = HeadSpec("chunk4_s2", chunk=4, label_shift=2)
FRAME_S0 = HeadSpec("frame_s0", chunk=1, label_shift=0, n_items=96, n_new=24, pil_weight=0.0, fifo_weight=0.0625)
FRAME_S2 = dataclasses.replace(FRAME_S0, mode="frame_s2", label_shift=2)
# Same as LEGACY, but what a cell *is* leaves the frame's dimensions: four for the
# chair it belongs to, four for the region it sits in, with in_proj narrowed to 376.
# pos_emb goes with the chair id, so the FIFO is no longer ordered.
ID4 = dataclasses.replace(LEGACY, mode="chunk4_s2_id4", slot_id_dim=4, seg_id_dim=4)
FRAME_S0_ID4 = dataclasses.replace(FRAME_S0, mode="frame_s0_id4", slot_id_dim=4, seg_id_dim=4)
# Lychee-FD. Its audio encoder sees each 0.4 s window on its own and emits 5 features
# (12.5 Hz, the same rate as Mimi), so every feature in a window has already heard the
# whole window: label_shift 0, and the natural chunk is the window itself, 5 frames,
# aligned to window boundaries. Lookahead inside the chunk (0-4 frames) is the lookahead
# the encoder already has, so the head adds no latency beyond Lychee's own 0.4 s.
# Cache / FIFO keep their seconds: 48 frames = 3.84 s, 50 frames = 4.0 s.
LYCHEE = HeadSpec("lychee_c5", chunk=5, label_shift=0, align=5)
LYCHEE_ID4 = dataclasses.replace(LYCHEE, mode="lychee_c5_id4", slot_id_dim=4, seg_id_dim=4)
PRESETS = {s.mode: s for s in (LEGACY, FRAME_S0, FRAME_S2, ID4, FRAME_S0_ID4, LYCHEE, LYCHEE_ID4)}


def spec_from_checkpoint(blob: dict) -> HeadSpec:
    """Checkpoints without ``head_spec`` are legacy: chunk 4, shift ``blob['label_shift']`` (2)."""
    if blob.get("head_spec"):
        return HeadSpec.from_dict(dict(blob["head_spec"]))
    return dataclasses.replace(LEGACY, label_shift=int(blob.get("label_shift", LEGACY.label_shift)))


def decode_sidecar(ckpt_path) -> "Path":
    """``<ckpt stem>.decode.json`` next to the checkpoint: tuned thresholds, never written into the blob."""
    from pathlib import Path

    p = Path(ckpt_path)
    return p.with_name(p.stem + ".decode.json")


def file_sha1(path) -> str:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def decode_thresholds(blob: dict, spec: HeadSpec, ckpt_path=None) -> tuple[float, float, str]:
    """``(theta_act, theta_write, source)``. Order: the sidecar (its ``ckpt_sha1`` must match the
    checkpoint file, else stop), then ``blob['decode']``, then the spec."""
    if ckpt_path is not None:
        side = decode_sidecar(ckpt_path)
        if side.is_file():
            import json

            dec = json.loads(side.read_text())
            if dec.get("ckpt_sha1") != file_sha1(ckpt_path):
                raise SystemExit(f"{side} was tuned on a different file than {ckpt_path} (sha1 mismatch)")
            return float(dec["theta_act"]), float(dec["theta_write"]), "sidecar"
    dec = blob.get("decode")
    if dec:
        return float(dec["theta_act"]), float(dec["theta_write"]), "blob"
    return float(spec.theta_act), float(spec.theta_write), "spec"


def ready_step(spec: HeadSpec, frame: int) -> int:
    """Streaming step at which the prediction for FEATURE frame ``frame`` is final.

    That prediction describes audio frame ``frame - spec.label_shift``.
    Chunk 4: the step that closes the chunk, ``4 * (frame // 4) + 3``.
    Chunk 1: ``frame``. Stage 2 can inject it at layer 16 of the same step.
    """
    c = int(spec.chunk)
    return c * (int(frame) // c) + c - 1
