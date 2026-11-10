"""Per-frame speaker labels. Every training and eval path imports this module.

Vendored from mpmoshi-oracle/moshi_ms/p1d_labels.py: ``apply_shift``, ``build_labels``
and ``assert_arrival`` are unchanged. ``activity_from_segments`` is
``activity_from_utterances`` with the speaker list taken from the segments. The AMI
helpers that need mpmoshi's vocab are dropped, and ``crop_frames`` replaces
``crop_bounds`` so a crop always starts and ends on a Lychee window (5 frames = 0.4 s).

Order: crop, arrival order inside the crop, keep 3 slots, mask later speakers,
then label shift. ``keep`` is True on frames that are kept.

Hard deps: numpy
"""

from __future__ import annotations

import numpy as np

FPS = 12.5
LABEL_SHIFT = 0          # Lychee: a feature frame has heard its whole 0.4 s window


def apply_shift(y, single, mask, first_frame, shift):
    """``out[t] = in[t - shift]``. First-frame indices move with the labels."""
    shift = int(shift)
    if shift == 0:
        return y.copy(), single.copy(), mask.copy(), list(first_frame)
    y_out = np.zeros_like(y)
    single_out = np.full(len(single), -1, dtype=np.int64)
    mask_out = np.zeros(len(mask), dtype=bool)
    if shift > 0:
        y_out[shift:] = y[:-shift]
        single_out[shift:] = single[:-shift]
        mask_out[shift:] = mask[:-shift]
    else:
        s = -shift
        y_out[:-s] = y[s:]
        single_out[:-s] = single[s:]
        mask_out[:-s] = mask[s:]
    moved = []
    for frame in first_frame:
        if frame is None:
            moved.append(None)
            continue
        index = int(frame) + shift
        moved.append(index if 0 <= index < len(y) else None)
    return y_out, single_out, mask_out, moved


def build_labels(act_users, user_ids, t0, t1, label_shift=LABEL_SHIFT, max_slots=3):
    """Return y, single, keep, slot_to_global, first_frame for one crop."""
    act = np.asarray(act_users, dtype=np.float32)[int(t0):int(t1)]
    if act.ndim != 2:
        raise ValueError(f"activity must be [T, n_users], got {act.shape}")
    ids = [str(x) for x in user_ids]
    if act.shape[1] != len(ids):
        raise ValueError(f"n_users {act.shape[1]} != len(user_ids) {len(ids)}")
    first = []
    for j in range(act.shape[1]):
        on = np.flatnonzero(act[:, j] > 0.5)
        first.append(int(on[0]) if len(on) else None)
    present = [j for j in range(act.shape[1]) if first[j] is not None]
    order = sorted(present, key=lambda j: (first[j], ids[j]))
    slots, masked = order[:max_slots], order[max_slots:]
    y = np.zeros((len(act), 3), dtype=np.float32)
    for k, j in enumerate(slots):
        y[:, k] = (act[:, j] > 0.5).astype(np.float32)
    if masked:
        keep = ~np.any(act[:, masked] > 0.5, axis=1)
    else:
        keep = np.ones(len(act), dtype=bool)
    n_on = y.sum(1)
    single = np.where((n_on == 1) & keep, y.argmax(1), -1).astype(np.int64)
    slot_to_global = [ids[j] for j in slots] + [None] * (3 - len(slots))
    first_frame = [first[j] for j in slots] + [None] * (3 - len(slots))
    y, single, keep, first_frame = apply_shift(y, single, keep, first_frame, label_shift)
    return y, single, keep, slot_to_global, first_frame


def assert_arrival(y, first_frame, clip_id):
    """Crash if slot 0 is not the first speaker inside this crop."""
    used = [(k, int(frame)) for k, frame in enumerate(first_frame) if frame is not None]
    for prev, nxt in zip(used, used[1:]):
        if nxt[1] < prev[1]:
            raise RuntimeError(f"arrival order broken on {clip_id}: {first_frame}")
    for k, frame in used:
        if frame < 0 or frame >= len(y) or float(y[frame, k]) < 0.5:
            raise RuntimeError(f"slot {k} first frame {frame} is not active on {clip_id}")


def activity_from_segments(segments, n_frames):
    """Raster ``[{speaker, start, end}]`` (seconds) at 12.5 Hz -> (act [T, n_spk], speaker ids).

    Same rule as mpmoshi's activity_from_utterances: a frame is on if the segment
    touches it (floor of the start, ceil of the end). Speakers in first-seen order.
    """
    ids = []
    for row in segments:
        s = str(row["speaker"])
        if s not in ids:
            ids.append(s)
    act = np.zeros((int(n_frames), len(ids)), dtype=np.float32)
    index = {s: i for i, s in enumerate(ids)}
    for row in segments:
        start = max(0, int(np.floor(float(row["start"]) * FPS)))
        end = min(int(n_frames), max(start, int(np.ceil(float(row["end"]) * FPS))))
        act[start:end, index[str(row["speaker"])]] = 1.0
    return act, ids


def crop_frames(n_frames, rng, min_s, max_s, align=5):
    """``(f0, f1)``: the whole clip when it fits in ``max_s``, else a random window of
    ``min_s``..``max_s`` seconds. Both ends on multiples of ``align`` frames (one Lychee
    window), so window ``w`` of the crop is window ``f0 // align + w`` of the clip."""
    n_frames = align * (int(n_frames) // align)
    lo, hi = int(min_s * FPS) // align, int(max_s * FPS) // align
    if n_frames <= hi * align:
        return 0, n_frames
    width = align * int(rng.integers(lo, hi + 1))
    start = align * int(rng.integers(0, (n_frames - width) // align + 1))
    return start, start + width
