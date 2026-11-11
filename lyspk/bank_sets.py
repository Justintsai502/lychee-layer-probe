"""Vendored unchanged from mpmoshi-oracle/phase1d/bank_sets.py (this line is the only edit).

Ported from the old ``scripts/make_cache_sets.py`` onto this branch's label
space (``single`` in 0..2 with -1 for not-solo, plus ``keep``) instead of the
8-class power-set ids. The selection rules are unchanged.

The point of the set. The shipped rule keeps a slot's ``cache_k`` most recent
solo frames, so "knows this voice" and "heard this voice a moment ago" are the
same condition and no evaluation can separate them. The first four sets hold
the same number of frames of the same slot speaking alone and differ only in
where in the clip they come from:

  first       the first cache_k solo frames              earliest evidence
  one turn    one unbroken run of that slot speaking      most concentrated
  recent      the last cache_k solo frames
  late half   spread over the second half of the clip
  spread      spread over the whole clip                  most dispersed

Those four alone cannot separate "any cache_k frames will do" from "the bank is
not read at all" -- opposite findings that look identical. The controls settle
it:

  wrong       each slot given the next slot's frames
  empty       no frames
  any voice   spread over every frame anyone speaks, slot ignored
  one xK      a single frame of that slot, copied into every entry
  one only    that same frame once, the rest left masked
  clean       only frames well inside a long solo stretch (>= 0.64 s from any change)
  edge        only frames beside a boundary or overlap (<= 0.16 s from one)

Unchanged accuracy under "wrong" means the contents are not used. A collapse
means they are, and then four alike dispersion scores mean the evidence
saturates early. "empty" bounds what the bank contributes at all, and "any
voice" says whether the gain comes from this slot's speech or merely from
having speech in the window. "one xK" holds no more evidence than one frame
while still filling the slot's share of the attention; "one only" holds the
same evidence with the rest masked, so the pair separates the evidence from the
room it takes up. "clean" against "edge" is the same count at the same spread,
differing only in whether the frame sits mid-turn or where the floor changes
hands -- alike scores would say the bank cannot tell a clean example from a
contaminated one.

``first`` and ``recent`` are the two ends of the same axis -- the earliest
frames of a slot against its latest -- and the shipped rule is ``recent``. Alike
scores say the bank holds an identity that does not go stale; a gap says it is
really carrying recent acoustic context, which is a different claim.

Two things to keep in mind when reading the result:

- These banks are pinned once per clip and never change, which is the "fixed
  bank" condition. The shipped ``GoldCache`` instead recomputes the last K solo
  frames before every chunk, so comparing ``recent`` against a live gold run is
  what measures the cost of freezing.
- Indices may point anywhere in the clip, the earliest frames included, so a
  run pinned to one of these is an oracle. The comparison *between* sets is the
  result; their distance from a live run is not.
"""

from __future__ import annotations

import numpy as np

NAMES = ("first", "one turn", "recent", "late half", "spread",
         "wrong", "empty", "any voice", "one xK", "one only",
         "clean", "edge")

CLEAN_MIN = 8   # >= 0.64 s from any change, at 12.5 Hz
EDGE_MAX = 2    # <= 0.16 s from one


def even(idx: np.ndarray, k: int) -> np.ndarray:
    """``k`` indices spread evenly over ``idx``, or all of them when there are too few."""
    idx = np.asarray(idx)
    if idx.size <= k:
        return idx
    return idx[np.linspace(0, idx.size - 1, k).round().astype(int)]


def longest_run(idx: np.ndarray, k: int) -> np.ndarray:
    """``k`` frames from inside the longest unbroken stretch, centred on it."""
    idx = np.asarray(idx)
    if idx.size <= k:
        return idx
    brk = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk + 1, [idx.size]])
    best = int(np.argmax(ends - starts))
    s, e = int(starts[best]), int(ends[best])
    if e - s >= k:                       # the run alone can fill the bank
        mid = (s + e) // 2
        s2 = max(s, min(mid - k // 2, e - k))
        return idx[s2:s2 + k]
    # short of k, grow outward from that run so the frames stay contiguous in time
    need = k - (e - s)
    lo = max(0, s - need // 2)
    hi = min(idx.size, e + (need - (s - lo)))
    lo = max(0, hi - k)
    return idx[lo:hi]


def distance_to_change(solo: np.ndarray) -> np.ndarray:
    """Per frame, how far to the nearest frame that is not this slot alone."""
    n = solo.size
    far = np.full(n, 1 << 20, dtype=np.int64)
    last = -(1 << 20)
    for t in range(n):
        if not solo[t]:
            last = t
        far[t] = t - last
    last = 1 << 20
    for t in range(n - 1, -1, -1):
        if not solo[t]:
            last = t
        far[t] = min(far[t], last - t)
    return far


def build_sets(single: np.ndarray, keep: np.ndarray, n_slots: int, k: int) -> dict[str, list[list[int]]]:
    """``name -> [frame indices per slot]``, every list at most ``k`` long.

    A slot with no solo frame at all gets an empty list, which ``_build`` leaves
    masked -- the same as "empty" for that slot.
    """
    single = np.asarray(single)
    keep = np.asarray(keep).astype(bool)
    n = min(single.size, keep.size)
    single, keep = single[:n], keep[:n]
    solo_of = [np.flatnonzero((single == c) & keep) for c in range(n_slots)]
    out: dict[str, list[list[int]]] = {name: [[] for _ in range(n_slots)] for name in NAMES}

    for c, solo in enumerate(solo_of):
        if solo.size == 0:
            continue
        half = solo[solo >= n // 2]
        out["first"][c] = solo[:k].tolist()
        out["one turn"][c] = longest_run(solo, k).tolist()
        out["recent"][c] = solo[-k:].tolist()
        out["late half"][c] = even(half if half.size >= k else solo, k).tolist()
        out["spread"][c] = even(solo, k).tolist()

        one = int(np.median(longest_run(solo, k)))   # middle of its cleanest run
        out["one xK"][c] = [one] * k
        out["one only"][c] = [one]

        mask = np.zeros(n, dtype=bool)
        mask[solo] = True
        far = distance_to_change(mask)
        clean = solo[far[solo] >= CLEAN_MIN]
        edge = solo[far[solo] <= EDGE_MAX]
        out["clean"][c] = np.sort(even(clean if clean.size else solo, k)).tolist()
        out["edge"][c] = np.sort(even(edge if edge.size else solo, k)).tolist()

    # "wrong" hands each slot the next slot's frames, taken from the most
    # dispersed set so the contents differ only in whose voice they are.
    for c in range(n_slots):
        out["wrong"][c] = list(out["spread"][(c + 1) % n_slots])

    # "any voice" ignores who is speaking: every frame where anyone is solo.
    speech = np.flatnonzero((single >= 0) & keep)
    if speech.size:
        sel = np.sort(even(speech, k)).tolist()
        for c in range(n_slots):
            out["any voice"][c] = list(sel)

    return out


def spread_stats(frames: list[int], frame_rate: float = 12.5) -> dict:
    """Seconds from the earliest to the latest frame, and the mean gap between them."""
    a = np.asarray(sorted(frames))
    if a.size < 2:
        return {"span_sec": None, "gap_sec": None, "n": int(a.size)}
    return {"span_sec": float((a[-1] - a[0]) / frame_rate),
            "gap_sec": float(np.mean(np.diff(a)) / frame_rate),
            "n": int(a.size)}
