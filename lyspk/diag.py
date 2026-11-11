#!/usr/bin/env python3
"""mpmoshi's speaker diagnostic battery (phase1d/diag_stage1.py) on a Lychee checkpoint.

Rules: mpmoshi-oracle docs/SPK_SCORING_RULES.md and docs/SPK_EVALUATION.md. The scoring code is
mpmoshi's own, vendored unchanged: lyspk/diag_metrics.py (frame / merged turn / change, paired
session bootstrap) and lyspk/bank_sets.py (the twelve fixed banks). The cache policies are
lyspk/stream.py (vendored speaker_stream.py). What is new here is only the front end: windows,
features and labels from this repo's host.

One pass scores every variant on the same features, because the head is cheap next to the trunk:

  live policies  self (SelfFill), gold (GoldCache), rotated (RotatedGold), empty, silenced
  fixed banks    first, one turn, recent, late half, spread, wrong, any voice, one xK, one only,
                 clean, edge   ("empty" is the same as the live empty, so it is scored once)

and every standing pair of both modes (gold(live) -> gold, self(live) -> self).

Differences from diag_stage1.py, all forced by the host:
  * silenced: mpmoshi replaces the agent's audio stream with silence codes. Lychee hears no agent
    audio (the input is the other participants' tracks); the agent exists only as its
    teacher-forced text / speech tokens. So here silenced = self with the agent's turns removed
    (the plain listening suffix). Dropped for a non-dialogue checkpoint, where it equals self.
  * gold_h1 is absent: it is a gold run on features from a held-out voice-prompt text, and this
    host has no voice prompt.
  * windows come from this repo's held-out turns manifests (work/spk/mt/{ami_eval,if_test}.turns.jsonl),
    one agent per window drawn among the agents with turns. Group (the bootstrap unit) = recording.
  * the boundary / per-slot headline and hspk_geometry are not ported; der (direct-slot, not a
    toolkit DER) is added from lyspk.metrics.

  python -m lyspk.diag --ckpt work/spk/runs/mt/ckpt_final.pt \\
      --manifest ami=work/spk/mt/ami_eval.turns.jsonl --manifest tt=work/spk/mt/if_test.turns.jsonl \\
      [--sessions 8] [--windows-per-session 2] [--window-sec 60] [--seed 0] [--boot 10000] [--limit N]

Writes into --out (default <ckpt dir>/diag_<ckpt stem>/): summary.json, per_window.json, meta.json
(time_base audio), preds.npz, hspk.npz.

Model deps (lazy): torch (+ the host's deps)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT  # noqa: E402
from lyspk.bank_sets import NAMES as BANK_NAMES  # noqa: E402
from lyspk.bank_sets import build_sets, spread_stats  # noqa: E402
from lyspk.data import crop_labels, is_multitrack, load_frames, load_manifest, load_turns, n_frames, view  # noqa: E402
from lyspk.diag_metrics import boot, per_window, ratio  # noqa: E402
from lyspk.metrics import counts  # noqa: E402

FPS = 12.5
LIVE = ("self", "gold", "rotated", "empty", "silenced")
BANKS = tuple(n for n in BANK_NAMES if n != "empty")
PAIRS = (
    ("gold", "empty"),        # what the cache is worth at all
    ("gold", "rotated"),      # live: the part only a correct cache gives (identity)
    ("rotated", "empty"),     # live: the part a wrong cache also gives (occupancy)
    ("self", "gold"),         # the cost of filling the cache from its own output
    ("gold", "silenced"),     # what the agent's own channel is worth
    ("gold", "wrong"),        # banks: identity
    ("wrong", "empty"),       # banks: occupancy
    ("first", "gold"),        # earliest evidence against the sliding latest
    ("one xK", "empty"),      # one frame, copied to fill the slot's share of attention
    ("one turn", "spread"),
    ("spread", "recent"),
    ("recent", "one xK"),
    ("one xK", "one only"),   # the same evidence, with and without the room it fills
    ("clean", "edge"),        # mid-turn frames vs frames at a handover
    ("spread", "any voice"),  # this speaker vs anyone speaking
)
METRICS = (("frame", "frame_ok", "n_single"),
           ("turn", "turn_mode_ok", "turn_n"),      # the turn-level verdict, the one to quote
           ("turn-ex", "turn_exact_ok", "turn_n"),
           ("turn-held", "turn_ok", "turn_n"),      # never quote alone
           ("change", "change_ok", "change_n"))
DER_KEYS = ("n_active", "missed", "false_alarm", "confusion")
SUBSETS = ("all", "ami", "tt")
LABEL = {"ami": "AMI", "tt": "IF", "all": "POOLED"}


def draw_windows(manifests, sessions, per_session, window_sec, seed, align=5):
    """diag_stage1.test_windows on this repo's rows. Per subset, ``sessions`` recordings drawn without
    replacement (all when there are fewer). A recording longer than 1.5 windows gives
    ``per_session`` windows spaced evenly through it with seeded jitter (an AMI meeting); a
    shorter one gives one window from its start (an IF scene). Frames snap to whole chunks."""
    rng = np.random.default_rng(seed)
    out = []
    for sub, path in manifests:
        rows = sorted((r for r in load_manifest(path) if is_multitrack(r) and r.get("turns")), key=lambda r: r["id"])
        if not rows:
            raise SystemExit(f"{path}: no multi-track rows with turns (run scripts/prep_spk_mt.sh)")
        pick = range(len(rows)) if len(rows) <= sessions else sorted(rng.choice(len(rows), sessions, replace=False))
        for i in pick:
            r = rows[int(i)]
            have = load_turns(r)
            cand = [a for a in (r.get("agents") or []) if have.get(a)]
            if not cand:
                continue
            agent = cand[int(rng.integers(len(cand)))]
            dur = float(r["duration"])
            width = min(window_sec, dur)
            if width <= 1.0:
                continue
            n_win = 1 if dur <= window_sec * 1.5 else per_session
            span = max(0.0, dur - width)
            for k in range(n_win):
                frac = 0.0 if n_win == 1 else k / (n_win - 1 + 1e-9)
                jitter = float(rng.random()) * span / max(n_win * 2, 1)
                t0 = min(span, frac * span * (n_win - 1) / max(n_win, 1) + jitter)
                f0 = align * (int(round(t0 * FPS)) // align)
                f1 = min(f0 + align * (int(width * FPS) // align), align * (n_frames(r) // align))
                if f1 - f0 >= 4 * align:
                    out.append({"row": r, "agent": agent, "subset": sub, "f0": f0, "f1": f1,
                                "group": f"{sub}:{r['id']}"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--manifest", action="append", required=True, metavar="SUBSET=PATH",
                    help="held-out turns manifest per subset; ami / tt are reported apart (repeatable)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sessions", type=int, default=8, help="recordings per subset")
    ap.add_argument("--windows-per-session", type=int, default=2)
    ap.add_argument("--window-sec", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (of recordings)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N windows (smoke)")
    ap.add_argument("--theta-act", type=float, default=None)
    ap.add_argument("--theta-write", type=float, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--device", default="cuda")
    cfg = ap.parse_args(argv)
    import torch

    from lyspk.dialog import crop_sequence
    from lyspk.evaluate import build_from_blob
    from lyspk.stream import ROTATION, CachePolicy, EmptyCache, GoldCache, RotatedGold, SelfFill, run_streams, to_audio_time

    class FixedBank(CachePolicy):
        """diag_stage1.FixedBank: a bank pinned once per clip (an oracle)."""

        def __init__(self, frames):
            self.frames = [list(f) for f in frames]

        def cache_frames(self, slot, c0):
            return self.frames[slot][: self.spec.cache_k]

    t_start = time.time()
    out_dir = cfg.out or cfg.ckpt.with_name(f"diag_{cfg.ckpt.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for m in cfg.manifest:
        sub, _, path = m.partition("=")
        if not path:
            raise SystemExit(f"--manifest wants SUBSET=PATH, got {m!r}")
        manifests.append((sub, Path(path)))

    blob = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    host, readout, head, spec = build_from_blob(blob, cfg.device, cfg.code_dir, cfg.model_dir)
    theta_act = spec.theta_act if cfg.theta_act is None else cfg.theta_act
    theta_write = spec.theta_write if cfg.theta_write is None else cfg.theta_write
    dialogue = bool(getattr(host, "dialogue", False))
    variants = tuple(v for v in LIVE if dialogue or v != "silenced") + BANKS
    pairs = tuple(p for p in PAIRS if p[0] in variants and p[1] in variants)

    windows = draw_windows(manifests, cfg.sessions, cfg.windows_per_session, cfg.window_sec, cfg.seed, spec.align)
    if cfg.limit:
        windows = windows[: cfg.limit]
    if not windows:
        raise SystemExit("no windows to score")
    n_by = Counter(w["subset"] for w in windows)
    print(f"step {blob.get('step')}  head={spec.mode}  theta_act={theta_act} theta_write={theta_write}  "
          f"cache_k={spec.cache_k}  {len(windows)} windows ({dict(n_by)}), {len(variants)} variants", flush=True)

    host.eval()
    readout.eval()
    head.eval()

    def feats_of(wave, seq):
        with torch.no_grad():
            return readout(host.features(wave, seq)["stacks"])[0].float().cpu().numpy()

    clips = []
    for i, w in enumerate(windows):
        v = view(w["row"], w["agent"])
        f0, f1, seq = w["f0"], w["f1"], None
        if dialogue:
            f0, seq, _ = crop_sequence(v["turns"], f0, f1, host.text_fn, host.stoken_delay,
                                       align=spec.align, train=False)
            if seq is None:
                print(f"  [{i + 1}/{len(windows)}] {v['id']} skipped: an agent turn covers the window", flush=True)
                continue
        y, single, keep, _slots, _first = crop_labels(v, f0, f1, spec.label_shift)
        wave = load_frames(v, f0, f1)
        clip = {"session": v["id"], "dataset": w["subset"], "group": w["group"], "f0": f0, "f1": f1,
                "y": y, "single": single, "keep": keep, "diar": feats_of(wave, seq)}
        if dialogue:
            clip["diar_s"] = feats_of(wave, None)      # the agent's turns removed: plain listening suffix
        clips.append(clip)
        print(f"  [{i + 1}/{len(windows)}] {v['id']} {f0 / FPS:.0f}-{f1 / FPS:.0f}s T={f1 - f0}", flush=True)
    if not clips:
        raise SystemExit("every window was skipped")
    t_feat = time.time()

    jobs = []
    for c in clips:
        sets = build_sets(c["single"], c["keep"], spec.n_slots, spec.cache_k)
        c["bank_spread"] = {n: [spread_stats(f) for f in sets[n]] for n in BANKS}
        live = {"self": (c["diar"], SelfFill(theta_act, theta_write)),
                "gold": (c["diar"], GoldCache(c["y"], c["single"], c["keep"])),
                "rotated": (c["diar"], RotatedGold(c["single"], c["keep"], ROTATION)),
                "empty": (c["diar"], EmptyCache())}
        if dialogue:
            live["silenced"] = (c["diar_s"], SelfFill(theta_act, theta_write))
        jobs += [live[n] if n in live else (c["diar"], FixedBank(sets[n])) for n in variants]
    probs = run_streams(head, spec, jobs)
    t_stream = time.time()

    rows = {v: [] for v in variants}
    dump = {}
    for ci, c in enumerate(clips):
        y_a = to_audio_time(c["y"], spec)
        keep_a = to_audio_time(c["keep"], spec, fill=False)
        for vi, name in enumerate(variants):
            p = to_audio_time(probs[ci * len(variants) + vi], spec)
            der = counts(p, y_a, keep_a, theta_act)
            rows[name].append({**per_window(p, y_a, keep_a, theta_act), **{k: der[k] for k in DER_KEYS},
                               "window": ci, "dataset": c["dataset"], "session": c["session"], "group": c["group"]})
            dump[f"prob_{name}_{ci}"] = p.astype(np.float32)
        dump[f"y_{ci}"] = y_a.astype(np.int8)
        dump[f"keep_{ci}"] = keep_a.astype(bool)
    np.savez_compressed(out_dir / "preds.npz", **dump, windows=np.asarray([c["session"] for c in clips]),
                        datasets=np.asarray([c["dataset"] for c in clips]), variants=np.asarray(variants))
    H, G, W, F, D = [], [], [], [], []
    for ci, c in enumerate(clips):          # hspk.npz in scripts/hspk_geometry.py's layout (feature time)
        s = np.asarray(c["single"])[: len(c["diar"])]
        k = np.asarray(c["keep"]).astype(bool)[: len(c["diar"])]
        H.append(c["diar"].astype(np.float32))
        G.append(np.where(k & (s >= 0), s + 1, 0).astype(np.int16))
        W.append(np.full(len(s), ci, np.int32))
        F.append(np.arange(len(s), dtype=np.int32))
        D.append(np.full(len(s), c["dataset"]))
    np.savez_compressed(out_dir / "hspk.npz", h_spk=np.concatenate(H), gold=np.concatenate(G),
                        window=np.concatenate(W), frame=np.concatenate(F), dataset=np.concatenate(D))

    def sel(name, sub):
        return [r for r in rows[name] if sub == "all" or r["dataset"] == sub]

    def pooled(name, ok, n, sub):
        rs = sel(name, sub)
        return ratio(sum(r[ok] for r in rs), sum(r[n] for r in rs))

    subsets = [s for s in SUBSETS if s == "all" or n_by.get(s)] + sorted(set(n_by) - set(SUBSETS))
    summary = {
        "ckpt": str(cfg.ckpt), "step": blob.get("step"), "head_mode": spec.mode,
        "theta_act": theta_act, "theta_write": theta_write, "label_shift": spec.label_shift,
        "cache_k": spec.cache_k, "dialogue": dialogue, "manifests": {s: str(p) for s, p in manifests},
        "n_windows": len(clips), "seed": cfg.seed, "window_sec": cfg.window_sec,
        "sessions": cfg.sessions, "windows_per_session": cfg.windows_per_session,
        "n_windows_by": dict(Counter(c["dataset"] for c in clips)),
        "n_groups_by": {**{s: len({c["group"] for c in clips if c["dataset"] == s}) for s in n_by},
                        "all": len({c["group"] for c in clips})},
        "time_base": "audio", "boot_iters": cfg.boot, "feat_sec": t_feat - t_start, "stream_sec": t_stream - t_feat,
        "bank_spread": {n: {k: (lambda v: float(np.mean(v)) if v else None)(
            [st[k] for c in clips for st in c["bank_spread"][n] if st[k] is not None]) for k in ("span_sec", "gap_sec", "n")}
            for n in BANKS},
        "by_variant": {}, "paired": {},
    }
    for sub in subsets:
        summary["by_variant"][sub] = {}
        for name in variants:
            rs = sel(name, sub)
            tot = {k: sum(r[k] for r in rs) for k in DER_KEYS}
            parts = [ratio(tot[k], tot["n_active"]) for k in ("missed", "false_alarm", "confusion")]
            summary["by_variant"][sub][name] = {
                **{f"{m}_acc": pooled(name, ok, n, sub) for m, ok, n in METRICS},
                "turn_first_acc": pooled(name, "turn_first_ok", "turn_first_n", sub),
                "der": None if None in parts else sum(parts),
                "missed": parts[0], "false_alarm": parts[1], "confusion": parts[2],
                "n_single": sum(r["n_single"] for r in rs), "n_turns": sum(r["turn_n"] for r in rs),
                "n_change": sum(r["change_n"] for r in rs)}
        summary["paired"][sub] = {}
        for a, b in pairs:
            ra, rb = sel(a, sub), sel(b, sub)
            summary["paired"][sub][f"{a}-{b}"] = {
                m: boot([(x[ok], x[n], z[ok], z[n]) for x, z in zip(ra, rb)], cfg.boot, groups=[x["group"] for x in ra])
                for m, ok, n in METRICS}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "per_window.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out_dir / "meta.json").write_text(json.dumps(
        {"step": blob.get("step"), "head_mode": spec.mode, "ckpt": str(cfg.ckpt), "n_windows": len(clips),
         "theta_act": theta_act, "time_base": "audio", "label_shift": spec.label_shift}, indent=2) + "\n")

    f = lambda v: "  --  " if v is None else f"{v:.4f}"  # noqa: E731
    cols = [s for s in ("ami", "tt", "all") if s in summary["by_variant"]]
    print(f"\n{'variant':<11}" + "".join(f"{LABEL[s]:>28}" for s in cols))
    print(f"{'':11}" + "".join(f"{'frame / turn / change':>28}" for _ in cols))
    for name in variants:
        cells = ""
        for s in cols:
            v = summary["by_variant"][s][name]
            cells += f"{f(v['frame_acc'])} / {f(v['turn_acc'])} / {f(v['change_acc'])}".rjust(28)
        print(f"{name:<11}{cells}")
    for s in cols:
        v = summary["by_variant"][s][variants[0]]
        print(f"  {LABEL[s]}: {summary['n_windows_by'].get(s, len(clips)) if s != 'all' else len(clips)} windows / "
              f"{summary['n_groups_by'].get(s, 0)} recordings, {v['n_single']} single frames, "
              f"{v['n_turns']} turns, {v['n_change']} change frames")
    print(f"\nder (self): " + "  ".join(f"{LABEL[s]} {f(summary['by_variant'][s]['self']['der'])}" for s in cols))
    print(f"turn_first (self / gold): " + "  ".join(
        f"{LABEL[s]} {f(summary['by_variant'][s]['self']['turn_first_acc'])} / "
        f"{f(summary['by_variant'][s]['gold']['turn_first_acc'])}" for s in cols))
    print(f"\npaired, {cfg.boot} resamples of RECORDINGS, 95% interval (* = does not cross zero)")
    for s in cols:
        print(f"  --- {LABEL[s]}")
        for pair, got in summary["paired"][s].items():
            for m in ("frame", "turn", "change"):
                g = got.get(m) or {}
                if g.get("delta") is None:
                    continue
                if g.get("lo") is None:
                    print(f"    {pair:<20}{m:<8}{g['delta']:+.4f} (one recording: no interval)")
                    continue
                sig = "*" if g["lo"] > 0 or g["hi"] < 0 else " "
                print(f"    {pair:<20}{m:<8}{g['delta']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] {sig}")
    print(f"\nwrote {out_dir}  ({time.time() - t_start:.0f}s: features {t_feat - t_start:.0f}s, "
          f"streams {t_stream - t_feat:.0f}s)")
    torch.set_grad_enabled(True)
    return summary


if __name__ == "__main__":
    main()
