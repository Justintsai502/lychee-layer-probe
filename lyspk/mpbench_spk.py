#!/usr/bin/env python3
"""Speaker-id accuracy on the MP-Bench 100-clip subset (mpbench_mp100_eval_kit) for a Lychee checkpoint.

Only part 2 of the kit (speaker-id). Part 1 (MP-Bench answers, GPT-5 judge) needs Lychee to speak,
and the speaker head is a side branch that does not drive Lychee's output, so it is not run here.

Per task: tasks/data/<task>/all_stitched.wav (resampled to 16 kHz, padded to whole 0.4 s windows) ->
host.features(wave, None) (the plain listening suffix: the agent has said nothing during the input,
which is the "silenced" condition of lyspk.diag) -> readout -> head, causally, under these caches:

  self   SelfFill (the deployable setting; the number to quote)
  gold   GoldCache from the kit's gold labels (oracle)
  empty  no cache

Probabilities -> bits at theta_act -> the kit's ids (0 none, 1/2/3 chair, 4 "1,2", 5 "1,3", 6 "2,3",
7 "1,2,3"), one per 80 ms frame in audio time (label_shift moved out), cut to the gold length.
Chairs are not permuted by the kit: head slot k must be the k-th speaker heard, which is the
head's arrival-order training target.

Writes into --out: preds_<variant>/<task>.npz (key ids; the kit's --pred-dir), probs.npz (float16,
for offline theta sweeps), per_task.json, summary.json, and score_<variant>_w{1,12}.txt -- the kit's
own scripts/score_spk.py output, which is the official number.

  python -m lyspk.mpbench_spk --ckpt work/spk/runs/mt/ckpt_step001200.pt \\
      --kit work/mpbench_mp100_eval_kit --audio-root /work/kcire3twcc/datasets/MP_Bench_data \\
      --out work/spk/runs/mt/mp100_ckpt_step001200 [--limit N]

Model deps (lazy): torch (+ the host's deps). Audio: soundfile.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT  # noqa: E402

FPS = 12.5
VARIANTS = ("self", "gold", "empty")
# bitmask (slot0 = 1, slot1 = 2, slot2 = 4) -> the kit's id
MASK_TO_KIT = np.array([0, 1, 2, 4, 3, 5, 6, 7], dtype=np.int64)
KIT_TO_BITS = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)], bool)


def kit_ids(prob: np.ndarray, theta: float) -> np.ndarray:
    on = np.asarray(prob) >= theta
    mask = on[:, 0].astype(np.int64) | (on[:, 1].astype(np.int64) << 1) | (on[:, 2].astype(np.int64) << 2)
    return MASK_TO_KIT[mask]


def gold_tracks(spk: np.ndarray, T: int):
    """Kit gold ids (0..3, no overlap) -> y [T, 3], single [T] (slot or -1), keep [T], padded to T."""
    spk = np.asarray(spk, np.int64)
    n = min(len(spk), T)
    y = np.zeros((T, 3), np.float32)
    y[:n] = KIT_TO_BITS[spk[:n]]
    single = np.full(T, -1, np.int64)
    s = spk[:n]
    single[:n] = np.where((s >= 1) & (s <= 3), s - 1, -1)
    keep = np.zeros(T, bool)
    keep[:n] = True
    return y, single, keep


def find_wav(root: Path, task: str, rel: str | None) -> Path:
    cands = ([root / rel] if rel else []) + [root / "tasks" / "data" / task / "all_stitched.wav",
                                              root / "tasks" / task / "all_stitched.wav", root / task / "all_stitched.wav"]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(f"all_stitched.wav for {task} not under {root}")


def load_wave(path: Path, n_frames: int, align: int):
    """16 kHz mono, exactly f1 * 1280 samples with f1 = n_frames rounded up to whole chunks."""
    from lyspk.data import _read_mono
    f1 = align * -(-n_frames // align)
    return _read_mono(str(path), 0, f1), f1


def score(pred: np.ndarray, gold: np.ndarray) -> dict:
    """The kit's frame-level rules (score_spk.py, window 1), as counts."""
    n = min(len(pred), len(gold))
    p, g = np.clip(pred[:n], 0, 7), np.asarray(gold[:n], np.int64)
    hit = p == g
    sp = g != 0
    out = {"n": int(n), "ok": int(hit.sum()), "n_speech": int(sp.sum()), "ok_speech": int(hit[sp].sum())}
    pb, gb = KIT_TO_BITS[p], KIT_TO_BITS[g]
    for k in range(3):
        out[f"tp{k + 1}"] = int((pb[:, k] & gb[:, k]).sum())
        out[f"on_p{k + 1}"] = int(pb[:, k].sum())
        out[f"on_g{k + 1}"] = int(gb[:, k].sum())
    return out


def pool(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    if not s:
        return {}
    res = {"n_tasks": len(rows), "frames": s["n"], "spk_acc": s["ok"] / max(1, s["n"]),
           "excl_silence": s["ok_speech"] / max(1, s["n_speech"])}
    for k in (1, 2, 3):
        res[f"chair{k}_P"] = s[f"tp{k}"] / max(1, s[f"on_p{k}"])
        res[f"chair{k}_R"] = s[f"tp{k}"] / max(1, s[f"on_g{k}"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True, help="mpbench_mp100_eval_kit (subset/, gold_speakers/, scripts/)")
    ap.add_argument("--audio-root", type=Path, required=True, help="MP-Bench release root (tasks/data/<task>/all_stitched.wav)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--theta-act", type=float, default=None)
    ap.add_argument("--theta-write", type=float, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--device", default="cuda")
    cfg = ap.parse_args(argv)
    import torch

    from lyspk.evaluate import build_from_blob
    from lyspk.stream import EmptyCache, GoldCache, SelfFill, run_streams, to_audio_time

    t0 = time.time()
    out = cfg.out or cfg.ckpt.with_name(f"mp100_{cfg.ckpt.stem}")
    out.mkdir(parents=True, exist_ok=True)
    meta = list(csv.DictReader(open(cfg.kit / "subset" / "instances.tsv"), delimiter="\t"))
    rel = {}
    ent = cfg.kit / "subset" / "entries.jsonl"
    if ent.is_file():
        for line in ent.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                rel[e.get("task_dir_name")] = e.get("input_audio_file")
    if cfg.limit:
        meta = meta[: cfg.limit]
    missing = [r["task"] for r in meta if not (cfg.kit / "gold_speakers" / f"{r['task']}.npz").is_file()]
    if missing:
        raise SystemExit(f"gold labels missing for {len(missing)} tasks, e.g. {missing[:2]}")
    wavs = {}
    for r in meta:
        try:
            wavs[r["task"]] = find_wav(cfg.audio_root, r["task"], rel.get(r["task"]))
        except FileNotFoundError as e:
            raise SystemExit(f"{e} -- check --audio-root")

    blob = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    host, readout, head, spec = build_from_blob(blob, cfg.device, cfg.code_dir, cfg.model_dir)
    theta_act = spec.theta_act if cfg.theta_act is None else cfg.theta_act
    theta_write = spec.theta_write if cfg.theta_write is None else cfg.theta_write
    host.eval()
    readout.eval()
    head.eval()
    print(f"step {blob.get('step')}  head={spec.mode}  theta_act={theta_act} theta_write={theta_write}  "
          f"{len(meta)} tasks  audio {cfg.audio_root}", flush=True)

    per_task, probs_dump = [], {}
    for i, r in enumerate(meta):
        task = r["task"]
        gold = np.asarray(np.load(cfg.kit / "gold_speakers" / f"{task}.npz")["spk"], np.int64).reshape(-1)
        wave, T = load_wave(wavs[task], len(gold), spec.align)
        with torch.no_grad():
            feats = readout(host.features(wave, None)["stacks"])[0].float().cpu().numpy()
        T = min(T, len(feats))
        feats = feats[:T]
        y, single, keep = gold_tracks(gold, T)
        jobs = [(feats, SelfFill(theta_act, theta_write)), (feats, GoldCache(y, single, keep)), (feats, EmptyCache())]
        probs = run_streams(head, spec, jobs)
        rec = {"task": task, "family": r["family"], "should_speak": r["should_speak"],
               "input_dur_s": float(r["input_dur_s"]), "n_gold": int(len(gold))}
        for name, p in zip(VARIANTS, probs):
            pa = to_audio_time(p, spec)[: len(gold)]
            if len(pa) < len(gold):
                pa = np.concatenate([pa, np.zeros((len(gold) - len(pa), pa.shape[1]), pa.dtype)])
            ids = kit_ids(pa, theta_act)
            d = out / f"preds_{name}"
            d.mkdir(exist_ok=True)
            np.savez_compressed(d / f"{task}.npz", ids=ids.astype(np.int16))
            probs_dump[f"{name}__{task}"] = pa.astype(np.float16)
            rec[name] = score(ids, gold)
        probs_dump[f"gold__ids__{task}"] = gold.astype(np.int8)
        per_task.append(rec)
        print(f"  [{i + 1}/{len(meta)}] {task} {len(gold) / FPS:.1f}s  "
              + "  ".join(f"{n} {rec[n]['ok'] / max(1, rec[n]['n']):.3f}" for n in VARIANTS), flush=True)

    np.savez_compressed(out / "probs.npz", **probs_dump)
    (out / "per_task.json").write_text(json.dumps(per_task, indent=1))
    fams = sorted({r["family"] for r in per_task})
    summary = {"ckpt": str(cfg.ckpt), "step": blob.get("step"), "head_mode": spec.mode,
               "theta_act": theta_act, "theta_write": theta_write, "n_tasks": len(per_task),
               "agent_channel": "plain listening suffix (no agent tokens)", "time_base": "audio",
               "by_variant": {n: pool([r[n] for r in per_task]) for n in VARIANTS},
               "by_family": {f: {n: pool([r[n] for r in per_task if r["family"] == f]) for n in VARIANTS} for f in fams}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    scorer = cfg.kit / "scripts" / "score_spk.py"
    for name in VARIANTS:
        for w in (1, 12):
            res = subprocess.run([sys.executable, str(scorer), "--pred-dir", str(out / f"preds_{name}"),
                                  "--window", str(w), "--kit", str(cfg.kit)], capture_output=True, text=True)
            txt = res.stdout + (res.stderr if res.returncode else "")
            (out / f"score_{name}_w{w}.txt").write_text(txt)
            print(f"\n== kit score_spk.py  cache={name}  window={w}\n{txt.strip()}", flush=True)
    print(f"\n{'variant':7s} {'spk acc':>8s} {'excl sil':>9s}   chair1 P/R    chair2 P/R    chair3 P/R")
    for n in VARIANTS:
        s = summary["by_variant"][n]
        print(f"{n:7s} {100 * s['spk_acc']:7.2f}% {100 * s['excl_silence']:8.2f}%   "
              + "   ".join(f"{100 * s[f'chair{k}_P']:4.1f}/{100 * s[f'chair{k}_R']:4.1f}" for k in (1, 2, 3)))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")
    return summary


if __name__ == "__main__":
    main()
