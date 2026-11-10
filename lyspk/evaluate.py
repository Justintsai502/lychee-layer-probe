#!/usr/bin/env python3
"""
Score a checkpoint by streaming the head over held-out recordings (md section 6).

Four cache policies on the same features (lyspk/stream.py):
  self    SelfFill, the deployment path (theta_act 0.30 / theta_write 0.70 by default)
  gold    GoldCache, the last 48 gold single-speaker frames of each slot
  rot     RotatedGold, gold frames given to the wrong slots (is the cache read at all?)
  empty   EmptyCache, the floor

  python -m lyspk.evaluate --ckpt work/spk/runs/main/ckpt_latest.pt --manifest work/spk/synth/val.jsonl
      [--seconds 60] [--clips 0 = all] [--dump work/spk/eval_dump]

Writes <ckpt dir>/eval_<ckpt stem>.json (and per-clip probabilities in audio time with
--dump). Thresholds: --theta-act / --theta-write, else the checkpoint's spec defaults;
they are NOT tuned (md section 9: tune on held-out data before comparing checkpoints).

Model deps (lazy): torch (+ the host's deps)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT  # noqa: E402
from lyspk.data import crop_labels, load_frames, load_manifest, n_frames  # noqa: E402
from lyspk.metrics import counts, pool  # noqa: E402

ROTATION = (2, 0, 1)      # lyspk.stream.ROTATION


def eval_windows(rows, seconds, max_clips=0, align=5):
    """Fixed windows: the first ``seconds`` of each recording, cut to whole Lychee windows."""
    out = []
    for r in rows[:max_clips or None]:
        f1 = align * (min(n_frames(r), int(seconds * 12.5)) // align)
        if f1 >= 4 * align:
            out.append((r, 0, f1))
    return out


def stream_scores(host, readout, head, spec, windows, theta_act=None, theta_write=None, dump=None):
    import torch

    from lyspk.stream import EmptyCache, GoldCache, RotatedGold, SelfFill, run_streams

    theta_act = spec.theta_act if theta_act is None else float(theta_act)
    theta_write = spec.theta_write if theta_write is None else float(theta_write)
    rows = {k: [] for k in ("self", "gold", "rot", "rot_asis", "empty")}
    was = (host.training, readout.training, head.training)
    host.eval()
    readout.eval()
    try:
        for r, f0, f1 in windows:
            y, single, keep, _slots, _first = crop_labels(r, f0, f1, spec.label_shift)
            with torch.no_grad():
                diar = readout(host.features(load_frames(r, f0, f1))["stacks"])[0].float().cpu().numpy()
            jobs = {"self": SelfFill(theta_act, theta_write), "gold": GoldCache(None, single, keep),
                    "rot": RotatedGold(single, keep), "empty": EmptyCache()}
            probs = dict(zip(jobs, run_streams(head, spec, list((diar, p) for p in jobs.values()))))
            for k, p in probs.items():
                rows[k].append(counts(p, y, keep, theta_act, relabel=ROTATION if k == "rot" else None))
            rows["rot_asis"].append(counts(probs["rot"], y, keep, theta_act))
            if dump is not None:
                dump.mkdir(parents=True, exist_ok=True)
                np.savez(dump / f"{r['id']}.npz", y=y, keep=keep, single=single, **probs)
    finally:
        host.train(was[0])
        readout.train(was[1])
        head.train(was[2])
    out = {k: pool(v) for k, v in rows.items()}
    g, rr = out["gold"]["single_acc"], out["rot"]["single_acc"]
    out["rotation_follow"] = None if g is None or rr is None else rr - out["rot_asis"]["single_acc"]
    out["theta_act"], out["theta_write"], out["n_windows"] = theta_act, theta_write, len(windows)
    if dump is not None:
        (dump / "meta.json").write_text(json.dumps({"time_base": "audio", "label_shift": spec.label_shift,
                                                    "policies": list(rows)[:4]}, indent=1))
    return out


def summary_line(res):
    f = lambda v: "  -  " if v is None else f"{v:.3f}"  # noqa: E731
    return (f"self single {f(res['self']['single_acc'])} change {f(res['self']['change_acc'])} der {f(res['self']['der'])} | "
            f"gold single {f(res['gold']['single_acc'])} change {f(res['gold']['change_acc'])} | "
            f"empty single {f(res['empty']['single_acc'])} change {f(res['empty']['change_acc'])} | "
            f"rotation_follow {f(res['rotation_follow'])}")


def build_from_blob(blob, device, code_dir, model_dir=None, host_kind=None):
    """Host (+ LoRA weights), readout and head exactly as the checkpoint was trained."""
    import torch

    from lyspk.head_spec import HeadSpec
    from lyspk.host import LycheeHost, tiny_host
    from lyspk.lora import load_lora_state
    from lyspk.readout import GroupReadout
    from lyspk.sortformer_head import SortformerHead

    hc, lc = blob["host_config"], blob["lora_config"]
    kind = host_kind or hc["kind"]
    if kind == "tiny":
        host = tiny_host(code_dir, llm_layers=hc["llm_layers"]).to(device)
    else:
        host = LycheeHost.from_pretrained(model_dir or hc["model_dir"], code_dir, device, hc["llm_layers"],
                                          attn=hc.get("attn", "sdpa"))
    host.apply_lora(**lc)
    load_lora_state(host, blob["lora"])
    rc = blob["readout_config"]
    readout = GroupReadout({k: tuple(v) for k, v in rc["groups"].items()}, rc["diar_dim"], rc["con_dim"]).to(device)
    readout.load_state_dict(blob["readout"])
    spec = HeadSpec.from_dict(blob["head_spec"])
    head = SortformerHead(rc["diar_dim"], spec=spec).to(device)
    head.load_state_dict(blob["head"])
    torch.set_grad_enabled(False)
    return host, readout, head, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--clips", type=int, default=0)
    ap.add_argument("--theta-act", type=float, default=None)
    ap.add_argument("--theta-write", type=float, default=None)
    ap.add_argument("--model-dir", type=Path, default=None, help="override the path stored in the checkpoint")
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    cfg = ap.parse_args()
    import torch

    blob = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    host, readout, head, spec = build_from_blob(blob, cfg.device, cfg.code_dir, cfg.model_dir)
    windows = eval_windows(load_manifest(cfg.manifest), cfg.seconds, cfg.clips, spec.align)
    res = stream_scores(host, readout, head, spec, windows, cfg.theta_act, cfg.theta_write, cfg.dump)
    res.update({"ckpt": str(cfg.ckpt), "step": blob.get("step"), "manifest": str(cfg.manifest),
                "seconds": cfg.seconds, "head_spec": blob["head_spec"]})
    out = cfg.out or cfg.ckpt.with_name(f"eval_{cfg.ckpt.stem}.json")
    out.write_text(json.dumps(res, indent=1))
    print(summary_line(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
