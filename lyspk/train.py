#!/usr/bin/env python3
"""
Train the streaming speaker head on Lychee-FD, with LoRA in the audio encoder and the
trunk (docs/SPK_HEAD_LYCHEE.md has the design and what was re-derived from the
mpmoshi-oracle head, docs/SPK_PREDICTION_ARCH.md there).

One micro-batch = one recording crop (30-60 s by default, starting on a 0.4 s window):

  wave -> LycheeHost.features         enc stack [33, T, 1280], llm stack [L+1, T, 3584]
       -> GroupReadout                 diar [T, 384], con [T, 256]
       -> 24 gold-cache items           [cache 3 x 48 | FIFO 50 | chunk 5] = 199 cells
       -> SortformerHead (8 layers)     3 sigmoids per cell

  loss = arrival-order BCE + 0.5 x PIL on the chunk            (lyspk.items.diar_loss)
       + con_weight  x SupCon on 0.96 s single-speaker anchors (lyspk.contrast)
       + keep_weight x relative drift of trunk layer L (or the adaptor output when L = 0)
                       from the frozen model                   (lyspk.host.keep_loss)

With --dialogue (multi-track manifests from lyspk.mt_data + lyspk.stoken: AMI / IF / SD)
mpmoshi's dialogue loss is back, as Lychee's own: one participant per crop is the agent,
the input is everyone else's tracks summed, the head labels only the others, and the
agent's turns are teacher-forced into the text / stoken / control channels (lyspk.dialog):

       + dlg_weight x (text CE + stoken CE + control CE)       (LycheeHost.dialogue_losses)

The keep term then defaults to 0 (--keep-weight sets it). Without --dialogue the keep
term stands in for it: the LoRA is held to the frozen model at the one tensor everything
downstream reads.

  python -m lyspk.train --dialogue --train work/spk/mt/if_train.turns.jsonl work/spk/mt/ami_train.turns.jsonl \
      --val work/spk/mt/if_test.turns.jsonl --name mt

  python -m lyspk.train --train work/spk/synth/train.jsonl --val work/spk/synth/val.jsonl --name main
  python -m lyspk.train --host tiny --device cpu ... --steps 3      # CPU smoke, no checkpoint

Outputs in work/spk/runs/<name>/: train.jsonl, eval.jsonl, ckpt_step*.pt, ckpt_latest.pt,
run_config.json.

Model deps (lazy): torch, librosa, soundfile, transformers==4.57.x
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, setup_project_cache  # noqa: E402
from lyspk.data import crop_labels, is_multitrack, load_frames, load_manifest, load_turns, n_frames, view  # noqa: E402
from lyspk.labels import assert_arrival, crop_frames  # noqa: E402


def lr_multiplier(step, warmup, total):
    """Linear warmup, then cosine from 1 to 0.1 (mpmoshi p1c_step.lr_multiplier)."""
    if step < warmup:
        return (step + 1) / float(max(warmup, 1))
    progress = min(1.0, (step - warmup) / float(max(total - warmup, 1)))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="manifests; sampled evenly per manifest")
    ap.add_argument("--train-weights", type=float, nargs="+", default=None,
                    help="draw probability per --train manifest (normalised), e.g. 0.4 0.4 0.2")
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--name", default="main")
    ap.add_argument("--run-dir", type=Path, default=None, help="default work/spk/runs/<name>")
    # host
    ap.add_argument("--host", choices=["lychee", "tiny"], default="lychee")
    ap.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/lychee_fd")
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm-layers", type=int, default=14,
                    help="trunk layers read and LoRA'd (0 = encoder only, no trunk at all). Must stay below "
                         "any layer where speaker information derived from these labels is later injected")
    ap.add_argument("--enc-lora-rank", type=int, default=16)
    ap.add_argument("--llm-lora-rank", type=int, default=16)
    ap.add_argument("--adaptor-lora-rank", type=int, default=16)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    # head
    ap.add_argument("--mode", default="lychee_c5", help="HeadSpec preset (lyspk/head_spec.py PRESETS)")
    # optimisation
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--crop-min", type=float, default=30.0, help="seconds")
    ap.add_argument("--crop-max", type=float, default=60.0, help="seconds; shorter recordings are used whole")
    ap.add_argument("--con-weight", type=float, default=0.2)
    ap.add_argument("--keep-weight", type=float, default=None, help="default 1.0, or 0.0 with --dialogue")
    ap.add_argument("--dialogue", action="store_true",
                    help="load all of Lychee and add its own text/stoken/control loss on the agent's turns")
    ap.add_argument("--dlg-weight", type=float, default=1.0)
    # bookkeeping
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-clips", type=int, default=16)
    ap.add_argument("--eval-seconds", type=float, default=60.0)
    ap.add_argument("--save-every", type=int, default=500,
                    help="ckpt_step*.pt (weights only, to pick a step) + ckpt_latest.pt (full state, to resume)")
    ap.add_argument("--keep-last", type=int, default=0, help="keep only the newest N ckpt_step*.pt (0 = all)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path, default=None)
    return ap.parse_args(argv)


def build(cfg):
    import torch

    from lyspk.head_spec import PRESETS
    from lyspk.host import LycheeHost, tiny_host
    from lyspk.readout import GroupReadout
    from lyspk.sortformer_head import SortformerHead

    if not (cfg.code_dir / "LycheeFD.py").exists():
        raise SystemExit(f"{cfg.code_dir}/LycheeFD.py missing: run scripts/fetch_lychee_code.sh")
    if cfg.mode not in PRESETS:
        raise SystemExit(f"--mode {cfg.mode}: choose from {sorted(PRESETS)}")
    spec = PRESETS[cfg.mode]
    if spec.label_shift != 0:
        raise SystemExit(f"{cfg.mode} has label_shift {spec.label_shift}; Lychee's features need 0")
    if cfg.dialogue and cfg.llm_layers < 1:
        raise SystemExit("--dialogue needs --llm-layers >= 1 (the agent tokens enter through the trunk)")
    if cfg.host == "tiny":
        host = tiny_host(cfg.code_dir, llm_layers=min(cfg.llm_layers, 2), grad_ckpt=not cfg.no_grad_ckpt,
                         dialogue=cfg.dialogue)
        host = host.to(cfg.device)
    else:
        from lyprobe.extract_lychee import resolve_model_dir
        cfg.model_dir = resolve_model_dir(cfg.model_dir)
        host = LycheeHost.from_pretrained(cfg.model_dir, cfg.code_dir, cfg.device, cfg.llm_layers,
                                          attn=cfg.attn, grad_ckpt=not cfg.no_grad_ckpt, dialogue=cfg.dialogue)
    lora_config = {"enc_rank": cfg.enc_lora_rank, "llm_rank": cfg.llm_lora_rank,
                   "adaptor_rank": cfg.adaptor_lora_rank, "scaling": cfg.lora_scaling}
    n_lora = host.apply_lora(**lora_config)
    readout = GroupReadout(host.groups()).to(cfg.device)
    head = SortformerHead(384, spec=spec).to(cfg.device)
    tau = torch.nn.Parameter(torch.tensor(0.1, device=cfg.device))
    host_config = {"kind": cfg.host, "model_dir": str(cfg.model_dir), "llm_layers": host.llm_layers,
                   "attn": cfg.attn, "dialogue": bool(cfg.dialogue)}
    return host, readout, head, tau, spec, lora_config, host_config, n_lora


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.keep_weight is None:
        cfg.keep_weight = 0.0 if cfg.dialogue else 1.0
    setup_project_cache()
    import random

    import torch

    from lyspk.dialog import crop_sequence

    from lyspk.contrast import push_queue, segments, supcon
    from lyspk.evaluate import eval_windows, stream_scores, summary_line
    from lyspk.host import keep_loss
    from lyspk.items import diar_loss, sample_batch
    from lyspk.lora import lora_disabled, lora_params, lora_state, load_lora_state

    run = cfg.run_dir or PROJECT_ROOT / "work/spk/runs" / cfg.name
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_sets = [load_manifest(p) for p in cfg.train]
    for p, rows in zip(cfg.train, train_sets):
        if not rows:
            raise SystemExit(f"{p} is empty")
        if cfg.dialogue:
            n_mt = sum(is_multitrack(r) for r in rows)
            n_turns = sum(bool(r.get("turns")) for r in rows)
            print(f"{p}: {len(rows)} rows, {n_mt} multi-track, {n_turns} with agent turns")
    if cfg.dialogue and not any(r.get("turns") for rows in train_sets for r in rows):
        raise SystemExit("--dialogue: no row has turns; run lyspk.stoken on the multi-track manifests first")
    py_rng = random.Random(cfg.seed)
    if cfg.train_weights is not None and len(cfg.train_weights) != len(train_sets):
        raise SystemExit(f"--train-weights has {len(cfg.train_weights)} values for {len(train_sets)} manifests")
    mix = np.asarray(cfg.train_weights or [1.0] * len(train_sets), float)
    mix = mix / mix.sum()
    val_windows = eval_windows(load_manifest(cfg.val), cfg.eval_seconds, cfg.eval_clips,
                               dialogue=cfg.dialogue) if cfg.val else []

    host, readout, head, tau, spec, lora_config, host_config, n_lora = build(cfg)
    enc_lora = [p for n, p in host.named_parameters() if "lora_" in n and n.startswith("encoder.")]
    llm_lora = [p for n, p in host.named_parameters() if "lora_" in n and not n.startswith("encoder.")]
    head_params = list(readout.parameters()) + list(head.parameters()) + [tau]
    groups = [{"params": enc_lora, "lr": cfg.lr_lora, "name": "enc_lora"},
              {"params": head_params, "lr": cfg.lr_head, "name": "head"}]
    if llm_lora:
        groups.insert(1, {"params": llm_lora, "lr": cfg.lr_lora, "name": "llm_lora"})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    n_train = {"lora": sum(p.numel() for p in lora_params(host)),
               "readout": sum(p.numel() for p in readout.parameters()),
               "head": sum(p.numel() for p in head.parameters())}
    print(f"host {host_config['kind']}: trunk layers read {host.llm_layers} | readout groups {host.groups()}")
    print(f"LoRA linears {n_lora} | trainable {n_train} | spec {spec.mode} seq_len {spec.seq_len}")

    step, queue = 0, []
    if cfg.resume:
        blob = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if blob["head_spec"] != spec.to_dict():
            raise SystemExit(f"resume: checkpoint head_spec {blob['head_spec']['mode']} != --mode {spec.mode}")
        load_lora_state(host, blob["lora"])
        readout.load_state_dict(blob["readout"])
        head.load_state_dict(blob["head"])
        with torch.no_grad():
            tau.copy_(torch.tensor(float(blob["tau"])))
        step, queue = int(blob["step"]), list(blob.get("queue", []))
        if "optimizer" in blob:
            opt.load_state_dict(blob["optimizer"])
            rng.bit_generator.state = blob["rng"]
        else:
            print(f"resume: {cfg.resume.name} is weights only (a ckpt_step file): optimizer state starts fresh; "
                  f"use ckpt_latest.pt to continue exactly")
        print(f"resumed {cfg.resume} at step {step}")

    (run / "run_config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(cfg).items()}, "head_spec": spec.to_dict(), "lora": lora_config,
         "host": host_config, "host_meta": getattr(host, "meta", {}), "lora_linears": n_lora,
         "trainable": n_train, "readout_groups": host.groups()}, indent=1))

    def save(final=False):
        """ckpt_latest.pt / ckpt_final.pt: everything resume needs. ckpt_step*.pt: weights only
        (no optimizer / queue / rng), about a third of the size, for picking a step by eval."""
        blob = {"step": step, "head_spec": spec.to_dict(), "label_shift": spec.label_shift,
                "host_config": host_config, "lora_config": lora_config, "lora": lora_state(host),
                "readout_config": readout.config(), "readout": readout.state_dict(), "head": head.state_dict(),
                "tau": float(tau.detach()), "layer_weights": readout.layer_weights()}
        full = dict(blob, optimizer=opt.state_dict(), rng=rng.bit_generator.state, queue=queue)
        tmp = run / "ckpt_latest.pt.tmp"
        torch.save(full, tmp)
        tmp.replace(run / "ckpt_latest.pt")          # a crash mid-save never leaves a broken latest
        if final:
            torch.save(full, run / "ckpt_final.pt")
            print(f"saved {run / 'ckpt_final.pt'}", flush=True)
            return
        torch.save(blob, run / f"ckpt_step{step:06d}.pt")
        if cfg.keep_last > 0:
            for old in sorted(run.glob("ckpt_step*.pt"))[:-cfg.keep_last]:
                old.unlink()
        print(f"saved {run / f'ckpt_step{step:06d}.pt'} (+ ckpt_latest.pt)", flush=True)

    def evaluate():
        if not val_windows:
            return
        res = stream_scores(host, readout, head, spec, val_windows)
        res.update({"step": step, "layer_weights": readout.layer_weights()})
        with open(run / "eval.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"EVAL step {step}: {summary_line(res)}", flush=True)

    def draw():
        rows = train_sets[int(rng.choice(len(train_sets), p=mix))]
        r = rows[int(rng.integers(len(rows)))]
        seq = None
        if is_multitrack(r):
            cand = list(r.get("agents") or [])
            if cfg.dialogue and r.get("turns"):
                have = load_turns(r)
                cand = [a for a in cand if have.get(a)]
            r = view(r, cand[int(rng.integers(len(cand)))] if cand else None)
        f0, f1 = crop_frames(n_frames(r), rng, cfg.crop_min, cfg.crop_max, align=spec.align)
        if cfg.dialogue and r.get("turns"):
            f0, seq, _ = crop_sequence(r["turns"], f0, f1, host.text_fn, host.stoken_delay, rng=py_rng,
                                       align=spec.align)
            if seq is None:
                return None
        if f1 - f0 < 4 * spec.align:
            return None
        y, single, keep, slots, first = crop_labels(r, f0, f1, spec.label_shift)
        if float(y.sum()) <= 0:
            return None
        assert_arrival(y, first, r["id"])
        return r, f0, f1, y, single, keep, slots, first, seq

    host.train()
    readout.train()
    head.train()
    t_start, log = time.time(), open(run / "train.jsonl", "a")
    while step < cfg.steps:
        mult = lr_multiplier(step, cfg.warmup, cfg.steps)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        opt.zero_grad(set_to_none=True)
        peers, parts = [], []
        t0 = time.time()
        done = misses = 0
        while done < cfg.accum:
            got = draw()
            if got is None:
                misses += 1
                if misses > 1000:
                    raise SystemExit("1000 draws in a row gave no usable crop: check the manifests' segments")
                continue
            r, f0, f1, y, single, keep, slots, first, seq = got
            wave = load_frames(r, f0, f1)
            ref = None
            if cfg.keep_weight > 0:
                with torch.no_grad(), lora_disabled():
                    ref = host.features(wave, seq)["keep"]
            out = host.features(wave, seq, losses=cfg.dialogue)
            diar, con = readout(out["stacks"])
            items = sample_batch(y, single, keep, first, rng, spec=spec)
            dloss, dparts = diar_loss(head, diar, items, spec)
            emb, ids = segments(con, single, keep, slots, rng)
            closs, cst = supcon(emb, ids, peers, queue, tau)
            kloss = keep_loss(out["keep"], ref) if ref is not None else diar.sum() * 0.0
            loss = dloss + cfg.con_weight * closs + cfg.keep_weight * kloss
            dlg = {}
            if "dlg" in out:
                d = out["dlg"]
                loss = loss + cfg.dlg_weight * (d["text"][0] + d["stoken"][0] + d["control"][0])
                dlg = {f"dlg_{k}": float(v[0].detach()) for k, v in d.items()}
                dlg.update({f"n_{k}": v[1] for k, v in d.items()})
                dlg["ctrl_acc"] = d["control"][2] / max(d["control"][1], 1)
            (loss / cfg.accum).backward()
            if emb is not None:
                peers.append((emb.detach(), ids))
            parts.append({"loss": float(loss.detach()), "diar_sort": dparts["diar_sort"], "diar_pil": dparts["diar_pil"],
                          "n_items": dparts["n_items"], "con": float(closs.detach()), "con_pos": cst["n_pos"],
                          "con_rel": float(closs.detach()) / cst["chance"] if cst.get("chance") else 0.0,
                          "con_cross": cst.get("n_cross", 0), "keep": float(kloss.detach()),
                          "frames": int(f1 - f0), "source": r["source"], **dlg})
            done += 1
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
        opt.step()
        for e, i in peers:
            push_queue(queue, e, i)
        step += 1
        keys = sorted({k for p in parts for k in p} - {"source"})
        rec = {k: float(np.mean([p[k] for p in parts if k in p])) for k in keys}
        rec.update({"step": step, "lr_mult": mult, "grad_norm": gnorm, "sec": time.time() - t0,
                    "tau": float(tau.detach()), "queue": len(queue), "elapsed_h": (time.time() - t_start) / 3600})
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % cfg.log_every == 0 or step == 1:
            w = readout.layer_weights()
            top = {k: int(np.argmax(v)) for k, v in w.items()}
            print(f"step {step} loss {rec['loss']:.4f} sort {rec['diar_sort']:.4f} pil {rec['diar_pil']:.4f} "
                  f"con {rec['con']:.3f} (/chance {rec['con_rel']:.2f}) keep {rec['keep']:.4f} gn {gnorm:.2f} {rec['sec']:.1f}s "
                  f"top-layer {top}" + (f" | dlg text {rec['dlg_text']:.3f} stoken {rec['dlg_stoken']:.3f} "
                                        f"control {rec['dlg_control']:.3f} acc {rec['ctrl_acc']:.2f}"
                                        if "dlg_text" in rec else ""), flush=True)
        if cfg.eval_every and step % cfg.eval_every == 0:
            evaluate()
        if cfg.save_every and step % cfg.save_every == 0:
            save()
    log.close()
    if not (cfg.eval_every and step % cfg.eval_every == 0 and step > 0) or cfg.resume:
        evaluate()
    save(final=True)


if __name__ == "__main__":
    sys.exit(main())
