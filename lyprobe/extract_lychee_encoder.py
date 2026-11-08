#!/usr/bin/env python3
"""
Lychee-FD's audio encoder, layer by layer: the part BEFORE the LLM.

Path of the user's audio (LycheeFD.forward + modeling_step_audio_2.py, commit 7bb1068):

  0.4 s window -> 128-bin log-mel (42 frames) -> conv1, conv2 (stride 2) + positions
    -> n_audio_layer Whisper blocks (50 Hz, 1280-d)        enc_0 .. enc_L
    -> AvgPool(2) + LayerNorm (25 Hz)                       enc_pool
    -> Adaptor: conv stride 2 + MLP (12.5 Hz, 3584-d)       adaptor   (= what is added to
                                                             the <audio_patch> slots)

Every 0.4 s window is encoded on its own (no context across windows), exactly as the
model does while streaming. Windows, lead silence and trimming are identical to
extract_lychee.py, so the clips line up with work/features/lychee.npz by utt_id.

Points (depth < 0 = before the LLM; trunk_0 of lychee.npz is depth 0):
  enc_0 .. enc_L, enc_pool, adaptor   depth -(L + 3) .. -1

Output (default work/features/lychee_encoder.npz):
  X_<k>  [N, D] per point (1280-d for enc_*, 3584-d for adaptor), mean over speech frames
  point_names, point_depths, point_track ('encoder'), utt_id, speaker, sentence_id, meta

Only the encoder and adaptor are run, but the whole model is loaded (one checkpoint).

Hard deps: numpy
Model deps (lazy): torch, librosa, transformers==4.57.x, soundfile
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import (PROJECT_ROOT, SIBLING_MANIFEST, load_audio, read_jsonl,  # noqa: E402
                            setup_project_cache, trim_silence)
from lyprobe.extract_lychee import load_lychee, log_mel, resolve_model_dir  # noqa: E402
from lyprobe.layout import CHUNK_S, SR, prepare_waveform, window_frame_mask  # noqa: E402


def save_points(path, names, depths, track, feats, meta_rows, meta):
    """Shared writer for the encoder files: one X_<k> array per point."""
    utt, spk, sent = map(np.array, zip(*meta_rows))
    arrays = {}
    for k, name in enumerate(names):
        X = np.stack([f[k] for f in feats])
        X16 = X.astype(np.float16)
        arrays[f"X_{k}"] = X16 if np.isfinite(X16).all() else X.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, point_names=np.array(names), point_depths=np.array(depths),
             point_track=np.array([track] * len(names)), utt_id=utt, speaker=spk,
             sentence_id=sent, meta=json.dumps(meta), **arrays)
    print(f"wrote {path}: {len(names)} points x {len(utt)} clips")


def encoder_windows(wave):
    """The model's per-0.4 s mel windows (same as extract_lychee.build_inputs)."""
    import torch

    win = int(round(CHUNK_S * SR))
    feats, lens = [], []
    for i in range(0, len(wave), win):
        mel = log_mel(wave[i:i + win])
        feats.append(mel.t())
        lens.append(mel.shape[1] - 2)
    wavs = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).transpose(1, 2)
    return wavs, torch.tensor(lens, dtype=torch.int32)


def run_encoder(model, wavs, wav_lens):
    """-> list of (name, [n_win, frames, D] float32, n_valid_frames, frame_s)."""
    import torch

    enc = model.encoder
    blocks = list(enc.blocks)
    captured = {}
    hooks = [blocks[0].register_forward_pre_hook(
        lambda m, args: captured.__setitem__(0, args[0].detach()))]
    for k, b in enumerate(blocks, start=1):
        hooks.append(b.register_forward_hook(
            lambda m, args, out, k=k: captured.__setitem__(k, out.detach())))
    try:
        dtype = next(enc.parameters()).dtype
        dev = next(enc.parameters()).device
        with torch.no_grad():
            out, out_lens = enc(wavs.to(dev, dtype), wav_lens.to(dev))
            ada = model.adapter(out)
    finally:
        for h in hooks:
            h.remove()

    T = wavs.shape[-1]
    x_len = int(wav_lens[0])
    assert (wav_lens == x_len).all(), "all windows are 0.4 s, so every window has the same length"
    n50 = sum(1 for k in range((T + 1) % 2, T, 2) if k < x_len)    # valid 50 Hz frames (mask in AudioEncoder)
    n25 = int(out_lens[0])
    n12 = (n25 - 1) // 2 + 1                                        # feat_lens in LycheeFD.forward
    f = lambda t: t.float().cpu().numpy()  # noqa: E731
    rows = [(f"enc_{k}", f(captured[k]), n50, 0.02) for k in range(len(blocks) + 1)]
    rows.append(("enc_pool", f(out), n25, 0.04))
    rows.append(("adaptor", f(ada), n12, 0.08))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SIBLING_MANIFEST)
    ap.add_argument("--tts-dir", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/lychee_fd")
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--lead-chunks", type=int, default=4, help="same as extract_lychee.py")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "work/features/lychee_encoder.npz")
    cfg = ap.parse_args()
    setup_project_cache()

    if not (cfg.code_dir / "LycheeFD.py").exists():
        raise SystemExit(f"{cfg.code_dir}/LycheeFD.py missing: run scripts/fetch_lychee_code.sh")
    if not cfg.manifest.exists():
        raise SystemExit(f"manifest not found: {cfg.manifest}")
    rows = read_jsonl(cfg.manifest)
    if cfg.limit:
        rows = rows[:cfg.limit]
    root = cfg.tts_dir or cfg.manifest.parent

    cfg.model_dir = resolve_model_dir(cfg.model_dir)
    model = load_lychee(cfg.model_dir, cfg.code_dir, cfg.device, cfg.attn)
    n_blocks = len(model.encoder.blocks)
    print(f"audio encoder: {n_blocks} blocks | adaptor -> {model.config.audio_encoder_config.llm_dim}-d")

    names, feats, meta_rows = None, [], []
    for i, r in enumerate(rows):
        x, _ = load_audio(root / r["wav"], sr=SR)
        wave, lead_s, speech_s, n_chunks = prepare_waveform(x, SR, cfg.lead_chunks, trim_silence)
        wavs, wav_lens = encoder_windows(wave)
        pts = run_encoder(model, wavs, wav_lens)
        vec = []
        for name, H, n_valid, frame_s in pts:
            m = window_frame_mask(H.shape[0], n_valid, frame_s, lead_s, speech_s,
                                  min_overlap=cfg.min_overlap)
            vec.append(H[:, :n_valid][m].mean(axis=0))
        if names is None:
            names = [p[0] for p in pts]
        feats.append(vec)
        meta_rows.append((r["utt_id"], r["speaker"], r["sentence_id"]))
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}", flush=True)

    depths = [k - len(names) for k in range(len(names))]
    meta = {"model": "Lychee-FD", "part": "audio encoder", "model_dir": str(cfg.model_dir),
            "lead_chunks": cfg.lead_chunks, "n_blocks": n_blocks,
            "depth_note": "negative = before the LLM; -1 = adaptor output, LLM trunk_0 = 0"}
    save_points(cfg.out, names, depths, "encoder", feats, meta_rows, meta)


if __name__ == "__main__":
    sys.exit(main())
