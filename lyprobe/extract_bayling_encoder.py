#!/usr/bin/env python3
"""
BayLing-Duplex's audio encoder, layer by layer: the GLM-4-Voice speech tokenizer that
turns the user's audio into the discrete tokens the LLM reads.

WhisperVQEncoder (bayling_duplex_runtime/speech_tokenizer/modeling_whisper.py; released
config THUDM/glm-4-voice-tokenizer: quantize_encoder_only, quantize_position 16,
pooling_position 16, pooling_kernel_size 4, causal convolutions, block-causal attention
over 200 frames):

  128-bin log-mel -> conv1, conv2 (stride 2) + positions      enc_0      50 Hz, 1280-d
    -> 16 Whisper encoder layers                               enc_1..16  50 Hz
    -> AvgPool(4) -> nearest codebook vector (16384 codes)     vq         12.5 Hz, 1280-d
  the code index is the token the LLM sees (its embedding = BayLing layer_0)

Only the tokenizer is loaded (no LLM). The call replicates BayLingDuplex.tokenize_audio
(30 s segments, WhisperFeatureExtractor with padding='longest', pad_to_multiple_of=
1280 samples), and the waveform is prepared exactly like
bayling-speaker-probe/spkprobe/extract_bayling.py (trim, --lead-blocks x 0.8 s silence,
pad to a block), so clips line up with bayling_silence.npz by utt_id.

Points (depth < 0 = before the LLM; layer_0 of bayling_silence.npz is depth 0):
  enc_0 .. enc_16, vq    depth -18 .. -1

Output (default work/features/bayling_encoder.npz): same format as
extract_lychee_encoder.py (X_<k> [N, 1280] per point).

Hard deps: numpy
Model deps (lazy): torch, transformers, bayling_duplex (pip-installed or --bayling-repo)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import (PROJECT_ROOT, SIBLING, SIBLING_MANIFEST, load_audio,  # noqa: E402
                            read_jsonl, setup_project_cache, trim_silence)
from lyprobe.extract_lychee_encoder import save_points  # noqa: E402
from lyprobe.layout import window_frame_mask  # noqa: E402

SR = 16000
BLOCK_S = 0.8          # 10 user tokens x 80 ms (released 10:5:10 checkpoint)


def prepare_waveform_bayling(x, sr, lead_blocks, block_s=BLOCK_S):
    """extract_bayling.prepare_waveform: trim, lead silence, pad to a block boundary."""
    speech, _, _ = trim_silence(x, sr)
    lead = np.zeros(int(round(lead_blocks * block_s * sr)), np.float32)
    body = np.concatenate([lead, speech])
    block_n = int(round(block_s * sr))
    n_blocks = int(np.ceil(len(body) / block_n))
    wave = np.zeros(n_blocks * block_n, np.float32)
    wave[:len(body)] = body
    return wave, len(lead) / sr, len(speech) / sr


def load_tokenizer(path, device, bayling_repo=None):
    if bayling_repo:
        sys.path.insert(0, str(Path(bayling_repo).resolve()))
    from transformers import WhisperFeatureExtractor
    # importing duplex puts bayling_duplex_runtime on sys.path, as BayLingDuplex does
    from bayling_duplex.duplex import WhisperVQEncoder

    enc = WhisperVQEncoder.from_pretrained(str(path)).eval().to(device)
    return enc, WhisperFeatureExtractor.from_pretrained(str(path))


def run_tokenizer(enc, fe, wave, device):
    """-> list of (name, [n_valid_frames, 1280] float32, frame_s), 30 s segments joined."""
    import torch

    pool = enc.config.pooling_kernel_size or 1
    conv = enc.conv1.stride[0] * enc.conv2.stride[0]
    stride = conv * pool * fe.hop_length
    layers = list(enc.layers)
    per = {k: [] for k in range(len(layers) + 1)}
    per["vq"] = []
    captured = {}
    hooks = [layers[0].register_forward_pre_hook(
        lambda m, args: captured.__setitem__(0, args[0].detach()))]
    for k, layer in enumerate(layers, start=1):
        hooks.append(layer.register_forward_hook(
            lambda m, args, out, k=k: captured.__setitem__(k, out[0].detach())))
    try:
        t = 0
        while t * SR < len(wave):
            seg = wave[t * SR:(t + 30) * SR]
            feats = fe([seg], sampling_rate=SR, return_attention_mask=True, return_tensors="pt",
                       padding="longest", pad_to_multiple_of=stride).to(device)
            captured.clear()
            with torch.no_grad():
                out = enc(**feats)
                vq = enc.codebook(out.quantized_token_ids)
            m50 = feats.attention_mask[0, ::conv].bool()
            m12 = m50[::pool]
            for k in range(len(layers) + 1):
                h = captured[k][0]
                per[k].append(h[:len(m50)][m50].float().cpu().numpy())
            per["vq"].append(vq[0][:len(m12)][m12].float().cpu().numpy())
            t += 30
    finally:
        for h in hooks:
            h.remove()
    rows = [(f"enc_{k}", np.concatenate(per[k]), 0.02) for k in range(len(layers) + 1)]
    rows.append(("vq", np.concatenate(per["vq"]), 0.02 * pool))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SIBLING_MANIFEST)
    ap.add_argument("--tts-dir", type=Path, default=None)
    ap.add_argument("--speech-tokenizer-path", type=Path, default=SIBLING / "models/speech_tokenizer")
    ap.add_argument("--bayling-repo", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lead-blocks", type=int, default=2, help="same as extract_bayling.py")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "work/features/bayling_encoder.npz")
    cfg = ap.parse_args()
    setup_project_cache()

    if not cfg.manifest.exists():
        raise SystemExit(f"manifest not found: {cfg.manifest}")
    if not (cfg.speech_tokenizer_path / "config.json").exists():
        raise SystemExit(f"no config.json in {cfg.speech_tokenizer_path}; pass --speech-tokenizer-path")
    rows = read_jsonl(cfg.manifest)
    if cfg.limit:
        rows = rows[:cfg.limit]
    root = cfg.tts_dir or cfg.manifest.parent

    enc, fe = load_tokenizer(cfg.speech_tokenizer_path, cfg.device, cfg.bayling_repo)
    c = enc.config
    print(f"speech tokenizer: {len(enc.layers)} layers | pooling x{c.pooling_kernel_size} at layer "
          f"{c.pooling_position} | VQ at layer {c.quantize_position} ({c.quantize_vocab_size} codes)")

    names, feats, meta_rows = None, [], []
    for i, r in enumerate(rows):
        x, _ = load_audio(root / r["wav"], sr=SR)
        wave, lead_s, speech_s = prepare_waveform_bayling(x, SR, cfg.lead_blocks)
        pts = run_tokenizer(enc, fe, wave, cfg.device)
        vec = []
        for name, H, frame_s in pts:
            m = window_frame_mask(1, len(H), frame_s, lead_s, speech_s, window_s=0.0,
                                  min_overlap=cfg.min_overlap)[0]
            vec.append(H[m].mean(axis=0))
        if names is None:
            names = [p[0] for p in pts]
        feats.append(vec)
        meta_rows.append((r["utt_id"], r["speaker"], r["sentence_id"]))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}", flush=True)

    depths = [k - len(names) for k in range(len(names))]
    meta = {"model": "BayLing-Duplex", "part": "speech tokenizer (GLM-4-Voice WhisperVQEncoder)",
            "speech_tokenizer_path": str(cfg.speech_tokenizer_path), "lead_blocks": cfg.lead_blocks,
            "n_layers": len(enc.layers),
            "depth_note": "negative = before the LLM; -1 = VQ codebook vector, LLM layer_0 = 0"}
    save_points(cfg.out, names, depths, "encoder", feats, meta_rows, meta)


if __name__ == "__main__":
    sys.exit(main())
