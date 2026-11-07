#!/usr/bin/env python3
"""
Run Lychee-FD over every utterance and keep the hidden states of every layer on every
path through the model, pooled over the user's speech.

Lychee is a tree, not a stack (released config.json):

  trunk    backbone Qwen2, 28 layers; the text head reads its end
  control  branches at trunk layer 20 (28 - control_branch_layer 8), + 4 layers
  speech   branches at trunk layer 24 (28 - stoken layers 4),        + 4 stoken layers
  merge    speech output + left-shifted text embedding,             + 4 merge layers

Points saved (depth = layers from the input embedding):

  trunk_0 .. trunk_28     depth 0..28   (trunk_0 = embedding sum, before any block)
  control_1 .. control_4  depth 21..24
  speech_1 .. speech_4    depth 25..28
  merge_1 .. merge_4      depth 29..32

The backbone forward is LycheeFD.forward itself (training path), so the four input
channels and the encoder features are inserted exactly as the model was trained.
The branch models are then re-run with output_hidden_states=True, because forward
only returns their last layer.

Output (default work/features/lychee.npz):
  pooled       [N, P, 3584]   mean over speech slots, P = 41 points
  point_names, point_depths, point_track
  utt_id, speaker, sentence_id, n_speech_slots, meta

Hard deps: numpy
Model deps (lazy): torch, torchaudio, librosa, transformers==4.57.x, soundfile
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import (PROJECT_ROOT, SIBLING_MANIFEST, load_audio, read_jsonl,  # noqa: E402
                            setup_project_cache, trim_silence)
from lyprobe.layout import (CHUNK_S, SR, SYSTEM_MESSAGE_PREFIX, compute_token_num,  # noqa: E402
                            listening_suffix, mel_frames, prepare_waveform, speech_slot_mask)


def point_table(n_trunk, branch_at, n_branch):
    """[(name, depth, track)] for trunk + branches. branch_at = {track: trunk index}."""
    pts = [(f"trunk_{i}", i, "trunk") for i in range(n_trunk + 1)]
    for track in ("control", "speech"):
        for k in range(1, n_branch[track] + 1):
            pts.append((f"{track}_{k}", branch_at[track] + k, track))
    for k in range(1, n_branch["merge"] + 1):
        pts.append((f"merge_{k}", branch_at["speech"] + n_branch["speech"] + k, "merge"))
    return pts


# ---------------------------------------------------------------- model side

def resolve_model_dir(model_dir):
    """Return the folder holding config.json + weights, or exit with what is actually there.

    Without a local config.json, transformers treats the path as a Hub repo id and fails
    with a misleading HFValidationError. The official README downloads with
    --include "lychee_full_duplex/*", which nests the files one level down.
    """
    model_dir = Path(model_dir)
    if (model_dir / "config.json").exists():
        return model_dir
    nested = sorted(model_dir.glob("*/config.json")) if model_dir.is_dir() else []
    if len(nested) == 1:
        print(f"using {nested[0].parent} (config.json found one level down)")
        return nested[0].parent
    found = sorted(p.name for p in model_dir.iterdir())[:20] if model_dir.is_dir() else None
    raise SystemExit(
        f"no config.json in {model_dir}\n"
        f"  contents: {found if found is not None else 'folder does not exist'}\n"
        f"  download: export HF_HOME=$PWD/cache/huggingface && "
        f"hf download HIT-TMG/Lychee-FD --local-dir {model_dir}\n"
        f"  or pass --model-dir to the folder that contains config.json")


def load_lychee(model_dir, code_dir, device, attn):
    import torch

    sys.path.insert(0, str(Path(code_dir).resolve().parent))
    from lychee_fd_model.LycheeFD import LycheeFD, LycheeFDConfig

    cfg = LycheeFDConfig.from_pretrained(model_dir)
    # sub-configs are plain Qwen2Configs built inside the config class; give them the
    # attention backend explicitly (the official runtime defaults to eager)
    for sub in ("text_config", "stoken_layer_config", "control_layer_config", "merge_layer_config"):
        c = getattr(cfg, sub, None)
        if c is not None:
            c._attn_implementation = attn
    cfg._attn_implementation = attn
    model = LycheeFD.from_pretrained(model_dir, config=cfg, torch_dtype=torch.bfloat16,
                                     attn_implementation=attn)
    return model.to(device).eval()


def log_mel(audio, n_mels=128, padding=479):
    """Lychee's log_mel_spectrogram (LycheeFDDataset.py), on CPU."""
    import librosa
    import torch
    import torch.nn.functional as F

    audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    mags = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels))
    log_spec = torch.clamp(filters @ mags, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def build_inputs(model, tokenizer, wave, n_chunks):
    """Per-0.4 s log-mel windows (LycheeFDDataset.stepaudio_audio_prepreocess) + the four channels."""
    import torch

    win = int(round(CHUNK_S * SR))
    feats, lens = [], []
    for i in range(0, len(wave), win):
        mel = log_mel(wave[i:i + win])
        assert compute_token_num(mel.shape[1]) * 2 == 10, "window must give 5 features"
        feats.append(mel.t())
        lens.append(mel.shape[1] - 2)
    wavs = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).transpose(1, 2)
    wav_lens = torch.tensor(lens, dtype=torch.int32)

    prefix = tokenizer([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
    text, stoken, control, audio = listening_suffix(n_chunks)
    dev = model.device
    t = lambda ids: torch.tensor([ids], dtype=torch.long, device=dev)  # noqa: E731
    return dict(input_ids=t(prefix + text), prefix_input_ids=t(prefix), stoken_ids=t(stoken),
                control_input_ids=t(control), audio_input_ids=t(audio),
                wavs=wavs.to(dev), wav_lens=wav_lens.to(dev)), len(prefix)


def run_all_points(model, inputs, idx):
    """Hidden states of every point at sequence positions idx -> float32 [P, len(idx), H]."""
    import torch

    with torch.no_grad():
        out = model(**inputs, apply_internal_merge=False, use_cache=False)
        hs = out.hidden_states
        L = model.config.text_config.num_hidden_layers
        assert len(hs) == L + 1, f"expected {L + 1} backbone hidden states, got {len(hs)}"
        pick = lambda h: h[0, idx].float().cpu()  # noqa: E731
        rows = [pick(h) for h in hs]

        def branch(m, x):
            o = m(inputs_embeds=x, output_hidden_states=True, return_dict=True, use_cache=False)
            return o.hidden_states[1:], o.last_hidden_state

        ctrl_hs, _ = branch(model.control_model, hs[model.control_model_index])
        spk_hs, spk_last = branch(model.stoken_model, hs[model.stoken_model_index])
        rows += [pick(h) for h in ctrl_hs] + [pick(h) for h in spk_hs]

        if model.merge_model is not None:
            # same as forward's teacher-forcing merge: speech output + embedding of the
            # next text token (all text_pad while listening), detached
            emb = model.get_input_embeddings()(inputs["input_ids"])
            shifted = torch.zeros_like(emb)
            shifted[:, :-1] = emb[:, 1:]
            mrg_hs, _ = branch(model.merge_model, spk_last + shifted)
            rows += [pick(h) for h in mrg_hs]
    return torch.stack(rows).numpy()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SIBLING_MANIFEST,
                    help="manifest.kept.jsonl from bayling-speaker-probe (wav paths relative to it)")
    ap.add_argument("--tts-dir", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/lychee_fd")
    ap.add_argument("--code-dir", type=Path, default=PROJECT_ROOT / "third_party/lychee_fd_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", default="eager", help="eager (official runtime default) or sdpa")
    ap.add_argument("--lead-chunks", type=int, default=4, help="0.4 s chunks of silence before speech")
    ap.add_argument("--slots", choices=["all", "audio"], default="all",
                    help="pool every speech slot, or only the even slots that carry encoder features")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "work/features/lychee.npz")
    cfg = ap.parse_args()
    setup_project_cache()

    if not (cfg.code_dir / "LycheeFD.py").exists():
        raise SystemExit(f"{cfg.code_dir}/LycheeFD.py missing: run scripts/fetch_lychee_code.sh")

    from transformers import AutoTokenizer

    if not cfg.manifest.exists():
        raise SystemExit(f"manifest not found: {cfg.manifest}\n"
                         "expected bayling-speaker-probe next to this repo, or pass --manifest")
    rows = read_jsonl(cfg.manifest)
    if cfg.limit:
        rows = rows[:cfg.limit]
    root = cfg.tts_dir or cfg.manifest.parent

    cfg.model_dir = resolve_model_dir(cfg.model_dir)
    model = load_lychee(cfg.model_dir, cfg.code_dir, cfg.device, cfg.attn)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, use_fast=False, trust_remote_code=True)
    c = model.config
    pts = point_table(
        c.text_config.num_hidden_layers,
        {"control": model.control_model_index, "speech": model.stoken_model_index},
        {"control": c.control_layer_config.num_hidden_layers,
         "speech": c.stoken_layer_config.num_hidden_layers,
         "merge": c.merge_layer_config.num_hidden_layers if model.merge_model is not None else 0})
    print(f"trunk {c.text_config.num_hidden_layers} layers | control branches at {model.control_model_index}"
          f" | speech branches at {model.stoken_model_index} | {len(pts)} points")

    pooled, meta_rows = [], []
    for i, r in enumerate(rows):
        x, _ = load_audio(root / r["wav"], sr=SR)
        wave, lead_s, speech_s, n_chunks = prepare_waveform(x, SR, cfg.lead_chunks, trim_silence)
        inputs, n_prefix = build_inputs(model, tokenizer, wave, n_chunks)
        n_slots = n_chunks * 10
        mask = speech_slot_mask(n_slots, lead_s, speech_s, min_overlap=cfg.min_overlap)
        if cfg.slots == "audio":
            mask &= (np.arange(n_slots) % 2 == 0)
        if not mask.any():
            print(f"skip {r['utt_id']}: no speech slots")
            continue
        h = run_all_points(model, inputs, n_prefix + np.where(mask)[0])   # [P, n, H]
        pooled.append(h.mean(axis=1))
        meta_rows.append((r["utt_id"], r["speaker"], r["sentence_id"], int(mask.sum())))
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}", flush=True)

    pooled = np.stack(pooled)
    store = pooled.astype(np.float16)
    if not np.isfinite(store).all():
        print("float16 overflow; storing float32")
        store = pooled.astype(np.float32)
    utt, spk, sent, n_sp = map(np.array, zip(*meta_rows))
    meta = {"model": "Lychee-FD", "model_dir": str(cfg.model_dir), "lead_chunks": cfg.lead_chunks,
            "slots": cfg.slots, "attn": cfg.attn,
            "control_branch_at": int(model.control_model_index),
            "speech_branch_at": int(model.stoken_model_index)}
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.out, pooled=store, utt_id=utt, speaker=spk, sentence_id=sent, n_speech_slots=n_sp,
             point_names=np.array([p[0] for p in pts]), point_depths=np.array([p[1] for p in pts]),
             point_track=np.array([p[2] for p in pts]), meta=json.dumps(meta))
    print(f"wrote {cfg.out}: pooled {tuple(store.shape)}")


if __name__ == "__main__":
    sys.exit(main())
