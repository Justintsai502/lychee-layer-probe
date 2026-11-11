#!/usr/bin/env python3
"""
Agent turns with Lychee speech tokens, offline, for every participant of every multi-track row.

Lychee's stoken is the 25 Hz CosyVoice-2 tokenizer that ships with Step-Audio-2-mini
(token2wav/speech_tokenizer_v2_25hz.onnx). Step-Audio-2's own token2wav.py loads it with
``s3tokenizer.load_model(...onnx)`` and feeds the prompt's tokens to the same flow model as
the LLM's output tokens, so they are one token space; this script uses the same calls.

For each participant L of a row:
  L's utterances -> turns (lyspk.dialog.merge_utterances: pauses < --merge-gap s are joined,
  unless that makes a turn longer than --max-turn s) -> L's own track, 16 kHz, turn span
  -> tokens (pieces of at most 30 s, the tokenizer's window, concatenated)

Writes <turns-dir>/<row id>.json  {"id", "merge_gap", "max_turn", "tokenizer",
"turns": {L: [{start, end, text, stoken: [...]}]}} and a copy of the manifest with a
``turns`` field per row (--out). Rows whose JSON exists are skipped unless --force.

  python -m lyspk.stoken --manifest work/spk/mt/if_train.jsonl --out work/spk/mt/if_train.turns.jsonl \\
      --tokenizer models/step_audio2_tok/token2wav/speech_tokenizer_v2_25hz.onnx

Model deps (lazy): torch, s3tokenizer, soundfile
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, resample, write_jsonl  # noqa: E402
from lyspk.data import load_manifest  # noqa: E402
from lyspk.dialog import merge_utterances  # noqa: E402

SR = 16000
PIECE_S = 30.0
TOKEN_HZ = 25


def build_turns(utts, merge_gap, max_turn):
    """Like merge_utterances, but never joins across a pause into a turn longer than ``max_turn``."""
    out = []
    for t in merge_utterances(utts, 0.0):
        if out and t["start"] - out[-1]["end"] < merge_gap and t["end"] - out[-1]["start"] <= max_turn:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
            out[-1]["text"] = (out[-1]["text"] + " " + t["text"]).strip()
        else:
            out.append(dict(t))
    return out


def read_span(path, start, end):
    import soundfile as sf

    info = sf.info(path)
    sr = int(info.samplerate)
    a, b = int(round(start * sr)), int(round(end * sr))
    x, _ = sf.read(path, start=min(a, info.frames), stop=min(b, info.frames), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr, SR)


def pieces(x):
    n = int(PIECE_S * SR)
    return [x[i:i + n] for i in range(0, max(len(x), 1), n)] or [x]


class S3Tokenizer:
    """s3tokenizer exactly as Step-Audio-2's token2wav.py calls it, batched."""

    def __init__(self, onnx_path, device="cuda", batch=16):
        import s3tokenizer
        import torch

        self.s3, self.torch = s3tokenizer, torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = s3tokenizer.load_model(str(onnx_path)).to(self.device).eval()
        self.batch = batch

    def __call__(self, audios):
        out = []
        for i in range(0, len(audios), self.batch):
            mels = [self.s3.log_mel_spectrogram(self.torch.from_numpy(np.ascontiguousarray(a, np.float32)))
                    for a in audios[i:i + self.batch]]
            mels, lens = self.s3.padding(mels)
            with self.torch.no_grad():
                codes, clens = self.model.quantize(mels.to(self.device), lens.to(self.device))
            out += [codes[j, :int(clens[j])].cpu().tolist() for j in range(len(mels))]
        return out


def tokenize_row(row, tokenize, merge_gap, max_turn):
    """{letter: [turn with stoken]} for every participant who has utterances."""
    turns = {}
    for L, path in sorted(row["tracks"].items()):
        mine = [u for u in row.get("utterances", []) if u["speaker"] == L]
        tt = build_turns(mine, merge_gap, max_turn)
        if not tt:
            continue
        chunks, owner = [], []
        for k, t in enumerate(tt):
            for p in pieces(read_span(path, t["start"], t["end"])):
                if len(p) >= int(0.08 * SR):
                    chunks.append(p)
                    owner.append(k)
        toks = tokenize(chunks) if chunks else []
        for t in tt:
            t["stoken"] = []
        for k, tk in zip(owner, toks):
            tt[k]["stoken"] += [int(v) for v in tk]
        turns[L] = [dict(t, start=round(t["start"], 3), end=round(t["end"], 3)) for t in tt if t["stoken"]]
    return turns


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="multi-track manifest from lyspk.mt_data")
    ap.add_argument("--out", type=Path, required=True, help="manifest copy with a turns field per row")
    ap.add_argument("--turns-dir", type=Path, default=None, help="default work/spk/turns/<manifest stem>")
    ap.add_argument("--tokenizer", type=Path,
                    default=PROJECT_ROOT / "models/step_audio2_tok/token2wav/speech_tokenizer_v2_25hz.onnx")
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--max-turn", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args(argv)

    rows = load_manifest(cfg.manifest)
    tdir = cfg.turns_dir or PROJECT_ROOT / "work/spk/turns" / cfg.manifest.stem
    tdir.mkdir(parents=True, exist_ok=True)
    tok = None
    out, n_new, n_fail, t0 = [], 0, 0, time.time()
    for i, r in enumerate(rows):
        f = tdir / f"{safe_name(r['id'])}.json"
        if not f.exists() or cfg.force:
            if tok is None:
                if not cfg.tokenizer.exists():
                    raise SystemExit(f"{cfg.tokenizer} missing (scripts/run_spk_mt.sh downloads it)")
                tok = S3Tokenizer(cfg.tokenizer, cfg.device, cfg.batch)
            try:
                turns = tokenize_row(r, tok, cfg.merge_gap, cfg.max_turn)
            except Exception as e:  # a bad wav should not stop a 3000-row run
                n_fail += 1
                print(f"[{i + 1}/{len(rows)}] {r['id']}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            f.write_text(json.dumps({"id": r["id"], "merge_gap": cfg.merge_gap, "max_turn": cfg.max_turn,
                                     "tokenizer": str(cfg.tokenizer), "turns": turns}))
            n_new += 1
            n_tok = sum(len(t["stoken"]) for v in turns.values() for t in v)
            print(f"[{i + 1}/{len(rows)}] {r['id']}: {sum(map(len, turns.values()))} turns, {n_tok} tokens "
                  f"({n_tok / TOKEN_HZ / 60:.1f} min) {time.time() - t0:.0f}s", flush=True)
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        row["turns"] = str(f.resolve())
        out.append(row)
    write_jsonl(cfg.out, out)
    print(f"wrote {cfg.out}: {len(out)} rows ({n_new} tokenized now, {n_fail} failed) turns in {tdir}")


if __name__ == "__main__":
    sys.exit(main())
