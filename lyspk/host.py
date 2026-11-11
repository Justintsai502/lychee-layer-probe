"""Lychee-FD as a host for the speaker head: causal 12.5 Hz feature stacks, with LoRA.

What one call does (``LycheeHost.features(wave)``), exactly as LycheeFD.forward does it
while the model is listening (lyprobe/layout.py has the token layout):

  wave, a multiple of 0.4 s
    -> one 128-bin log-mel per 0.4 s window (42 frames), windows encoded independently
    -> conv1, conv2 + positions, 32 Whisper blocks (50 Hz, 1280-d)   [LoRA]   enc stack
    -> AvgPool(2) + LayerNorm -> Adaptor conv s2 + MLP (12.5 Hz, 3584-d) [LoRA]
    -> masked_scatter into the <audio_patch> slots; + text_pad / stoken_pad / control
       embeddings; Qwen2 trunk layers 1..L                              [LoRA]   llm stack

Feature frame t is window t // 5, adaptor output t % 5, suffix slot 2t. Each window's
features have heard the whole window, so frame t has heard audio up to the end of its
window and nothing after: causal at the 0.4 s chunk the model itself streams in.

Only what the readout and the LoRA can reach is kept on the GPU: the trunk is cut after
layer L, and the control / speech / merge branches and the LM head are dropped (they
read layers above L, so nothing trained here changes their inputs except through the
trunk, which the preservation target pins; see ``keep``).

``keep`` is the tensor everything downstream of the trained part reads: trunk layer L's
output at every suffix slot (or the adaptor output when L = 0). Matching it to the
frozen model's value keeps the rest of Lychee computing what it computed before.

Dialogue mode (``from_pretrained(..., dialogue=True)``, lyspk.dialog builds the inputs):
the whole trunk, the speech-token branch + merge branch, the control branch and the LM
head stay loaded, frozen, and ``features(wave, seq, losses=True)`` also returns Lychee's
own three losses on the agent's turns, computed as LycheeFD.forward does (shifted CE,
clamped vocab ranges), but with the LM head applied only at labelled positions. LoRA
stays where it was (encoder, adaptor, trunk 1..L); gradients reach it through the frozen
upper layers. Without ``losses`` the forward stops after trunk layer L.

Model deps (lazy): torch, librosa, transformers==4.57.x (and the Lychee code in
third_party/lychee_fd_model, scripts/fetch_lychee_code.sh)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.layout import CHUNK_S, SR, SYSTEM_MESSAGE_PREFIX, TOK, listening_suffix  # noqa: E402
from lyspk.dialog import CONTROL_RANGE, IGNORE, STOKEN_DELAY, STOKEN_RANGE  # noqa: E402
from lyspk.dialog import TOK as DTOK  # noqa: E402
from lyspk.lora import QWEN2_LINEARS, wrap_linears  # noqa: E402

WIN = int(round(CHUNK_S * SR))        # 6400 samples per window
FEATS_PER_WIN = 5
ENC_PER_FEAT = 4                      # 50 Hz encoder frames per 12.5 Hz feature


class _Stop(Exception):
    """Raised from a forward hook to end the trunk after the last layer the readout reads."""


def _import_lychee(code_dir):
    sys.path.insert(0, str(Path(code_dir).resolve().parent))
    import lychee_fd_model.modeling_step_audio_2 as m
    return m


class LycheeHost(nn.Module):
    def __init__(self, encoder, adapter, lm, embed, prefix_ids, *, llm_layers, no_text_label=False,
                 no_stoken_label=False, grad_ckpt=True, mask_fns=None, branches=None, text_fn=None,
                 stoken_delay=STOKEN_DELAY):
        super().__init__()
        self.encoder, self.adapter = encoder, adapter
        self.lm, self.embed = lm, embed
        self.llm_layers = int(llm_layers)
        self.prefix_ids = list(prefix_ids)
        self.no_text_label, self.no_stoken_label = bool(no_text_label), bool(no_stoken_label)
        self.grad_ckpt = bool(grad_ckpt)
        self._make_mask, self._mask_bias = mask_fns
        self.dialogue = branches is not None
        self.text_fn, self.stoken_delay = text_fn, int(stoken_delay)
        if self.dialogue:
            if not self.llm_layers:
                raise ValueError("dialogue mode needs the trunk (llm_layers >= 1)")
            self.stoken_model, self.merge_model = branches["stoken"], branches["merge"]
            self.control_model, self.lm_head = branches["control"], branches["lm_head"]
            self.stoken_index, self.control_index = int(branches["stoken_index"]), int(branches["control_index"])
            if not self.llm_layers <= len(lm.layers):
                raise ValueError(f"llm_layers {self.llm_layers} > trunk depth {len(lm.layers)}")
        elif self.llm_layers and (lm is None or len(lm.layers) != self.llm_layers):
            raise ValueError(f"trunk must be cut to {self.llm_layers} layers")
        for p in self.parameters():
            p.requires_grad_(False)
        self._mel_filters = None

    # ------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(cls, model_dir, code_dir, device, llm_layers, attn="sdpa", grad_ckpt=True, dialogue=False):
        from transformers import AutoTokenizer

        m = _import_lychee(code_dir)
        from lychee_fd_model.LycheeFD import LycheeFD, LycheeFDConfig

        cfg = LycheeFDConfig.from_pretrained(model_dir)
        for sub in ("text_config", "stoken_layer_config", "control_layer_config", "merge_layer_config"):
            c = getattr(cfg, sub, None)
            if c is not None:
                c._attn_implementation = attn
        cfg._attn_implementation = attn
        model = LycheeFD.from_pretrained(model_dir, config=cfg, torch_dtype=torch.bfloat16,
                                         attn_implementation=attn, low_cpu_mem_usage=True)
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        prefix = tok([SYSTEM_MESSAGE_PREFIX], add_special_tokens=False).input_ids[0]
        n_trunk = len(model.model.layers)
        if not 0 <= llm_layers <= n_trunk:
            raise SystemExit(f"--llm-layers {llm_layers}: the trunk has {n_trunk} layers")
        lm = embed = branches = text_fn = None
        if llm_layers:
            lm = model.model
            if not dialogue:
                lm.layers = nn.ModuleList(list(lm.layers)[:llm_layers])
            embed = lm.embed_tokens
        if dialogue:
            check_token_ids(cfg, tok)
            if model.merge_model is None or model.stoken_model is None or model.control_model is None:
                raise SystemExit("dialogue mode expects the stoken, merge and control branches")
            for b in (model.stoken_model, model.merge_model, model.control_model):
                b.embed_tokens = None       # the branches only ever take inputs_embeds: 3 x 1.1 GB freed
            branches = {"stoken": model.stoken_model, "merge": model.merge_model, "control": model.control_model,
                        "lm_head": model.lm_head, "stoken_index": model.stoken_model_index,
                        "control_index": model.control_model_index}
            text_fn = lambda text: tok.encode(text, add_special_tokens=False)  # noqa: E731
        host = cls(model.encoder, model.adapter, lm, embed, prefix, llm_layers=llm_layers,
                   no_text_label=model.no_text_label, no_stoken_label=model.no_stoken_label,
                   grad_ckpt=grad_ckpt, mask_fns=(m.make_non_pad_mask, m.mask_to_bias), branches=branches,
                   text_fn=text_fn, stoken_delay=int(getattr(cfg, "stoken_delay_num", STOKEN_DELAY) or 0))
        host.meta = {"model_dir": str(model_dir), "trunk_layers": n_trunk, "dialogue": bool(dialogue),
                     "control_branch_at": int(model.control_model_index),
                     "speech_branch_at": int(model.stoken_model_index), "stoken_delay": host.stoken_delay}
        del model
        return host.to(device)

    def apply_lora(self, enc_rank=16, llm_rank=16, adaptor_rank=16, scaling=2.0, enc_blocks=None, llm_high_rank=0):
        """LoRA on the encoder blocks (all, or the indices in ``enc_blocks``), the adaptor's two
        linears (only when the trunk is read: with L = 0 the adaptor feeds nothing the loss
        sees) and q/k/v/o/gate/up/down of trunk layers 1..L; in dialogue mode also of layers
        L+1..end at ``llm_high_rank``. Returns counts."""
        n = {"encoder": 0, "adaptor": 0, "llm": 0}
        blocks = list(self.encoder.blocks)
        pick = range(len(blocks)) if enc_blocks is None else enc_blocks
        if enc_rank:
            for i in pick:
                n["encoder"] += wrap_linears(blocks[i], enc_rank, scaling)
        if adaptor_rank and self.llm_layers:
            n["adaptor"] = wrap_linears(self.adapter, adaptor_rank, scaling, names=("linear1", "linear2"))
        if llm_rank and self.llm_layers:
            for layer in list(self.lm.layers)[:self.llm_layers]:
                n["llm"] += wrap_linears(layer, llm_rank, scaling, names=QWEN2_LINEARS)
        if llm_high_rank and self.dialogue:
            # mpmoshi's second LoRA group (Moshi layers 16-31, rank 8): the trunk above the readout.
            # It only changes what the readout never sees, so the head's features are unaffected;
            # the branches stay frozen as mpmoshi keeps the depformer free of LoRA.
            n["llm_high"] = 0
            for layer in list(self.lm.layers)[self.llm_layers:]:
                n["llm_high"] += wrap_linears(layer, llm_high_rank, scaling, names=QWEN2_LINEARS)
        if self.llm_layers and self.grad_ckpt:
            mods = [self.lm] + ([self.stoken_model, self.merge_model, self.control_model] if self.dialogue else [])
            for mod in mods:
                mod.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return n

    def groups(self) -> dict:
        """Readout groups: ``{name: (n_layers, dim)}``."""
        g = {"enc": (len(self.encoder.blocks) + 1, self.encoder.after_norm.normalized_shape[0])}
        if self.llm_layers:
            g["llm"] = (self.llm_layers + 1, self.embed.embedding_dim)
        return g

    @property
    def device(self):
        return self.encoder.conv1.weight.device

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    # ------------------------------------------------------------ audio -> features

    def log_mel(self, audio, n_mels=128, padding=479):
        """Lychee's log_mel_spectrogram (same as lyprobe.extract_lychee.log_mel)."""
        if self._mel_filters is None:
            import librosa
            self._mel_filters = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels))
        audio = F.pad(torch.from_numpy(np.asarray(audio, dtype=np.float32)), (0, padding))
        stft = torch.stft(audio, 400, 160, window=torch.hann_window(400), return_complex=True)
        mags = stft[..., :-1].abs() ** 2
        log_spec = torch.clamp(self._mel_filters @ mags, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0

    def windows(self, wave):
        if len(wave) % WIN:
            raise ValueError(f"waveform must be a multiple of {WIN} samples, got {len(wave)}")
        mels = [self.log_mel(wave[i:i + WIN]) for i in range(0, len(wave), WIN)]
        wavs = torch.stack(mels)                                   # [n_win, 128, 42]
        lens = torch.full((len(mels),), mels[0].shape[1] - 2, dtype=torch.int32)
        return wavs.to(self.device, self.dtype), lens.to(self.device)

    def run_encoder(self, wavs, wav_lens):
        """AudioEncoder.forward with every block's output kept (and checkpointed when training).
        Returns (outs: input of block 1 + each block output, pooled+normed output, lens)."""
        e = self.encoder
        T = wavs.size(-1)
        x = F.gelu(e.conv1(wavs))
        x = F.gelu(e.conv2(x)).permute(0, 2, 1)
        mask = self._make_mask(wav_lens, T).unsqueeze(1)
        mask = self._mask_bias(mask[:, :, (T + 1) % 2::2], x.dtype).unsqueeze(1)
        x = (x + e.positional_embedding.weight[:x.shape[1], :]).to(x.dtype)
        outs = [x]
        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for block in e.blocks:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
            outs.append(x)
        y = e.avg_pooler(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = e.after_norm(y.contiguous())
        n50 = sum(1 for k in range((T + 1) % 2, T, 2) if k < int(wav_lens[0]))
        if n50 != FEATS_PER_WIN * ENC_PER_FEAT:
            raise RuntimeError(f"expected 20 valid encoder frames per window, got {n50}")
        return outs, y, (wav_lens + 1) // 2 // 2

    def embeds(self, ada, seq=None):
        """LycheeFD.forward's input embedding (training path) with ``ada`` [T, H] in the patch slots.
        ``seq`` (lyspk.dialog.build_sequence) replaces the listening text / stoken / control channels."""
        n_frames = ada.shape[0]
        n_chunks = n_frames // FEATS_PER_WIN
        text, stoken, control, audio = listening_suffix(n_chunks)
        if seq is not None:
            if len(seq["text"]) != len(text):
                raise ValueError(f"sequence has {len(seq['text'])} slots, audio gives {len(text)}")
            text, stoken, control = (np.asarray(seq[k]).tolist() for k in ("text", "stoken", "control"))
        dev = self.device
        ids = lambda x: torch.tensor(x, dtype=torch.long, device=dev)  # noqa: E731
        E = self.embed
        pre = E(ids(self.prefix_ids))
        suf = E(ids(text))
        if self.no_text_label:
            suf = torch.zeros_like(suf)
        if not self.no_stoken_label:
            suf = suf + E(ids(stoken))
        suf = suf + E(ids(control))
        aud = E(ids(audio))
        patch = torch.nonzero(ids(audio) == TOK["audio_patch"]).squeeze(1)
        assert len(patch) == n_frames and bool((patch == 2 * torch.arange(n_frames, device=dev)).all())
        aud = aud.index_copy(0, patch, ada.to(aud.dtype))
        return torch.cat([pre, suf + aud], 0).unsqueeze(0), len(self.prefix_ids)

    def run_trunk(self, embeds, full=False):
        """Outputs of trunk layers 1..L at every position, via forward hooks: [L, S, H].
        ``full``: run every trunk layer and also return the model output (hidden_states and the
        final normed state); otherwise stop right after layer L."""
        got = {}
        stop_at = self.llm_layers - 1 if not full and len(self.lm.layers) > self.llm_layers else None

        def hook(mod, args, out, i):
            got[i] = out[0] if isinstance(out, tuple) else out
            if i == stop_at:
                raise _Stop

        hooks = [layer.register_forward_hook(lambda mod, args, out, i=i: hook(mod, args, out, i))
                 for i, layer in enumerate(list(self.lm.layers)[:self.llm_layers])]
        res = None
        try:
            res = self.lm(inputs_embeds=embeds, use_cache=False, return_dict=True, output_hidden_states=full)
        except _Stop:
            pass
        finally:
            for h in hooks:
                h.remove()
        layers = [got[i][0] for i in range(self.llm_layers)]
        return (layers, res) if full else layers

    def _ce(self, h, labels, lo, hi):
        """LycheeFD.clamp_loss_fn (logits[i] predicts labels[i + 1], vocab cut to [lo, hi)) with the
        LM head applied only where a label exists. Returns (loss, n, n_correct)."""
        lab = labels[1:]
        pos = torch.nonzero(lab != IGNORE).squeeze(1)
        if len(pos) == 0:
            return h.sum() * 0.0, 0, 0
        w = self.lm_head.weight[lo:hi]
        logits = (h[:-1][pos] @ w.T).float()
        target = lab[pos] - lo
        loss = F.cross_entropy(logits, target)
        return loss, int(len(pos)), int((logits.argmax(-1) == target).sum())

    def dialogue_losses(self, res, emb, seq, n_prefix):
        """Text, stoken and control CE (LycheeFD.forward's three branches) for one sequence."""
        dev = emb.device
        pad = lambda a: torch.cat([torch.full((n_prefix,), IGNORE, dtype=torch.long, device=dev),  # noqa: E731
                                   torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)])
        hs = res.hidden_states
        out = {}
        h_text = res.last_hidden_state[0]
        out["text"] = self._ce(h_text, pad(seq["text_label"]), 0, self.lm_head.weight.shape[0])
        st = self.stoken_model(inputs_embeds=hs[self.stoken_index], use_cache=False, return_dict=True).last_hidden_state
        ids = torch.cat([torch.tensor(self.prefix_ids, dtype=torch.long, device=dev),
                         torch.as_tensor(np.asarray(seq["text"]), dtype=torch.long, device=dev)])
        txt = self.embed(ids).detach()
        shifted = torch.zeros_like(txt)
        shifted[:-1] = txt[1:]
        mg = self.merge_model(inputs_embeds=st + shifted.unsqueeze(0), use_cache=False,
                              return_dict=True).last_hidden_state[0]
        out["stoken"] = self._ce(mg, pad(seq["stoken_label"]), *STOKEN_RANGE)
        ct = self.control_model(inputs_embeds=hs[self.control_index], use_cache=False,
                                return_dict=True).last_hidden_state[0]
        out["control"] = self._ce(ct, pad(seq["control_label"]), *CONTROL_RANGE)
        if "control_label_start" in seq:      # eval: start_* positions only
            out["control_start"] = self._ce(ct, pad(seq["control_label_start"]), *CONTROL_RANGE)
        return out

    def features(self, wave, seq=None, losses=False):
        """``wave`` (16 kHz, multiple of 0.4 s) -> {'stacks': {'enc': [33, T, 1280], 'llm': [L+1, T, 3584]},
        'keep': the preservation target, 'T': frames}. ``seq``: agent tokens in the suffix
        (lyspk.dialog); with ``losses`` (dialogue mode) also 'dlg': {text|stoken|control: (loss, n, correct)}."""
        wavs, lens = self.windows(wave)
        outs, pooled, _ = self.run_encoder(wavs, lens)
        n_win = wavs.shape[0]
        T = n_win * FEATS_PER_WIN

        def to12(o):
            o = o[:, :FEATS_PER_WIN * ENC_PER_FEAT]
            return o.reshape(n_win, FEATS_PER_WIN, ENC_PER_FEAT, -1).mean(2).reshape(T, -1)

        stacks = {"enc": torch.stack([to12(o) for o in outs])}
        ada = self.adapter(pooled)[:, :FEATS_PER_WIN].reshape(T, -1)
        if not self.llm_layers:
            return {"stacks": stacks, "keep": ada, "T": T}
        emb, n_prefix = self.embeds(ada, seq)
        full = bool(losses and self.dialogue and seq is not None)
        got = self.run_trunk(emb, full=full)
        layers, res = got if full else (got, None)
        slots = n_prefix + 2 * torch.arange(T, device=emb.device)
        stacks["llm"] = torch.stack([emb[0, slots]] + [h[slots] for h in layers])
        out = {"stacks": stacks, "keep": layers[-1][n_prefix:], "T": T}
        if full:
            out["dlg"] = self.dialogue_losses(res, emb, seq, n_prefix)
        return out


def keep_loss(h, ref):
    """Relative squared drift from the frozen model: 0 = unchanged."""
    h, ref = h.float(), ref.float()
    return ((h - ref) ** 2).mean() / (ref ** 2).mean().clamp_min(1e-6)


def check_token_ids(cfg, tok):
    """The ids lyspk.dialog hard-codes must be the checkpoint's."""
    want = {"start_speaking": cfg.start_speaking_token_id, "start_listening": cfg.start_listening_token_id,
            "keep_listening": cfg.keep_listening_token_id, "keep_speaking": cfg.keep_speaking_token_id,
            "detect": cfg.detect_token_id, "sleep": cfg.sleep_token_id, "text_pad": cfg.text_pad_token_id,
            "stoken_pad": cfg.stoken_pad_token_id, "stoken_delay": cfg.stoken_delay_token_id,
            "tts_start": tok.convert_tokens_to_ids("<tts_start>"), "tts_end": tok.convert_tokens_to_ids("<tts_end>"),
            "tts_pad": tok.convert_tokens_to_ids("<tts_pad>"), "eot": tok.convert_tokens_to_ids("<|EOT|>"),
            "audio_patch": tok.convert_tokens_to_ids("<audio_patch>")}
    bad = {k: (DTOK[k], v) for k, v in want.items() if int(v) != DTOK[k]}
    if tok.convert_tokens_to_ids("<audio_0>") != 151696:
        bad["<audio_0>"] = (151696, tok.convert_tokens_to_ids("<audio_0>"))
    if (cfg.stoken_token_ids_min, cfg.stoken_token_ids_max) != STOKEN_RANGE:
        bad["stoken_range"] = (STOKEN_RANGE, (cfg.stoken_token_ids_min, cfg.stoken_token_ids_max))
    if (cfg.control_token_ids_min, cfg.start_bc_token_id + 1) != CONTROL_RANGE:
        bad["control_range"] = (CONTROL_RANGE, (cfg.control_token_ids_min, cfg.start_bc_token_id + 1))
    if bad:
        raise SystemExit(f"token ids differ from lyspk/dialog.py (expected, checkpoint): {bad}")


def tiny_text_fn(text):
    return [100 + (ord(c) % 1000) for c in str(text)]


def tiny_host(code_dir, llm_layers=2, seed=0, grad_ckpt=True, dialogue=False):
    """A randomly initialised host with Lychee's exact module types at toy sizes, for CPU
    tests and ``train.py --host tiny`` (no checkpoint needed). ``dialogue``: a 4-layer trunk
    with 1-layer stoken / merge / control branches (at trunk layers 3 / - / 2) and an LM head."""
    from transformers import Qwen2Config, Qwen2Model

    m = _import_lychee(code_dir)
    torch.manual_seed(seed)
    enc = m.AudioEncoder(n_mels=128, n_ctx=1500, n_state=64, n_head=4, n_layer=3)
    ada = m.Adaptor(64, 96, kernel_size=3, stride=2)
    lm = embed = branches = None
    vocab = max(max(TOK.values()), max(DTOK.values())) + 8

    def qwen(n):
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=96, intermediate_size=192, num_hidden_layers=n,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
        cfg._attn_implementation = "eager"
        return Qwen2Model(cfg)

    if llm_layers:
        lm = qwen(4 if dialogue else llm_layers)
        embed = lm.embed_tokens
    if dialogue:
        branches = {"stoken": qwen(1), "merge": qwen(1), "control": qwen(1),
                    "lm_head": nn.Linear(96, vocab, bias=False), "stoken_index": 3, "control_index": 2}
        for k in ("stoken", "merge", "control"):    # the branches take embeddings, never ids
            branches[k].embed_tokens = None
    host = LycheeHost(enc, ada, lm, embed, prefix_ids=[1, 2, 3, 4], llm_layers=llm_layers,
                      grad_ckpt=grad_ckpt, mask_fns=(m.make_non_pad_mask, m.mask_to_bias), branches=branches,
                      text_fn=tiny_text_fn if dialogue else None, stoken_delay=STOKEN_DELAY)
    host.meta = {"model_dir": "tiny", "trunk_layers": 4 if dialogue else llm_layers, "dialogue": bool(dialogue)}
    return host
