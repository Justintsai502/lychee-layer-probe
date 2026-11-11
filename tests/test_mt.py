"""Multi-track data, agent turns and Lychee's own dialogue loss (lyspk.mt_data / stoken / dialog).

  python -m unittest tests.test_mt -v

The key check: ``LycheeHost.dialogue_losses`` gives the same three losses as
``LycheeFD.forward`` itself, on a randomly initialised LycheeFD built from the upstream
class at toy sizes.
"""

import importlib.util
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lyspk.dialog import (IGNORE, TOK, build_sequence, crop_sequence, mask_control_label,  # noqa: E402
                          place_turns, snap_start)

CODE = ROOT / "third_party/lychee_fd_model"
HAVE = {m: importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "soundfile", "librosa")}
MODEL_OK = all(HAVE.values()) and (CODE / "LycheeFD.py").exists()
VOICES = ["v_anna", "v_bert", "v_cleo", "v_dan"]


def fake_tokenizer(audios):
    """25 Hz, like the real one: one token per 640 samples at 16 kHz."""
    return [[(i * 7 + int(abs(a[:640 * (i + 1)]).sum() * 10)) % 6561 for i in range(max(1, len(a) // 640))]
            for a in audios]


def write_scenes(root, n=4, seconds=14.0, sr=24000):
    """Scene dirs as mpmoshi's IF set has them, plus the mpmoshi-style manifest."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    rows = []
    for k in range(n):
        scene = f"scene_{k}"
        d = root / "if" / scene
        d.mkdir(parents=True)
        letters = "ABC" if k % 2 else "ABCD"
        voices = {L: VOICES[(i + k) % 4] for i, L in enumerate(letters)}
        script, t = [], 0.3
        i = 0
        while t < seconds - 2:
            L = letters[i % len(letters)]
            dur = float(rng.uniform(1.0, 2.5))
            script.append({"utt_id": f"u{i}", "speaker": L, "start": round(t, 3), "end": round(t + dur, 3),
                           "text": f"utterance {i} by {L}"})
            t += dur + float(rng.uniform(-0.3, 1.5))
            i += 1
        for L in letters:
            x = np.zeros(int(seconds * sr), np.float32)
            f0 = 120 + 40 * VOICES.index(voices[L])
            for u in script:
                if u["speaker"] == L:
                    a, b = int(u["start"] * sr), int(u["end"] * sr)
                    x[a:b] = 0.3 * np.sin(2 * np.pi * f0 * np.arange(b - a) / sr)
            sf.write(d / f"audio_qwen3-tts-1.7b-clone_{L}.wav", x, sr)
        (d / "aligned_script.jsonl").write_text("\n".join(json.dumps(u) for u in script) + "\n")
        (d / "voices.json").write_text(json.dumps({L: f"/prompts/{v}.wav" for L, v in voices.items()}))
        rows.append({"audio_path": f"{scene}/audio_qwen3-tts-1.7b-clone_D.wav", "duration": seconds,
                     "session_id": scene, "meta": {"source": "if_4spk", "agent_wav": f"{scene}/x_D.wav"}})
    man = root / "mp_manifest.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return man


def write_fake_kit(d, sr=24000):
    """A two-task mpbench_mp100_eval_kit layout plus the MP-Bench audio tree it points at."""
    import soundfile as sf
    kit, audio = d / "kit", d / "mpbench"
    (kit / "subset").mkdir(parents=True)
    (kit / "gold_speakers").mkdir()
    (kit / "scripts").mkdir()
    (kit / "scripts" / "score_spk.py").write_text("import sys; print(' '.join(sys.argv[1:]))\n")
    rows = ["task\tdataset_index\tfamily\tshould_speak\tis_speaker_name\tquestion_type\tinput_dur_s\tn_gold_frames\tchairs"]
    rng = np.random.default_rng(3)
    for task, n in (("task_a", 131), ("task_b", 77)):
        spk = np.repeat(rng.integers(0, 4, n // 6 + 1), 6)[:n]
        np.savez(kit / "gold_speakers" / f"{task}.npz", spk=spk, frame_rate=np.float32(12.5))
        t = np.arange(int(n / 12.5 * sr)) / sr
        x = np.zeros_like(t, dtype=np.float32)
        for k, f0 in ((1, 140), (2, 220), (3, 330)):
            on = np.repeat(spk == k, int(sr / 12.5))[: len(t)]
            x[: len(on)] += 0.3 * on * np.sin(2 * np.pi * f0 * t[: len(on)])
        w = audio / "tasks" / "data" / task
        w.mkdir(parents=True)
        sf.write(w / "all_stitched.wav", x, sr)
        rows.append(f"{task}\t0\tunderstanding\ttrue\ttrue\tq\t{n / 12.5:.3f}\t{n}\t1=A;2=B;3=C")
    (kit / "subset" / "instances.tsv").write_text("\n".join(rows) + "\n")
    return kit, audio


def build_turns_manifest(d):
    """mpmoshi manifest -> lyspk.mt_data scenes -> turns with the fake tokenizer."""
    from lyspk import mt_data, stoken
    from lyspk.data import load_manifest
    man = write_scenes(d)
    mt = d / "mt.jsonl"
    mt_data.scenes_main(["--manifest", str(man), "--root", str(d / "if"), "--source", "if", "--out", str(mt)])
    rows = load_manifest(mt)
    out = []
    for r in rows:
        f = d / "turns" / f"{stoken.safe_name(r['id'])}.json"
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps({"id": r["id"], "turns": stoken.tokenize_row(r, fake_tokenizer, 1.0, 20.0)}))
        out.append({k: v for k, v in r.items() if not k.startswith("_")} | {"turns": str(f)})
    turns_man = d / "mt.turns.jsonl"
    turns_man.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    return mt, turns_man


class TestDialogSequence(unittest.TestCase):
    def test_layout_follows_lychee(self):
        turns = [{"start": 2.0, "end": 3.0, "text": "hi", "stoken": list(range(25))},
                 {"start": 3.4, "end": 4.0, "text": "ok", "stoken": list(range(12))}]
        placed, st = place_turns(turns)
        (a0, b0, _), (a1, b1, _) = placed
        self.assertEqual(a0, round((2.0 - 0.44) / 0.4))
        self.assertEqual(b0 - a0, int(np.ceil((10 + 2 + 25 + 1) / 10)))
        self.assertGreaterEqual(a1, b0 + 1)                       # one listening chunk in between
        self.assertEqual(st["pushed"], 1)
        seq = build_sequence(16, 0, placed, lambda s: [ord(c) for c in s])
        ctrl = seq["control"][9::10]
        self.assertEqual(ctrl[a0 - 1], TOK["start_speaking"])
        self.assertEqual(ctrl[b0 - 1], TOK["start_listening"])
        self.assertTrue(all(c == TOK["keep_speaking"] for c in ctrl[a0:b0 - 1]))
        self.assertTrue((seq["control"][8::10] == TOK["detect"]).all())
        lo = a0 * 10
        self.assertEqual(list(seq["stoken"][lo:lo + 11]), [TOK["stoken_delay"]] * 10 + [TOK["tts_start"]])
        self.assertEqual(seq["stoken"][lo + 11], 151696)
        self.assertEqual(list(seq["text"][lo:lo + 3]), [ord("h"), ord("i"), TOK["eot"]])
        self.assertEqual(seq["text"][lo + 3], TOK["tts_pad"])
        self.assertEqual(seq["text_label"][lo + 3], IGNORE)          # tts_pad not a label
        self.assertEqual(seq["stoken_label"][lo + 10], IGNORE)       # tts_start not a label
        self.assertEqual(seq["stoken_label"][lo + 11 + 25], TOK["tts_end"])
        self.assertEqual(int((seq["stoken_label"] != IGNORE).sum()), 25 + 1 + 12 + 1)

    def test_crop_never_starts_inside_a_turn(self):
        turns = [{"start": 2.0, "end": 6.0, "text": "a", "stoken": list(range(100))}]
        placed, _ = place_turns(turns)
        a, b, _ = placed[0]
        self.assertEqual(snap_start(a + 1, placed), b)
        self.assertEqual(snap_start(a - 1, placed), a - 1)
        f0, seq, _ = crop_sequence(turns, 5 * (a + 1), 5 * (b + 10), lambda s: [1], train=False)
        self.assertEqual(f0, 5 * b)
        self.assertFalse(seq["speaking"].any())
        with self.assertRaises(ValueError):
            build_sequence(10, a + 1, placed, lambda s: [1])

    def test_agent_weight_is_mpmoshis(self):
        from lyspk.dialog import agent_active_fraction, agent_text_weight
        segs = [{"start": 0.0, "end": 0.8}, {"start": 4.0, "end": 4.4}]       # 10 + 5 frames
        self.assertAlmostEqual(agent_active_fraction(segs, 0, 100), 0.15)
        self.assertAlmostEqual(agent_active_fraction(segs, 50, 100), 5 / 50)
        self.assertEqual(agent_text_weight(0.15), 1.0)
        self.assertEqual(agent_text_weight(0.5), 1.0)
        self.assertAlmostEqual(agent_text_weight(0.075), 0.5)
        self.assertEqual(agent_text_weight(0.0), 0.1)

    def test_control_mask_rule(self):
        lab = np.full(200, IGNORE)
        seq = [TOK["keep_listening"]] * 10 + [TOK["start_speaking"]] + [TOK["keep_speaking"]] * 6 + \
              [TOK["start_listening"]] + [TOK["keep_listening"]] * 2
        lab[9::10] = seq
        m = mask_control_label(lab, rng=random.Random(0))
        kept = m[9::10]
        for i in (0, 9, 10, 11, 16, 17, 18, 19):                 # starts and run ends always kept
            self.assertNotEqual(kept[i], IGNORE, i)
        n_normal = 8 + 4                                           # interiors of the two longer runs
        self.assertEqual(int((kept != IGNORE).sum()), 8 + round(0.2 * n_normal))


@unittest.skipUnless(MODEL_OK, "needs torch, transformers, soundfile, librosa and third_party/lychee_fd_model")
class TestMultiTrack(unittest.TestCase):
    def test_scenes_view_and_turns(self):
        from lyspk.data import load_frames, load_manifest, view
        with tempfile.TemporaryDirectory() as d:
            mt, turns_man = build_turns_manifest(Path(d))
            rows = load_manifest(turns_man)
            self.assertEqual(len(rows), 4)
            r = rows[0]
            self.assertEqual(sorted(r["tracks"]), list("ABCD"))
            self.assertEqual(set(r["speakers"].values()), set(VOICES))
            v = view(r, "B")
            self.assertEqual(len(v["wav_paths"]), 3)
            self.assertNotIn(r["speakers"]["B"], {s["speaker"] for s in v["segments"]})
            self.assertTrue(v["turns"])
            for t in v["turns"]:                                    # 25 Hz, the agent's own track
                self.assertAlmostEqual(len(t["stoken"]) / (t["end"] - t["start"]), 25, delta=2)
            x = load_frames(v, 0, 50)
            self.assertEqual(len(x), 50 * 1280)
            alone = load_frames({"wav_paths": v["wav_paths"][:1]}, 0, 50)
            self.assertGreater(np.abs(x).sum(), np.abs(alone).sum())

    def test_dialogue_losses_match_lycheefd(self):
        import torch
        sys.path.insert(0, str(CODE.parent))
        from lychee_fd_model.LycheeFD import LycheeFD, LycheeFDConfig
        import lychee_fd_model.modeling_step_audio_2 as m
        from lyprobe.layout import listening_suffix
        from lyspk.host import LycheeHost, tiny_text_fn

        V = 158370
        q = lambda n: dict(vocab_size=V, hidden_size=96, intermediate_size=192, num_hidden_layers=n,  # noqa: E731
                           num_attention_heads=4, num_attention_groups=2, num_key_value_heads=2,
                           torch_dtype="float32")
        cfg = LycheeFDConfig(
            start_speaking_token_id=TOK["start_speaking"], keep_listening_token_id=TOK["keep_listening"],
            start_listening_token_id=TOK["start_listening"], keep_speaking_token_id=TOK["keep_speaking"],
            detect_token_id=TOK["detect"], sleep_token_id=TOK["sleep"], text_pad_token_id=TOK["text_pad"],
            stoken_pad_token_id=TOK["stoken_pad"], audio_pad_token_id=TOK["audio_pad"],
            stoken_delay_token_id=TOK["stoken_delay"], start_bc_token_id=TOK["start_bc"],
            stoken_delay_num=10, stoken_token_ids_min=151694, stoken_token_ids_max=158352,
            control_token_ids_min=158352, control_token_ids_max=158356, control_branch_layer=2,
            text_config=q(4), stoken_layer_config=q(1), control_layer_config=q(1), merge_layer_config=q(1),
            audio_encoder_config=dict(n_mels=128, n_audio_ctx=1500, n_audio_state=64, n_audio_head=4,
                                      n_audio_layer=2, llm_dim=96, kernel_size=3, adapter_stride=2),
            torch_dtype="float32")
        for c in (cfg, cfg.text_config, cfg.stoken_layer_config, cfg.control_layer_config, cfg.merge_layer_config):
            c._attn_implementation = "eager"
        torch.manual_seed(0)
        model = LycheeFD(cfg).float().eval()
        self.assertEqual((model.stoken_model_index, model.control_model_index), (3, 2))
        prefix = [151666, 10, 20, 151665]
        host = LycheeHost(model.encoder, model.adapter, model.model, model.model.embed_tokens, prefix, llm_layers=2,
                          grad_ckpt=False, mask_fns=(m.make_non_pad_mask, m.mask_to_bias),
                          branches={"stoken": model.stoken_model, "merge": model.merge_model,
                                    "control": model.control_model, "lm_head": model.lm_head,
                                    "stoken_index": model.stoken_model_index,
                                    "control_index": model.control_model_index},
                          text_fn=tiny_text_fn, stoken_delay=10).eval()
        rng = np.random.default_rng(1)
        n_chunks = 14
        turns = [{"start": 1.5, "end": 2.6, "text": "hello there", "stoken": list(rng.integers(0, 6561, 27))},
                 {"start": 4.0, "end": 4.6, "text": "yes", "stoken": list(rng.integers(0, 6561, 15))}]
        placed, _ = place_turns(turns)
        seq = build_sequence(n_chunks, 0, placed, tiny_text_fn)
        seq["control_label"] = mask_control_label(seq["control_label"], rng=random.Random(0))
        wave = (0.1 * rng.standard_normal(n_chunks * 6400)).astype(np.float32)
        with torch.no_grad():
            mine = host.features(wave, seq, losses=True)["dlg"]
            wavs, lens = host.windows(wave)
            audio = listening_suffix(n_chunks)[3]
            P = len(prefix)
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long).unsqueeze(0)  # noqa: E731
            lab = lambda a: t(np.concatenate([np.full(P, IGNORE), np.asarray(a)]))  # noqa: E731
            sys.modules.setdefault("wandb", types.SimpleNamespace(run=None))
            model.lm_head.weight.requires_grad_(True)      # forward skips the control loss otherwise
            ref = model(input_ids=t(prefix + list(seq["text"])), stoken_ids=t(seq["stoken"]),
                        control_input_ids=t(seq["control"]), audio_input_ids=t(audio), prefix_input_ids=t(prefix),
                        wavs=wavs, wav_lens=lens, labels=lab(seq["text_label"]),
                        stoken_label_ids=lab(seq["stoken_label"]), control_label_ids=lab(seq["control_label"]))
        self.assertAlmostEqual(float(mine["text"][0]), float(ref.loss_text), places=4)
        self.assertAlmostEqual(float(mine["stoken"][0]), float(ref.loss_stoken), places=4)
        self.assertAlmostEqual(float(mine["control"][0]), float(ref.loss_control), places=4)
        # stopping after layer L gives the readout the same layers as the full forward
        with torch.no_grad():
            short = host.features(wave, seq)
            full = host.features(wave, seq, losses=True)
        self.assertNotIn("dlg", short)
        self.assertTrue(torch.allclose(short["stacks"]["llm"], full["stacks"]["llm"], atol=1e-5))

    def test_dialogue_gradient_reaches_encoder_lora(self):
        import torch
        from lyspk.host import tiny_host, tiny_text_fn
        host = tiny_host(CODE, llm_layers=2, dialogue=True)
        host.apply_lora(enc_rank=4, llm_rank=4, adaptor_rank=4)
        for n, p in host.named_parameters():                       # move B off zero so A gets a gradient
            if "lora_B" in n:
                p.data.normal_(0, 0.02)
        turns = [{"start": 1.0, "end": 2.0, "text": "go", "stoken": list(range(25))}]
        placed, _ = place_turns(turns)
        seq = build_sequence(8, 0, placed, tiny_text_fn)
        out = host.features(np.random.default_rng(0).standard_normal(8 * 6400).astype(np.float32) * 0.1,
                            seq, losses=True)
        d = out["dlg"]
        (d["text"][0] + d["stoken"][0] + d["control"][0]).backward()
        enc = [p.grad for n, p in host.named_parameters() if n.startswith("encoder.") and "lora_A" in n]
        self.assertTrue(enc and all(g is not None and float(g.abs().sum()) > 0 for g in enc))
        frozen = [p for n, p in host.named_parameters() if "lora_" not in n]
        self.assertTrue(all(not p.requires_grad for p in frozen))

    def test_batched_mel_equals_per_window(self):
        import torch
        from lyspk.host import WIN, tiny_host
        host = tiny_host(CODE, llm_layers=2)
        wave = (0.1 * np.random.default_rng(3).standard_normal(7 * WIN)).astype(np.float32)
        wave[2 * WIN:3 * WIN] = 0.0                                      # a silent window: own max per window
        wavs, lens = host.windows(wave)
        ref = torch.stack([host.log_mel(wave[i:i + WIN]) for i in range(0, len(wave), WIN)])
        self.assertEqual(tuple(wavs.shape), tuple(ref.shape))
        self.assertTrue(torch.allclose(wavs.float(), ref.float(), atol=1e-5))
        self.assertTrue(bool((lens == ref.shape[2] - 2).all()))

    def test_prefetch_gives_the_same_run(self):
        from lyspk import train
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, man = build_turns_manifest(d)
            losses = {}
            for k in ("0", "3"):
                run = d / f"run{k}"
                train.main(["--host", "tiny", "--device", "cpu", "--dialogue", "--train", str(man), "--run-dir", str(run),
                            "--steps", "2", "--accum", "2", "--eval-every", "0", "--save-every", "0", "--log-every", "1",
                            "--crop-min", "6", "--crop-max", "10", "--enc-lora-rank", "4", "--llm-lora-rank", "4",
                            "--adaptor-lora-rank", "4", "--warmup", "1", "--prefetch", k])
                losses[k] = [json.loads(x)["loss"] for x in (run / "train.jsonl").read_text().splitlines()]
            self.assertEqual(losses["0"], losses["3"])

    def test_train_eval_dialogue(self):
        import torch
        from lyspk import evaluate, train
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _, man = build_turns_manifest(d)
            run = d / "run"
            train.main(["--host", "tiny", "--device", "cpu", "--dialogue", "--train", str(man), "--val", str(man),
                        "--run-dir", str(run), "--steps", "3", "--accum", "2", "--eval-every", "3",
                        "--save-every", "0", "--eval-clips", "2", "--eval-seconds", "12", "--log-every", "1",
                        "--crop-min", "6", "--crop-max", "10", "--enc-lora-rank", "4", "--llm-lora-rank", "4",
                        "--adaptor-lora-rank", "4", "--warmup", "1", "--llm-high-lora-rank", "2",
                        "--mode", "lychee_c5_id4"])
            rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(np.isfinite(r["loss"]) and "dlg_text" in r for r in rows))
            self.assertTrue(all(0.1 <= r["w_agent_text"] <= 1.0 and 0 <= r["agent_frac"] <= 1 for r in rows))
            rc = json.loads((run / "run_config.json").read_text())
            self.assertEqual(rc["lora_linears"]["llm_high"], 2 * 7)      # tiny trunk layers 3-4, 7 linears each
            self.assertEqual(rc["head_spec"]["mode"], "lychee_c5_id4")
            self.assertTrue(all(r["keep"] == 0.0 for r in rows))           # keep defaults off with --dialogue
            ev = json.loads((run / "eval.jsonl").read_text().splitlines()[-1])
            self.assertIn("dlg", ev)
            self.assertGreater(ev["dlg"]["control"]["n"], 0)
            self.assertGreater(ev["dlg"]["control_start"]["n"], 0)
            blob = torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)
            self.assertTrue(blob["host_config"]["dialogue"])
            host, readout, head, spec = evaluate.build_from_blob(blob, "cpu", CODE)
            res = evaluate.stream_scores(host, readout, head, spec, evaluate.eval_windows(
                evaluate.load_manifest(man), 12, 2, dialogue=True))
            self.assertAlmostEqual(res["dlg"]["text"]["ce"], ev["dlg"]["text"]["ce"], places=5)
            torch.set_grad_enabled(True)

            # mpmoshi's diagnostic battery on the same checkpoint
            from lyspk import diag
            out = d / "diag"
            s = diag.main(["--ckpt", str(run / "ckpt_final.pt"), "--manifest", f"tt={man}", "--out", str(out),
                           "--device", "cpu", "--window-sec", "12", "--sessions", "3", "--boot", "50"])
            for f in ("summary.json", "per_window.json", "meta.json", "preds.npz", "hspk.npz"):
                self.assertTrue((out / f).exists(), f)
            self.assertEqual(s["n_windows"], 3)
            self.assertEqual(set(s["by_variant"]), {"all", "tt"})
            want = set(diag.LIVE) | set(diag.BANKS)
            self.assertEqual(set(s["by_variant"]["all"]), want)
            per = json.loads((out / "per_window.json").read_text())
            for name in want:                 # every variant is counted over the same frames
                self.assertEqual([r["n_single"] for r in per[name]], [r["n_single"] for r in per["self"]])
                self.assertEqual([r["change_n"] for r in per[name]], [r["change_n"] for r in per["self"]])
                for m in ("frame_acc", "turn_acc", "change_acc"):
                    v = s["by_variant"]["all"][name][m]
                    self.assertTrue(v is None or 0.0 <= v <= 1.0)
            self.assertIn("gold-wrong", s["paired"]["tt"])
            self.assertEqual(s["paired"]["tt"]["gold-empty"]["frame"]["n_groups"], 3)
            pz = np.load(out / "preds.npz")       # the policies really feed the head different caches
            for a, b in (("gold", "empty"), ("gold", "wrong"), ("self", "silenced"), ("spread", "one only")):
                self.assertFalse(all(np.array_equal(pz[f"prob_{a}_{i}"], pz[f"prob_{b}_{i}"]) for i in range(3)), (a, b))
            torch.set_grad_enabled(True)

            # MP-Bench 100 speaker-id on a fake kit of two tasks
            from lyspk import mpbench_spk
            kit, audio = write_fake_kit(d)
            mp = d / "mp100"
            s = mpbench_spk.main(["--ckpt", str(run / "ckpt_final.pt"), "--kit", str(kit), "--audio-root", str(audio),
                                  "--out", str(mp), "--device", "cpu"])
            self.assertEqual(s["n_tasks"], 2)
            for name in mpbench_spk.VARIANTS:
                for t, n in (("task_a", 131), ("task_b", 77)):
                    ids = np.load(mp / f"preds_{name}" / f"{t}.npz")["ids"]
                    self.assertEqual(len(ids), n)                     # cut / padded to the gold length
                    self.assertTrue(((ids >= 0) & (ids <= 7)).all())
                self.assertEqual(s["by_variant"][name]["frames"], 131 + 77)
                self.assertIn("--pred-dir", (mp / f"score_{name}_w12.txt").read_text())
            pz = np.load(mp / "probs.npz")
            self.assertFalse(np.array_equal(pz["gold__task_a"], pz["empty__task_a"]))
            torch.set_grad_enabled(True)

    def test_mpbench_id_mapping(self):
        from lyspk.mpbench_spk import KIT_TO_BITS, gold_tracks, kit_ids, score
        prob = KIT_TO_BITS.astype(np.float32) * 0.9            # every kit id, as probabilities
        self.assertEqual(kit_ids(prob, 0.5).tolist(), list(range(8)))
        gold = np.array([0, 1, 1, 2, 3, 3, 0])
        y, single, keep = gold_tracks(gold, 10)
        self.assertEqual(single.tolist(), [-1, 0, 0, 1, 2, 2, -1, -1, -1, -1])
        self.assertEqual(keep.sum(), 7)
        self.assertEqual(kit_ids(y, 0.5)[:7].tolist(), gold.tolist())
        sc = score(np.array([0, 1, 4, 2, 3, 1, 0, 2, 2]), gold)   # longer prediction is cut
        self.assertEqual((sc["n"], sc["ok"], sc["n_speech"], sc["ok_speech"]), (7, 5, 5, 3))
        self.assertEqual((sc["tp1"], sc["on_p1"], sc["on_g1"]), (2, 3, 2))

    def test_diag_metrics_on_gold(self):
        from lyspk.bank_sets import build_sets
        from lyspk.diag_metrics import per_window
        y = np.zeros((40, 3), int)
        y[0:10, 0] = 1
        y[8:20, 1] = 1
        y[25:40, 2] = 1
        keep = np.ones(40, bool)
        r = per_window(y.astype(float), y, keep, 0.3)
        self.assertEqual((r["frame_ok"], r["turn_mode_ok"], r["change_ok"]), (r["n_single"], r["turn_n"], r["change_n"]))
        self.assertEqual(r["turn_n"], 3)                   # per-slot runs, not runs of constant id
        single = np.where(y.sum(1) == 1, y.argmax(1), -1)
        sets = build_sets(single, keep, 3, 4)
        self.assertEqual(sets["wrong"][0], sets["spread"][1])
        self.assertEqual(sets["one only"][0], sets["one xK"][0][:1])


if __name__ == "__main__":
    unittest.main()
