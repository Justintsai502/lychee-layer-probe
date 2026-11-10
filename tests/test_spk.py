"""
Speaker-head checks.  python -m unittest tests.test_spk -v

numpy-only tests always run. Tests that need torch / transformers / soundfile (and the
Lychee code in third_party/) skip when those are missing; they build a randomly
initialised host with Lychee's exact module types at toy sizes (lyspk.host.tiny_host).
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lyspk.labels import activity_from_segments, assert_arrival, build_labels, crop_frames  # noqa: E402
from lyspk.metrics import counts, pool  # noqa: E402

CODE = ROOT / "third_party/lychee_fd_model"
HAVE = {m: importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "soundfile", "librosa")}
MODEL_OK = all(HAVE.values()) and (CODE / "LycheeFD.py").exists()


def segs_abc():
    return [{"speaker": "b", "start": 0.0, "end": 2.0}, {"speaker": "a", "start": 1.5, "end": 4.0},
            {"speaker": "c", "start": 5.0, "end": 6.0}, {"speaker": "d", "start": 7.0, "end": 8.0}]


class LabelTest(unittest.TestCase):
    def test_arrival_order_and_fourth_speaker_mask(self):
        act, ids = activity_from_segments(segs_abc(), 125)
        y, single, keep, slots, first = build_labels(act, ids, 0, 125, label_shift=0)
        self.assertEqual(slots, ["b", "a", "c"])                 # by first frame, not by name
        assert_arrival(y, first, "t")
        self.assertFalse(keep[int(7.0 * 12.5)])                   # 4th speaker masked
        self.assertTrue((single[keep & (y.sum(1) == 1)] >= 0).all())
        self.assertTrue((single[y.sum(1) > 1] == -1).all())       # overlap is not "single"

    def test_crop_on_window_boundaries(self):
        rng = np.random.default_rng(0)
        for n in (100, 1003, 5000):
            f0, f1 = crop_frames(n, rng, 30, 60, align=5)
            self.assertEqual((f0 % 5, f1 % 5), (0, 0))
            self.assertLessEqual(f1, n)
            self.assertLessEqual(f1 - f0, 750)

    def test_metrics(self):
        y = np.zeros((10, 3), np.float32)
        y[:5, 0] = 1
        y[5:, 1] = 1
        keep = np.ones(10, bool)
        perfect = counts(y, y, keep)
        self.assertEqual((perfect["single_ok"], perfect["change_ok"], perfect["change_n"]), (10, 1, 1))
        hold = y.copy()
        hold[5:6] = [1, 0, 0]                                     # one frame late at the change
        r = pool([counts(hold, y, keep)])
        self.assertEqual(r["change_acc"], 0.0)
        self.assertAlmostEqual(r["single_acc"], 0.9)
        rot = np.roll(y, 1, axis=1)                               # predicted slot k means gold k-1
        self.assertEqual(counts(rot, y, keep, relabel=(0, 1, 2))["single_ok"], 0)
        self.assertEqual(counts(rot, y, keep, relabel=(2, 0, 1))["single_ok"], 10)


class ItemsTest(unittest.TestCase):
    def setUp(self):
        if not HAVE["torch"]:
            self.skipTest("torch missing")

    def test_items_aligned_and_cache_weightless(self):
        from lyspk.head_spec import LYCHEE
        from lyspk.items import sample_batch
        act, ids = activity_from_segments(segs_abc() + [{"speaker": "b", "start": 9, "end": 14}], 200)
        y, single, keep, _, first = build_labels(act, ids, 0, 200, 0)
        items = sample_batch(y, single, keep, first, np.random.default_rng(0), spec=LYCHEE)
        self.assertGreater(len(items), 10)
        for b in items:
            chunk = b["src"][b["seg"] == 2]
            chunk = chunk[chunk >= 0]
            self.assertEqual(chunk[0] % 5, 0)
            self.assertEqual(float(b["w"][: LYCHEE.n_slots * LYCHEE.cache_k].sum()), 0.0)
        self.assertEqual(LYCHEE.seq_len, 199)

    def test_supcon_queue_same_speaker_left_out(self):
        import torch
        import torch.nn.functional as F
        from lyspk.contrast import supcon
        a = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1)
        tau = torch.tensor(0.1)
        base, _ = supcon(a, ["x", "x", "y"], [], [], tau)
        same_q, _ = supcon(a, ["x", "x", "y"], [], [(a[0], "x")] * 50, tau)   # "y" has no positive: only x anchors count
        other_q, st = supcon(a, ["x", "x", "y"], [], [(a[0], "z")] * 50, tau)
        torch.testing.assert_close(same_q, base)          # own-speaker queue rows: not negatives
        self.assertGreater(float(other_q), float(base))   # another speaker's rows: negatives
        self.assertEqual(st["n_pos"], 2)


@unittest.skipUnless(MODEL_OK, "needs torch, transformers, soundfile, librosa and third_party/lychee_fd_model")
class HostTest(unittest.TestCase):
    def wave(self, n_win=6, seed=0):
        rng = np.random.default_rng(seed)
        return (0.1 * rng.normal(size=n_win * 6400)).astype(np.float32)

    def test_encoder_loop_matches_audio_encoder(self):
        import torch
        from lyspk.host import tiny_host
        host = tiny_host(CODE, llm_layers=2).eval()
        wavs, lens = host.windows(self.wave())
        with torch.no_grad():
            outs, pooled, plen = host.run_encoder(wavs, lens)
            ref, rlen = host.encoder(wavs, lens)
        torch.testing.assert_close(pooled, ref)
        self.assertTrue(bool((plen == rlen).all()))
        self.assertEqual(len(outs), len(host.encoder.blocks) + 1)

    def test_features_shapes_and_causality(self):
        import torch
        from lyspk.host import tiny_host
        host = tiny_host(CODE, llm_layers=2).eval()
        w = self.wave(6)
        with torch.no_grad():
            a = host.features(w)
            w2 = w.copy()
            w2[4 * 6400:] = 0.3 * np.random.default_rng(9).normal(size=2 * 6400)   # change windows 4, 5
            b = host.features(w2)
        self.assertEqual(tuple(a["stacks"]["enc"].shape), (4, 30, 64))
        self.assertEqual(tuple(a["stacks"]["llm"].shape), (3, 30, 96))
        for g in ("enc", "llm"):                      # frames of windows 0..3 must not move
            torch.testing.assert_close(a["stacks"][g][:, :20], b["stacks"][g][:, :20])
            self.assertFalse(torch.allclose(a["stacks"][g][:, 20:], b["stacks"][g][:, 20:]))

    def test_lora_starts_at_identity_and_toggles(self):
        import torch
        from lyspk.host import tiny_host
        from lyspk.lora import lora_disabled, lora_params
        host = tiny_host(CODE, llm_layers=2).eval()
        w = self.wave(3)
        with torch.no_grad():
            before = host.features(w)["keep"]
            n = host.apply_lora(enc_rank=4, llm_rank=4, adaptor_rank=4)
            same = host.features(w)["keep"]
            for p in lora_params(host):
                p.add_(0.05 * torch.randn_like(p))
            moved = host.features(w)["keep"]
            with lora_disabled():
                off = host.features(w)["keep"]
        self.assertEqual(n, {"encoder": 3 * 6, "adaptor": 2, "llm": 2 * 7})
        torch.testing.assert_close(before, same)
        torch.testing.assert_close(before, off)
        self.assertFalse(torch.allclose(before, moved))
        frozen = [p for n_, p in host.named_parameters() if "lora_" not in n_]
        self.assertTrue(all(not p.requires_grad for p in frozen))

    def test_checkpointing_gives_same_gradients(self):
        import torch
        from lyspk.host import tiny_host
        from lyspk.lora import lora_params
        w = self.wave(3)
        grads = []
        for ckpt in (False, True):
            host = tiny_host(CODE, llm_layers=2, seed=1, grad_ckpt=ckpt)
            host.apply_lora(enc_rank=4, llm_rank=4, adaptor_rank=4)
            torch.manual_seed(5)
            with torch.no_grad():
                for p in lora_params(host):
                    p.normal_(0, 0.02)
            host.train()
            out = host.features(w)
            (out["stacks"]["enc"].float().pow(2).mean() + out["stacks"]["llm"].float().pow(2).mean()).backward()
            grads.append(torch.cat([p.grad.flatten() for p in lora_params(host)]))
        self.assertGreater(float(grads[0].abs().sum()), 0)
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def write_toy_corpus(d, n=4, seconds=12.0):
    """Recordings where each speaker is a different tone, so a tiny model can learn it."""
    import soundfile as sf
    rng = np.random.default_rng(0)
    freqs = {"s1": 180.0, "s2": 330.0, "s3": 520.0}
    rows = []
    for i in range(n):
        t, segs = 0.2, []
        wave = np.zeros(int(seconds * 16000), np.float32)
        order = ["s1", "s2", "s3", "s1", "s2"] if i % 2 else ["s2", "s1", "s2", "s3", "s1"]
        for spk in order:
            dur = rng.uniform(1.0, 2.0)
            a, b = int(t * 16000), int(min(seconds, t + dur) * 16000)
            tt = np.arange(b - a) / 16000
            wave[a:b] += 0.3 * np.sin(2 * np.pi * freqs[spk] * tt)
            segs.append({"speaker": spk, "start": t, "end": b / 16000})
            t = b / 16000 + rng.uniform(0.1, 0.6)
            if t > seconds - 1:
                break
        sf.write(str(d / f"r{i}.wav"), wave, 16000)
        rows.append({"id": f"r{i}", "wav": f"r{i}.wav", "duration": seconds, "segments": segs})
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d / "train.jsonl"


@unittest.skipUnless(MODEL_OK, "needs torch, transformers, soundfile, librosa and third_party/lychee_fd_model")
class TrainSmokeTest(unittest.TestCase):
    def test_train_eval_resume(self):
        import torch
        from lyspk import evaluate, train
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            man = write_toy_corpus(d)
            run = d / "run"
            args = ["--host", "tiny", "--device", "cpu", "--train", str(man), "--val", str(man),
                    "--run-dir", str(run), "--steps", "3", "--accum", "2", "--eval-every", "3",
                    "--save-every", "0", "--eval-clips", "2", "--eval-seconds", "10", "--log-every", "1",
                    "--enc-lora-rank", "4", "--llm-lora-rank", "4", "--adaptor-lora-rank", "4", "--warmup", "1"]
            train.main(args)
            rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(np.isfinite(r["loss"]) for r in rows))
            self.assertGreater(rows[0]["keep"] + rows[-1]["keep"], -1)       # logged
            ev = json.loads((run / "eval.jsonl").read_text().splitlines()[-1])
            self.assertIn("rotation_follow", ev)
            self.assertIsNotNone(ev["gold"]["single_acc"])
            blob = torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(blob["head_spec"]["mode"], "lychee_c5")
            # the checkpoint rebuilds into the same predictions
            host, readout, head, spec = evaluate.build_from_blob(blob, "cpu", CODE)
            res = evaluate.stream_scores(host, readout, head, spec,
                                         evaluate.eval_windows(evaluate.load_manifest(man), 10, 2))
            self.assertAlmostEqual(res["gold"]["single_acc"], ev["gold"]["single_acc"], places=6)
            torch.set_grad_enabled(True)
            # resume continues the step count
            train.main(args + ["--resume", str(run / "ckpt_final.pt")])
            rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)          # already at --steps 3: nothing new


if __name__ == "__main__":
    unittest.main()
