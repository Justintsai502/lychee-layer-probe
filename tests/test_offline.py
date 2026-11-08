"""
Offline checks (numpy only):  python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lyprobe import analyze, analyze_eer, fake_features  # noqa: E402
from lyprobe.common import trim_silence  # noqa: E402
from lyprobe.extract_lychee import point_table  # noqa: E402
from lyprobe.extract_bayling_encoder import prepare_waveform_bayling  # noqa: E402
from lyprobe.extract_lychee_encoder import save_points  # noqa: E402
from lyprobe.layout import (TOK, compute_token_num, listening_suffix, mel_frames,  # noqa: E402
                            prepare_waveform, speech_slot_mask, window_frame_mask)
from lyprobe.metrics import (cosine_profile, eer, eer_cosine, eer_wccn, probe_accuracy,  # noqa: E402
                             trial_masks, variance_shares)


class LayoutTest(unittest.TestCase):
    def test_window_gives_five_features(self):
        # 0.4 s window -> 42 mel frames -> 5 encoder/adaptor features -> 10 slots
        self.assertEqual(mel_frames(6400), 42)
        self.assertEqual(compute_token_num(42), 5)

    def test_listening_chunk(self):
        text, stoken, control, audio = listening_suffix(3)
        self.assertTrue(len(text) == len(stoken) == len(control) == len(audio) == 30)
        self.assertEqual(control[:10], [TOK["sleep"]] * 8 + [TOK["detect"], TOK["keep_listening"]])
        self.assertEqual(audio.count(TOK["audio_patch"]), 15)
        self.assertEqual(audio[:2], [TOK["audio_patch"], TOK["audio_pad"]])
        self.assertTrue(all(t == TOK["text_pad"] for t in text))

    def test_prepare_waveform_chunk_aligned(self):
        sr = 16000
        x = np.zeros(sr * 3, np.float32)
        x[sr:2 * sr] = 0.5 * np.sin(np.arange(sr) * 0.05)
        wave, lead_s, speech_s, n_chunks = prepare_waveform(x, sr, 4, trim_silence)
        self.assertAlmostEqual(lead_s, 1.6)
        self.assertEqual(len(wave), n_chunks * 6400)
        self.assertEqual(float(np.abs(wave[:int(1.6 * sr)]).max()), 0.0)

    def test_speech_mask(self):
        m = speech_slot_mask(80, lead_s=1.6, speech_s=0.4)   # slots 40..49
        self.assertEqual(m.sum(), 10)
        self.assertTrue(m[40:50].all())


class PointTableTest(unittest.TestCase):
    def test_released_config_tree(self):
        pts = point_table(28, {"control": 20, "speech": 24}, {"control": 4, "speech": 4, "merge": 4})
        names = {n: (d, t) for n, d, t in pts}
        self.assertEqual(len(pts), 29 + 4 + 4 + 4)
        self.assertEqual(names["control_1"], (21, "control"))
        self.assertEqual(names["speech_4"], (28, "speech"))
        self.assertEqual(names["merge_4"], (32, "merge"))


class VarianceSharesTest(unittest.TestCase):
    def grid(self, a, b, seed=0, n_spk=10, n_sent=12, dim=40):
        rng = np.random.default_rng(seed)
        sv, tv = rng.normal(size=(n_spk, dim)), rng.normal(size=(n_sent, dim))
        X, spk, sent = [], [], []
        for s in range(n_spk):
            for t in range(n_sent):
                X.append(a * sv[s] + b * tv[t] + rng.normal(size=dim))
                spk.append(s)
                sent.append(t)
        return np.array(X), np.array(spk), np.array(sent)

    def test_noise_has_no_excess(self):
        r = variance_shares(*self.grid(0.0, 0.0))
        self.assertLess(r["excess_spk"], 0.02)
        self.assertLess(r["excess_sent"], 0.02)

    def test_speaker_only_and_sentence_only(self):
        self.assertGreater(variance_shares(*self.grid(2.0, 0.0))["acoustic_share"], 0.9)
        self.assertLess(variance_shares(*self.grid(0.0, 2.0))["acoustic_share"], 0.1)

    def test_weak_voice_under_strong_content_is_not_clipped(self):
        # the bug the real Lychee run exposed: sentence explains most of the variance,
        # a real but small speaker effect must still come out above zero
        r = variance_shares(*self.grid(0.4, 3.0, n_spk=25, n_sent=30, dim=64))
        self.assertGreater(r["excess_spk"], 0.0)
        self.assertTrue(0.0 < r["acoustic_share"] < 0.2)
        # and pure noise added to the same strong content still gives ~0 voice
        r0 = variance_shares(*self.grid(0.0, 3.0, n_spk=25, n_sent=30, dim=64))
        self.assertLess(r0["acoustic_share"], 0.005)

    def test_shares_add_up_to_at_most_one(self):
        r = variance_shares(*self.grid(1.0, 1.0))
        self.assertLessEqual(r["eta_spk"] + r["eta_sent"], 1.0 + 1e-9)
        self.assertTrue(0.3 < r["acoustic_share"] < 0.7)

    def test_cosine_profile_orders_pair_types(self):
        X, spk, sent = self.grid(2.0, 0.0)              # voice only
        c = cosine_profile(X, spk, sent)
        self.assertGreater(c["s_spk"], c["s_none"] + 0.3)
        self.assertLess(abs(c["s_txt"] - c["s_none"]), 0.05)
        X, spk, sent = self.grid(0.0, 2.0)              # words only
        c = cosine_profile(X, spk, sent)
        self.assertGreater(c["s_txt"], c["s_spk"] + 0.3)

    def test_probe_ignores_content(self):
        X, spk, sent = self.grid(0.0, 3.0)
        self.assertLess(probe_accuracy(X, spk, sent)[0], 0.3)


class EndToEndTest(unittest.TestCase):
    def test_recovers_planted_tree(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sys.argv = ["fake_features", "--out", str(d)]
            fake_features.main()
            sys.argv = ["analyze", "--features", str(d / "lychee_fake.npz"), "--out", str(d / "res"),
                        "--json", str(d / "res.json"), "--baselines", str(d / "none.npz"),
                        "--folds", "4", "--no-probe"]
            analyze.main()
            rows = {r.split(",")[1]: r.split(",") for r in (d / "res" / "metrics.csv").read_text().splitlines()[1:]}
            share = lambda p: float(rows[p][4])  # noqa: E731
            self.assertGreater(share("trunk_0"), share("trunk_20"))      # voice fades along the trunk
            self.assertGreater(share("merge_4"), share("speech_1"))      # speech branch brings it back
            self.assertTrue((d / "res" / "summary.md").exists())
            res = json.loads((d / "res.json").read_text())
            self.assertEqual(len(res["points"]), 41)
            self.assertEqual(res["n_speakers"], 8)


class EERTest(unittest.TestCase):
    def grid(self, a, b, n_spk=25, n_sent=30, dim=64, seed=0):
        return VarianceSharesTest.grid(None, a, b, seed, n_spk, n_sent, dim)

    def test_eer_extremes(self):
        self.assertEqual(eer([2, 3], [0, 1]), 0.0)
        self.assertEqual(eer([0, 1], [2, 3]), 1.0)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(eer(rng.normal(1, 1, 20000), rng.normal(0, 1, 200000)), 0.3085, delta=0.01)
        self.assertTrue(np.isnan(eer([], [1.0])))

    def test_trial_counts(self):
        _, spk, sent = self.grid(0, 0)
        t, n, h = trial_masks(spk, sent)
        self.assertEqual((t.sum(), n.sum(), h.sum()), (25 * 435, 750 * 696 // 2, 30 * 300))

    def test_voice_gives_low_eer_words_give_chance(self):
        X, spk, sent = self.grid(1.0, 0.0)
        self.assertLess(eer_cosine(X, spk, sent)[0], 0.05)
        self.assertLess(eer_wccn(X, spk, sent)[0], 0.05)
        X, spk, sent = self.grid(0.0, 2.0)
        self.assertAlmostEqual(eer_cosine(X, spk, sent)[0], 0.5, delta=0.03)
        self.assertGreater(eer_cosine(X, spk, sent)[1], 0.9)      # hard impostors win

    def test_wccn_recovers_voice_under_content(self):
        X, spk, sent = self.grid(0.5, 3.0)
        self.assertLess(eer_wccn(X, spk, sent)[0], eer_cosine(X, spk, sent)[0] - 0.1)


class EncoderLayoutTest(unittest.TestCase):
    def test_lychee_window_frames(self):
        # lead 4 chunks (1.6 s) + 0.4 s speech: only window 4 is speech, all 20 frames
        m = window_frame_mask(6, 20, 0.02, 1.6, 0.4)
        self.assertEqual(m.sum(), 20)
        self.assertTrue(m[4].all())
        self.assertEqual(window_frame_mask(6, 5, 0.08, 1.6, 0.4).sum(), 5)

    def test_bayling_waveform(self):
        sr = 16000
        x = np.zeros(sr * 2, np.float32)
        x[sr // 2:sr] = 0.5 * np.sin(np.arange(sr // 2) * 0.05)
        wave, lead_s, speech_s = prepare_waveform_bayling(x, sr, 2)
        self.assertAlmostEqual(lead_s, 1.6)
        self.assertEqual(len(wave) % int(0.8 * sr), 0)


class EEREndToEndTest(unittest.TestCase):
    def test_encoder_plus_llm(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sys.argv = ["fake_features", "--out", str(d)]
            fake_features.main()
            llm = np.load(d / "lychee_fake.npz")
            rng = np.random.default_rng(1)
            spk_ids = {s: i for i, s in enumerate(sorted(set(llm["speaker"].tolist())))}
            voice = rng.normal(size=(len(spk_ids), 32))
            feats = []
            for s in llm["speaker"]:
                v = voice[spk_ids[s]]
                feats.append([v + rng.normal(size=32) * 0.3, v + rng.normal(size=32) * 3.0])
            rows = list(zip(llm["utt_id"], llm["speaker"], llm["sentence_id"]))
            save_points(d / "enc.npz", ["enc_0", "enc_1"], [-2, -1], "encoder", feats, rows, {})
            out = d / "eer.json"
            sys.argv = ["analyze_eer", "--llm", str(d / "lychee_fake.npz"), "--encoder", str(d / "enc.npz"),
                        "--baselines", str(d / "none.npz"), "--json", str(out), "--folds", "4"]
            analyze_eer.main()
            res = json.loads(out.read_text())
            by = {p["point"]: p for p in res["points"]}
            self.assertEqual(len(res["points"]), 43)
            self.assertEqual(by["enc_0"]["depth"], -2)
            self.assertLess(by["enc_0"]["eer_cos"], by["enc_1"]["eer_cos"])
            self.assertLess(by["enc_0"]["eer_cos"], 0.1)


if __name__ == "__main__":
    unittest.main()
