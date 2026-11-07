# lychee-layer-probe

How much of each Lychee-FD layer encodes **who is speaking** (acoustic) versus **what is
said** (semantic)?

Lychee-FD's paper argues that acoustic and semantic objectives conflict in deep layers.
It bases this on gradients (Fig. 2) and splits the network into heads because of it.
This repo measures the same question from the **representation** side, at every point
of the actual tree.

It is the sibling of
[bayling-speaker-probe](https://github.com/Justintsai502/bayling-speaker-probe) and reuses
its synthesized clips: 25 cloned voices × the same Harvard sentences.

## What Lychee-FD actually looks like

The structure below was read from the released `config.json` and weights
(`HIT-TMG/Lychee-FD`). It differs from the paper's §4.3 description.

```
input embeddings (text + speech + control + user audio, summed per 40 ms slot)
  │
  trunk  layers 1 … 28 ───────────────────────────────► text head
  │        ├── at layer 20 ──► control  4 layers ─────► control head
  │        └── at layer 24 ──► speech   4 layers ─► merge 4 layers ─► speech head
```

| | paper §4.3 | released config |
|---|---|---|
| trunk | 24 shared + 4 text | **28** (text head reads the end) |
| control branch | 2 layers | branches at **layer 20** (`control_branch_layer=8`), 4 layers |
| speech branch | 4 layers | branches at **layer 24**, 4 stoken + 4 merge layers |

There are 41 measured points: `trunk_0…28`, `control_1…4`, `speech_1…4` and `merge_1…4`.
Each point's depth counts layers from the input embedding.

## Input: exactly what the model saw while listening in training

The layout comes from `Training_package/Code/DataLoaders/LycheeFDDataset.py` at
commit `7bb1068`.

- **System prefix:** `<|BOT|>system\nYou are a helpful assistant.<|EOT|>`
- **Chunks:** every 0.4 s chunk has 10 slots of 40 ms. Each slot sums four embeddings:
  - text `text_pad`
  - speech `stoken_pad`
  - control `[sleep ×8][detect][keep_listening]`
  - audio `[<audio_patch>, audio_pad] ×5`
- **User audio:** cut into 0.4 s windows. Each window gets its own 128-bin log-mel and goes
  through the Whisper-large-v3-style encoder and the adaptor, giving 5 features per window.
  These land on the even slots.
- **Lead silence:** 1.6 s of silence goes before each sentence. Speech starts on a chunk
  boundary, so each slot can be labeled speech or silence.
- **Pooling:** hidden states are mean-pooled over the speech slots.

The backbone forward is `LycheeFD.forward` itself, so the channels and encoder features are
inserted by the model's own code. The branch models are then re-run with
`output_hidden_states=True`. The model code is fetched from the official repo at a pinned
commit (`scripts/fetch_lychee_code.sh`, Apache-2.0); the released weights match its
parameter names exactly.

## Metrics

| metric | question | notes |
|---|---|---|
| **acoustic_share** | Of what this layer encodes about the clips, how much is the voice? | Two-way variance decomposition on the speaker × sentence table, chance floor removed. 1 = only voice, 0 = only words. Never saturates. |
| eta_spk / eta_sent | Share of variance explained by speaker / by sentence | The two parts behind acoustic_share |
| spk_acc | Can the speaker be read out? | Ridge probe, CV grouped by sentence |
| spk_index | Which factor organizes the space? | s_spk − s_txt, as in bayling-speaker-probe |

The BayLing study used a sentence probe for the semantic side. It scored 1.000 at every
layer and so carried no information. The variance decomposition replaces it.

## Run on the server

Requirements:
- A GPU with ≥ 32 GB. The weights are ~25.6 GB in bf16.
- The `spkprobe` conda env from bayling-speaker-probe already has every dependency.
- Both repos sit side by side under the same parent directory.

Place this repo **next to bayling-speaker-probe**; every default path assumes that layout
and only reads from the sibling:

```
/work/b12901015/
  bayling-speaker-probe/work/tts/manifest.kept.jsonl      (the 980 clips)
  bayling-speaker-probe/work/features/baselines.npz       (ECAPA, Mimi)
  bayling-speaker-probe/work/features/bayling_silence.npz (BayLing, re-scored here)
  lychee-layer-probe/                                     (this repo)
```

```bash
cd /work/b12901015
git clone https://github.com/Justintsai502/lychee-layer-probe
cd lychee-layer-probe
conda activate spkprobe

# model (~26 GB) into the project; the vocoder is not needed
export HF_HOME=$PWD/cache/huggingface
hf download HIT-TMG/Lychee-FD --local-dir models/lychee_fd

# smoke test: 6 clips, no push
SMOKE=1 bash scripts/run_server.sh

# full run (980 clips) in the background; PUSH=1 commits and pushes results/*.json at the end
PUSH=1 nohup bash scripts/run_server.sh > /dev/null 2>&1 &
tail -f work/logs/run_server.log
```

Outputs:

| path | tracked | content |
|---|---|---|
| `results/lychee.json` | **yes** | every point's metrics + baselines, the file to push |
| `results/bayling.json` | **yes** | BayLing scored with the same metric |
| `work/features/lychee.npz` | no | pooled `[N, 41, 3584]` plus point names, depths, tracks |
| `work/results/<name>/` | no | `metrics.csv`, `summary.md`, `curves.png` |

Caches, models and logs stay inside the project and are git-ignored; only `results/*.json`
is meant to be committed.

## Plot the pushed results (laptop)

```bash
git pull
python lyprobe/plot_results.py      # -> docs/fig_acoustic_share.png, docs/fig_speaker_probe.png
```

## Offline check (laptop, numpy only)

```bash
python -m unittest discover -s tests -v
python lyprobe/fake_features.py --out work/fake
python lyprobe/analyze.py --features work/fake/lychee_fake.npz --out work/fake/res --no-probe
```

The tests check:
- the listening layout: 0.4 s gives 42 mel frames, which gives 5 features and 10 slots
- the point tree of the released config
- that pure noise has no excess variance
- that a speaker-only signal gives acoustic_share > 0.9 and a sentence-only signal < 0.1
- that the analysis recovers a planted tree

## Caveats

- **The model code has not been run here.** It was written against the official source
  and verified on layout, token ids and weight names, but no GPU was available for this.
  Run the smoke test first.
- **The labels are narrow.** "Acoustic" means speaker identity and "semantic" means sentence
  identity. Prosody, emotion and channel effects are measured by neither.
- **The listening state is fixed.** Every clip is fed in the listening state. While
  listening, the speech and merge branches only ever see `stoken_pad`, so they are measured
  outside their speaking regime.
