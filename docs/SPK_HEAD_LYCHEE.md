# The speaker-prediction head on Lychee-FD

This is mpmoshi-oracle's streaming speaker head (`docs/SPK_PREDICTION_ARCH.md` there),
moved onto Lychee-FD following that document's porting checklist. LoRA goes into
Lychee's audio encoder as well as its trunk. The code is in `lyspk/`.

## What was copied, what was rewritten

| | file | status |
|---|---|---|
| head | `lyspk/sortformer_head.py` | copied unchanged; only the import path changes |
| spec | `lyspk/head_spec.py` | copied; adds an `align` field and the `LYCHEE` presets |
| items + loss | `lyspk/items.py` | copied; chunk starts are floored to a multiple of `align` |
| labels | `lyspk/labels.py` | `build_labels` / `apply_shift` / `assert_arrival` unchanged; segment rasterising kept; crop rewritten to snap to 0.4 s windows |
| streaming | `lyspk/stream.py` | copied unchanged (`SelfFill`, `GoldCache`, `RotatedGold`, `EmptyCache`, `run_streams`) |
| readout | `lyspk/readout.py` | **rewritten**: one learned layer mix per stack (encoder and trunk) |
| host | `lyspk/host.py` | **new**: Lychee features at 12.5 Hz, LoRA, the preservation target |
| contrastive | `lyspk/contrast.py` | SupCon ported from `train_p1d.py._contrastive` (see below) |
| trainer / eval | `lyspk/train.py`, `lyspk/evaluate.py` | new, single GPU |
| data | `lyspk/data.py`, `lyspk/build_synth.py` | new: JSONL manifest of segments, an RTTM converter, and synthetic dialogues |

## Re-derived numbers (the md's "do not copy" table)

| | mpmoshi (Moshi) | Lychee | why |
|---|---|---|---|
| frame rate | 12.5 Hz (Mimi) | **12.5 Hz** | the adaptor gives 5 features per 0.4 s window, so every window length in frames carries over unchanged |
| `dim` | 4096 | encoder **1280**, trunk **3584** | two stacks, so two readout groups |
| layers read | first 16 of 32 | encoder input + 32 blocks; trunk input + layers 1..L (default L = 14 of 28) | L must stay below any future `<spk>` injection layer. Lychee's control branch reads trunk layer 20 and its speech branch reads layer 24 |
| `label_shift` | 2 (0.16 s) | **0** | each 0.4 s window is encoded on its own, so every feature in a window has already heard the whole window |
| `chunk` | 4 (0.32 s) | **5 (0.4 s), aligned to windows** | the head's lookahead inside a chunk (0–4 frames) is lookahead the encoder already has, so Lychee's own 0.4 s latency is unchanged |
| `cache_k` / `fifo` | 48 / 50 | 48 / 50 | same seconds (3.84 s / 4.0 s) at the same rate |
| `n_slots` | 3 | 3 | |
| `seq_len` | 198 | 199 | 3·48 + 50 + 5 |

Feature frame t comes from window t // 5. Within that window it is adaptor output t % 5,
it sits in suffix slot 2t of the trunk, and it is the mean of encoder frames 4(t % 5)
to 4(t % 5) + 3. Frames from windows before a change do not move when later audio
changes; `tests/test_spk.py::test_features_shapes_and_causality` checks this.

## LoRA

- **Encoder:** every Linear in all 32 Whisper blocks (query, key, value, out, and the two MLP layers). Rank 16.
- **Adaptor:** `linear1` and `linear2`, but only when the trunk is read. With L = 0 the adaptor feeds nothing the loss can see.
- **Trunk:** q, k, v, o, gate, up and down in layers 1..L. Rank 16.

The scaling is passed directly (2.0), the same convention mpmoshi uses. B starts at
zero, so step 0 computes exactly what the frozen model does; a test checks this. The
trunk is cut after layer L. The control, speech and merge branches and the LM head
are not loaded onto the GPU, because nothing trained here reaches them except through
trunk layer L.

## Loss

```
diar  = arrival-order BCE + 0.5 × PIL on the chunk         (unchanged)
con   = SupCon on 0.96 s single-speaker anchors, × 0.2
keep  = ||h_L − h_L(frozen)||² / ||h_L(frozen)||², × 1.0
```

- **diar** is the mpmoshi loss as is. The cache cells have weight exactly 0, and this is asserted.
- **con:** the md says GE2E, but the trainer it describes (`train_p1d_fix.py`) actually runs SupCon. That version is the one ported:
  - Positives are the same global speaker id in the other recordings of the same step.
  - The last 1024 anchors are kept as negatives. One change from mpmoshi: queue rows of the anchor's own speaker are left out rather than counted as negatives. With 20 voices the queue is mostly same-voice rows, and on a toy run counting them as negatives made the term rise.
  - The temperature is learned.
- **keep** stands in for mpmoshi's dialogue loss, because there is no Lychee dialogue training data here:
  - `h_L` is trunk layer L at every suffix slot, or the adaptor output when L = 0.
  - Everything else in Lychee reads only that tensor, so holding it to the frozen model's value keeps the conversation behaviour intact while the encoder learns speakers.
  - The frozen value comes from the same modules with LoRA switched off (`lora_disabled()`), which costs one extra forward pass with no gradient.
  - `--keep-weight 0` lets the LoRA drift freely.

## Data

The manifest has one line per recording: `{id, wav, duration, source, segments: [{speaker, start, end}]}`.
Speaker ids must be global, because the contrastive term takes its cross-recording
positives from them.

- `python -m lyspk.build_synth` makes dialogues from the 25 TTS voices already on the server:
  - 2, 3 or 4 speakers. The 4-speaker dialogues exercise the 4th-speaker mask.
  - Overlaps, plus 2–4 s silences.
  - 5 voices are held out for validation, so validation always uses voices the head has never seen.
- `python -m lyspk.data rttm ...` converts any RTTM corpus, such as AMI.

## Running

```bash
SMOKE=1 bash scripts/run_spk.sh          # pipeline check on the GPU
bash scripts/run_spk.sh                  # synth data, 3000 steps, eval
NAME=enc_only LLM_LAYERS=0 bash scripts/run_spk.sh
python -m lyspk.evaluate --ckpt work/spk/runs/main/ckpt_final.pt --manifest work/spk/synth/val.jsonl
```

Each evaluation reports, for each cache policy:
- `single_acc`
- `change_acc` (frames where the speaker id changes, using mpmoshi's definition)
- DER parts
- `rotation_follow`: RotatedGold minus GoldCache after relabelling. Above 0 means the head reads the cache contents, which is the md's test for identity versus inertia.

## Not done / known gaps

- **Not run on the real model yet.** Verified so far:
  - Every code path runs on a randomly initialised host built from Lychee's own module classes at toy sizes, on CPU.
  - The encoder loop matches `AudioEncoder.forward`.
  - Gradient checkpointing gives the same gradients as running without it.
  - LoRA starts at the identity.
  - Causality holds.
  - Train → checkpoint → evaluate → resume works.

  Loading the real checkpoint, memory use and speed are first checked by `SMOKE=1`.
- **No dialogue loss.** `keep` constrains the drift; it does not train the conversation.
- **Small training set.** The synthetic data has only 20 training voices. For real meetings, add an RTTM manifest such as AMI.
- **Thresholds are untuned.** `theta_act` is 0.30 and `theta_write` is 0.70. The md's sidecar calibration is not ported.
- **Single GPU only**, and there is no on-policy data.
- **No speaker injection back into Lychee.** `<spk>` injection is the next step, and L has to stay below wherever it goes.
