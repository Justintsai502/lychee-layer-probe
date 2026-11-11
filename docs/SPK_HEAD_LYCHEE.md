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
- **keep** stands in for the dialogue loss when there are only single-wav recordings (the synthetic set). With the multi-track corpora, use `--dialogue` (next section) instead:
  - `h_L` is trunk layer L at every suffix slot, or the adaptor output when L = 0.
  - Everything else in Lychee reads only that tensor, so holding it to the frozen model's value keeps the conversation behaviour intact while the encoder learns speakers.
  - The frozen value comes from the same modules with LoRA switched off (`lora_disabled()`), which costs one extra forward pass with no gradient.
  - `--keep-weight 0` lets the LoRA drift freely.

## Dialogue loss (`--dialogue`, multi-track AMI / IF / SD)

mpmoshi's dialogue loss was Moshi's own LM loss on the same meetings: one participant is
cast as the agent. Here it is Lychee's own loss, built the way Lychee's training code
builds it (`Training_package/Code/DataLoaders/LycheeFDDataset.py` at the pinned commit,
token ids from the released `config.json`). `lyspk/dialog.py` has the details.

| Lychee target | from |
|---|---|
| text | the agent's utterance text for the turn, laid at the turn start: `ids + <\|EOT\|>`, then `<tts_pad>`. No word timing needed |
| stoken | the agent's own track for that turn through Step-Audio-2-mini's `speech_tokenizer_v2_25hz.onnx` (25 Hz), via `s3tokenizer` as Step-Audio-2's `token2wav.py` calls it. Laid as `<stoken_delay> x 10, <tts_start>, tokens, <tts_end>`. `stoken_delay_num` is 10 in the released config (the example script says 0) |
| control | per 0.4 s chunk: `keep_listening` / `start_speaking` / `keep_speaking` / `start_listening`, thinned with Lychee's own `mask_control_label` (all starts, the ends of every keep run, a random 20 % of the rest, at most 3 x the specials) |

- **Per crop:**
  - One agent is drawn from the recording's participants.
  - The audio input is the other tracks summed. The agent takes no head slot (as in mpmoshi).
  - The agent's turns are teacher-forced into the text, stoken and control channels.
- **One deliberate difference from Lychee's data:** while the agent speaks, Lychee's data feeds silence as user audio. Here the other participants' real audio stays in, because in a meeting they keep talking.
- **Turns:**
  - The agent's utterances with pauses < 1.0 s between them are joined, up to 20 s per turn.
  - A turn starts 0.44 s (delay + `<tts_start>`) before its real speech, so its speech tokens land on time.
  - At least one listening chunk follows each turn (where `start_speaking` goes). A turn that would start sooner is pushed later.
  - A crop never starts inside a turn.
- **Loss:** text CE + stoken CE + control CE, as `LycheeFD.forward` sums them (shifted, clamped vocab ranges). The LM head is applied only where a label exists.
  - `tests/test_mt.py` checks that the three numbers equal `LycheeFD.forward`'s on a randomly initialised LycheeFD built from the upstream class.
- **What is trained, and the cost:**
  - LoRA stays on the encoder, adaptor and trunk 1..L. The rest of Lychee is loaded frozen and gradients pass through it: the full trunk, the stoken/merge/control branches and the LM head.
  - The three branch embedding tables are never used and are dropped: about 22 GB of bf16 weights instead of 25.6 GB.
  - `--keep-weight` defaults to 0 with `--dialogue`.
- **Eval on the held-out set:**
  - The same teacher-forced sequence, reporting text / stoken / control CE and control accuracy, both over all chunk ends and over the `start_*` chunks alone.
  - At step 0 (LoRA = identity) these are the frozen Lychee's numbers on this data, which is the baseline the LoRA should not fall below.

Pipeline: `scripts/run_spk_mt.sh`.
1. `lyspk.mt_data` reads mpmoshi's manifests and the vendored AMI annotations. It holds out every meeting or scene in `ami_eval` / `if_4spk_test`, which includes the 131 SD scenes shared with the IF test set.
2. `lyspk.stoken` tokenizes every participant's turns once.
3. `lyspk.train --dialogue` samples if / sd / ami at 0.4 / 0.4 / 0.2 of draws (mpmoshi's 40/40/20 was by audio seconds).

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
- **Dialogue loss: written and tested on toy models only.** `--dialogue` loads all of Lychee: memory and step time are first measured by `SMOKE=1 bash scripts/run_spk_mt.sh`.
- **AMI crosstalk.** The other participants' headsets pick up the agent faintly, so on AMI the input still contains a little of the agent's own voice (the mpmoshi doc notes the same crosstalk).
- **Single-wav data** (the synthetic set) has no agent: the `keep` term is the only constraint there.
- **Small training set.** The synthetic data has only 20 training voices. For real meetings, add an RTTM manifest such as AMI.
- **Thresholds are untuned.** `theta_act` is 0.30 and `theta_write` is 0.70. The md's sidecar calibration is not ported.
- **Single GPU only**, and there is no on-policy data.
- **No speaker injection back into Lychee.** `<spk>` injection is the next step, and L has to stay below wherever it goes.
