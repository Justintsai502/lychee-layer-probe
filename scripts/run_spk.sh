#!/usr/bin/env bash
# Speaker head on Lychee-FD (lyspk/, docs/SPK_HEAD_LYCHEE.md). Run on a GPU node from this repo:
#
#   SMOKE=1 bash scripts/run_spk.sh        # 50 small dialogues, 20 steps at full size: loading, memory, speed
#   bash scripts/run_spk.sh                # build synthetic data once, train 3000 steps, evaluate
#   NAME=enc_only LLM_LAYERS=0 bash scripts/run_spk.sh          # encoder + LoRA only, no trunk
#   NAME=nokeep EXTRA="--keep-weight 0" bash scripts/run_spk.sh  # let the LoRA drift freely
#   PUSH=1 ...                             # copy the run's small JSON files to results/spk/<name>/ and push
#
# Needs: models/lychee_fd (as for run_server.sh) and ../bayling-speaker-probe/work/tts (the 25 voices).
# Other corpora: make a manifest with `python -m lyspk.data rttm ...` and pass it with
#   EXTRA="--train work/spk/synth/train.jsonl work/spk/ami_train.jsonl"  (the last --train wins, so list all)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
exec > >(tee -a work/logs/run_spk.log) 2>&1

PY=${PY:-python}
NAME=${NAME:-main}
LLM_LAYERS=${LLM_LAYERS:-14}
STEPS=${STEPS:-3000}
SYN=work/spk/synth
build=(--out "$SYN")
train=(--eval-every 250 --save-every 500)
if [[ "${SMOKE:-0}" == "1" ]]; then
  SYN=work/spk/synth_smoke; NAME=smoke; STEPS=20
  build=(--out "$SYN" --n-train 50 --n-val 5)
  train=(--eval-every 10 --save-every 0 --eval-clips 4 --log-every 1)
fi

[[ -f third_party/lychee_fd_model/LycheeFD.py ]] || bash scripts/fetch_lychee_code.sh
echo "python: $(command -v $PY) | env: ${CONDA_DEFAULT_ENV:-none}"

if [[ ! -f "$SYN/train.jsonl" ]]; then
  echo "== 1/3 synthetic dialogues -> $SYN"
  $PY -u -m lyspk.build_synth "${build[@]}"
else
  echo "== 1/3 skip: $SYN/train.jsonl exists"
fi

echo "== 2/3 train -> work/spk/runs/$NAME (trunk layers $LLM_LAYERS, $STEPS steps)"
resume=()
[[ "${SMOKE:-0}" != "1" && -f "work/spk/runs/$NAME/ckpt_latest.pt" && "${FRESH:-0}" != "1" ]] && resume=(--resume "work/spk/runs/$NAME/ckpt_latest.pt")
$PY -u -m lyspk.train --train "$SYN/train.jsonl" --val "$SYN/val.jsonl" --name "$NAME" \
  --llm-layers "$LLM_LAYERS" --steps "$STEPS" "${train[@]}" ${resume[@]+"${resume[@]}"} ${EXTRA:-}

echo "== 3/3 evaluate on all of $SYN/val.jsonl (unseen voices)"
$PY -u -m lyspk.evaluate --ckpt "work/spk/runs/$NAME/ckpt_final.pt" --manifest "$SYN/val.jsonl"

if [[ "${PUSH:-0}" == "1" && "${SMOKE:-0}" != "1" ]]; then
  mkdir -p "results/spk/$NAME"
  cp work/spk/runs/$NAME/{train.jsonl,eval.jsonl,run_config.json,eval_ckpt_final.json} "results/spk/$NAME/"
  git add "results/spk/$NAME"
  git commit -m "Speaker head results: $NAME"
  git push
  echo "pushed results/spk/$NAME"
fi
