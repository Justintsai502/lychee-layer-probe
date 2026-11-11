#!/usr/bin/env bash
# Speaker head + Lychee's own dialogue loss on mpmoshi's multi-track corpora (AMI / IF / IF-self-distill).
# docs/SPK_HEAD_LYCHEE.md ("Dialogue loss"). Data first, on CPU (scripts/prep_spk_mt.sh), then on a GPU node:
#
#   SMOKE=1 bash scripts/prep_spk_mt.sh && SMOKE=1 bash scripts/run_spk_mt.sh   # 12 recordings/pool, 20 steps
#   bash scripts/prep_spk_mt.sh            # CPU: manifests + speech tokens -> work/spk/mt/
#   bash scripts/run_spk_mt.sh             # GPU: train, evaluate
#   NAME=mt_L8 LLM_LAYERS=8 bash scripts/run_spk_mt.sh
#   PUSH=1 ...                             # copy the run's small JSON files to results/spk/<name>/ and push
#   EVERY=100 KEEP_LAST=10 ...             # eval + checkpoint every 100 steps (default 50), keep the newest 10
#                                          # ckpt_step*.pt (default all); ckpt_latest.pt always holds the resume state
#
# Needs the mpmoshi checkout (MP, default /work/b12901015/mpmoshi-train) for its manifests and the vendored
# AMI annotations, and the wav roots AMI_ROOT / IF_ROOT / IF_SD4_ROOT (read from $MP/scripts/stage1.env
# when that file exists, as mpmoshi's own scripts do).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
# Everything (stdout + stderr, every python step) goes to the screen AND to one log per run name,
# appended across restarts: work/logs/run_spk_mt_<NAME>.log (smoke: run_spk_mt_smoke_mt.log).
if [[ "${SMOKE:-0}" == "1" ]]; then LOG=work/logs/run_spk_mt_smoke_mt.log; else LOG=work/logs/run_spk_mt_${NAME:-mt}.log; fi
exec > >(tee -a "$LOG") 2>&1
echo "======== $(date '+%F %T') $0 SMOKE=${SMOKE:-0} NAME=${NAME:-mt} host=$(hostname) log=$LOG"

source scripts/spk_mt_env.sh
NAME=${NAME:-mt}
LLM_LAYERS=${LLM_LAYERS:-14}
STEPS=${STEPS:-1200}
MIX=${MIX:-"0.4 0.4 0.2"}            # if sd ami, draw shares (mpmoshi: tt=40,sd=40,ami=20 by audio seconds)
ACCUM=${ACCUM:-8}                    # crops per optimizer step, one crop per forward (mpmoshi: 1 GPU x accum 8)
EVERY=${EVERY:-50}                   # validation and checkpoint every EVERY steps
train=(--eval-every "$EVERY" --save-every "$EVERY" --keep-last "${KEEP_LAST:-0}")
# taken from mpmoshi's stage-1 run (train_p1d_fix + stage1.env), where they carry over:
#   AMI crops 60-120 s (IF / SD scenes whole), warmup ~4 % of the run, trunk LoRA rank 32 below the
#   readout at 1e-4 and rank 8 above it at 2e-5, head lr 3e-4, wd 0.01, clip 1.0, the id4 head
#   (chair + region ids instead of pos/seg embeddings: HEAD_MODE=chunk4_s2_id4 there)
mp=(--crop-min 60 --crop-max 120 --warmup 50 --llm-lora-rank 32 --llm-high-lora-rank 8
    --lr-lora 1e-4 --lr-lora-high 2e-5 --lr-head 3e-4 --mode lychee_c5_id4)
if [[ "${SMOKE:-0}" == "1" ]]; then
  NAME=smoke_mt; STEPS=20; ACCUM=2     # memory is per crop, so accum 2 suffices
  train=(--eval-every 10 --save-every 0 --eval-clips 4 --log-every 1)
fi
[[ -f third_party/lychee_fd_model/LycheeFD.py ]] || bash scripts/fetch_lychee_code.sh
echo "python: $(command -v $PY) | env: ${CONDA_DEFAULT_ENV:-none} | data $OUT"
# Data prep is CPU work and lives in prep_spk_mt.sh; run it before taking the GPU.
missing=()
for f in "${POOLS[@]/%/.turns.jsonl}" val.turns.jsonl; do [[ -f "$OUT/$f" ]] || missing+=("$OUT/$f"); done
if (( ${#missing[@]} )); then
  echo "missing ${missing[*]}"
  echo "run the CPU prep first: ${SMOKE:+SMOKE=1 }bash scripts/prep_spk_mt.sh"
  exit 1
fi

echo "== 1/2 train -> work/spk/runs/$NAME (trunk layers $LLM_LAYERS, $STEPS steps x $ACCUM crops, mix if/sd/ami $MIX)"
resume=()
[[ "${SMOKE:-0}" != "1" && -f "work/spk/runs/$NAME/ckpt_latest.pt" && "${FRESH:-0}" != "1" ]] && resume=(--resume "work/spk/runs/$NAME/ckpt_latest.pt")
$PY -u -m lyspk.train --dialogue --train "$OUT/if_train.turns.jsonl" "$OUT/sd_train.turns.jsonl" "$OUT/ami_train.turns.jsonl" \
  --train-weights $MIX --val "$OUT/val.turns.jsonl" --name "$NAME" --llm-layers "$LLM_LAYERS" --steps "$STEPS" --accum "$ACCUM" \
  "${mp[@]}" "${train[@]}" ${resume[@]+"${resume[@]}"} ${EXTRA:-}

echo "== 2/2 evaluate on all of $OUT/val.turns.jsonl"
$PY -u -m lyspk.evaluate --ckpt "work/spk/runs/$NAME/ckpt_final.pt" --manifest "$OUT/val.turns.jsonl"

if [[ "${PUSH:-0}" == "1" && "${SMOKE:-0}" != "1" ]]; then
  mkdir -p "results/spk/$NAME"
  cp work/spk/runs/$NAME/{train.jsonl,eval.jsonl,run_config.json,eval_ckpt_final.json} "results/spk/$NAME/"
  git add "results/spk/$NAME"
  git commit -m "Speaker head + dialogue loss results: $NAME"
  git push
  echo "pushed results/spk/$NAME"
fi
