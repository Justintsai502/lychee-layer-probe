#!/usr/bin/env bash
# EER version, encoder layers included. Run on a GPU node from this repo:
#
#   bash scripts/run_eer.sh           # extract both encoders, then EER for Lychee + BayLing
#   SMOKE=1 bash scripts/run_eer.sh   # first 6 clips only (pipeline check, nothing pushed)
#   PUSH=1 bash scripts/run_eer.sh    # also commit + push results/*_eer.json
#
# Needs the LLM features from the earlier runs (not recomputed):
#   work/features/lychee.npz                                     (scripts/run_server.sh)
#   ../bayling-speaker-probe/work/features/bayling_silence.npz   (bayling-speaker-probe)
# and BayLing's speech tokenizer at ../bayling-speaker-probe/models/speech_tokenizer.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
exec > >(tee -a work/logs/run_eer.log) 2>&1

PY=${PY:-python}
SIB=../bayling-speaker-probe/work
SMOKE=${SMOKE:-0}

limit=() ; suffix=""
if [[ "$SMOKE" == "1" ]]; then limit=(--limit 6); suffix="_smoke"; fi
LY_ENC=work/features/lychee_encoder${suffix}.npz
BL_ENC=work/features/bayling_encoder${suffix}.npz

[[ -f third_party/lychee_fd_model/LycheeFD.py ]] || bash scripts/fetch_lychee_code.sh
[[ -f work/features/lychee.npz ]] || { echo "work/features/lychee.npz missing: run scripts/run_server.sh first"; exit 1; }

echo "== 1/4 Lychee audio encoder (GPU)"
$PY -u lyprobe/extract_lychee_encoder.py ${limit[@]+"${limit[@]}"} --out "$LY_ENC"

echo "== 2/4 BayLing speech tokenizer encoder (GPU, small)"
$PY -u lyprobe/extract_bayling_encoder.py ${limit[@]+"${limit[@]}"} --out "$BL_ENC"

json_ly=() ; json_bl=()
if [[ "$SMOKE" == "1" ]]; then
  json_ly=(--json work/results/lychee_eer_smoke.json); json_bl=(--json work/results/bayling_eer_smoke.json)
fi
echo "== 3/4 EER: Lychee (encoder + LLM)"
$PY -u lyprobe/analyze_eer.py --llm work/features/lychee.npz --encoder "$LY_ENC" ${json_ly[@]+"${json_ly[@]}"}

echo "== 4/4 EER: BayLing (encoder + LLM)"
$PY -u lyprobe/analyze_eer.py --llm "$SIB/features/bayling_silence.npz" --encoder "$BL_ENC" ${json_bl[@]+"${json_bl[@]}"}

if [[ "${PUSH:-0}" == "1" && "$SMOKE" != "1" ]]; then
  git add results/*_eer.json
  git commit -m "Results (EER, encoder + LLM): lychee_eer.json bayling_eer.json"
  git push
  echo "pushed results/*_eer.json"
else
  echo "done. To share: git add results/*_eer.json && git commit -m eer && git push"
fi
