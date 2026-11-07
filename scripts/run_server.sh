#!/usr/bin/env bash
# Full pipeline on the GPU server. No arguments needed when this repo sits next to
# bayling-speaker-probe (e.g. /work/b12901015/{bayling-speaker-probe,lychee-layer-probe}):
#
#   bash scripts/run_server.sh            # extract + analyze, writes results/*.json
#   SMOKE=1 bash scripts/run_server.sh    # first 6 clips only
#   PUSH=1 bash scripts/run_server.sh     # also commit and push results/*.json
#
# Reads (never writes) ../bayling-speaker-probe/work/{tts/manifest.kept.jsonl,
# features/baselines.npz, features/bayling_silence.npz}.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
exec > >(tee -a work/logs/run_server.log) 2>&1

PY=${PY:-python}
SIB=../bayling-speaker-probe/work

limit=()
[[ "${SMOKE:-0}" == "1" ]] && limit=(--limit 6)

[[ -f third_party/lychee_fd_model/LycheeFD.py ]] || bash scripts/fetch_lychee_code.sh

echo "== 1/3 Lychee hidden states"
$PY -u lyprobe/extract_lychee.py ${limit[@]+"${limit[@]}"}

echo "== 2/3 Analysis: Lychee  -> results/lychee.json (smoke: work/results/smoke.json)"
json_args=()
[[ "${SMOKE:-0}" == "1" ]] && json_args=(--json work/results/smoke.json)
$PY -u lyprobe/analyze.py --features work/features/lychee.npz ${json_args[@]+"${json_args[@]}"}

if [[ -f "$SIB/features/bayling_silence.npz" && "${SMOKE:-0}" != "1" ]]; then
  echo "== 3/3 Analysis: BayLing with the same metric  -> results/bayling.json"
  $PY -u lyprobe/analyze.py --features "$SIB/features/bayling_silence.npz"
fi

if [[ "${PUSH:-0}" == "1" && "${SMOKE:-0}" != "1" ]]; then
  git add results/*.json
  git commit -m "Results: $(ls results/*.json | xargs -n1 basename | tr '\n' ' ')"
  git push
  echo "pushed results/*.json"
else
  echo "done. To share: git add results/*.json && git commit -m results && git push"
fi
