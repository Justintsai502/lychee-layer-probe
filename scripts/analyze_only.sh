#!/usr/bin/env bash
# Re-run only the analysis on features that already exist (CPU, a few minutes, no model):
#   bash scripts/analyze_only.sh            # rewrites results/lychee.json, results/bayling.json
#   PUSH=1 bash scripts/analyze_only.sh     # and commits + pushes them
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
exec > >(tee -a work/logs/analyze_only.log) 2>&1

PY=${PY:-python}
SIB=../bayling-speaker-probe/work

[[ -f work/features/lychee.npz ]] || { echo "work/features/lychee.npz missing: run scripts/run_server.sh first"; exit 1; }
$PY -u lyprobe/analyze.py --features work/features/lychee.npz
[[ -f "$SIB/features/bayling_silence.npz" ]] && $PY -u lyprobe/analyze.py --features "$SIB/features/bayling_silence.npz"

if [[ "${PUSH:-0}" == "1" ]]; then
  git add results/*.json
  git commit -m "Results (re-analysis): $(ls results/*.json | xargs -n1 basename | tr '\n' ' ')"
  git push
  echo "pushed results/*.json"
fi
