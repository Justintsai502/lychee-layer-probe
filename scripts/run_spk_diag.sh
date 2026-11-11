#!/usr/bin/env bash
# mpmoshi's speaker diagnostic battery on one checkpoint (lyspk/diag.py; rules in mpmoshi-oracle
# docs/SPK_SCORING_RULES.md). GPU, a few minutes at the default size. Needs the CPU prep's
# held-out turns manifests (scripts/prep_spk_mt.sh) and a finished run.
#
#   bash scripts/run_spk_diag.sh                            # runs/mt at step 1200, 8 recordings/source x 2 windows
#   CKPT=work/spk/runs/mt/ckpt_step000600.pt bash scripts/run_spk_diag.sh
#   SESSIONS=16 WPS=3 bash scripts/run_spk_diag.sh          # more sample
#   LIMIT=2 BOOT=200 bash scripts/run_spk_diag.sh           # smoke
#
# Log: work/logs/diag_spk_<name>_<ckpt stem>.log. Results: <run>/diag_<ckpt stem>/summary.json etc.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
source scripts/spk_mt_env.sh
NAME=${NAME:-mt}
RUN=work/spk/runs/$NAME
CKPT=${CKPT:-}
if [[ -z "$CKPT" ]]; then
  for c in "$RUN/ckpt_step001200.pt" "$RUN/ckpt_final.pt"; do [[ -f "$c" ]] && { CKPT=$c; break; }; done
fi
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "no checkpoint (looked for $RUN/ckpt_step001200.pt, ckpt_final.pt); set CKPT="; exit 1; }
STEM=$(basename "$CKPT" .pt)
LOG=work/logs/diag_spk_${NAME}_${STEM}.log
exec > >(tee -a "$LOG") 2>&1
echo "======== $(date '+%F %T') $0 ckpt=$CKPT host=$(hostname) log=$LOG"
for f in "$OUT/ami_eval.turns.jsonl" "$OUT/if_test.turns.jsonl"; do
  [[ -f "$f" ]] || { echo "missing $f -- run the CPU prep first: bash scripts/prep_spk_mt.sh"; exit 1; }
done
$PY -u -m lyspk.diag --ckpt "$CKPT" --out "${DIAG:-$RUN/diag_$STEM}" \
  --manifest ami="$OUT/ami_eval.turns.jsonl" --manifest tt="$OUT/if_test.turns.jsonl" \
  --sessions "${SESSIONS:-8}" --windows-per-session "${WPS:-2}" --window-sec "${WINDOW_SEC:-60}" \
  --seed "${SEED:-0}" --boot "${BOOT:-10000}" --limit "${LIMIT:-0}" ${EXTRA:-}
