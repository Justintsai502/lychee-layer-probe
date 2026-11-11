#!/usr/bin/env bash
# Speaker-id accuracy on the MP-Bench 100-clip subset (lyspk/mpbench_spk.py), one checkpoint. GPU,
# a few minutes. Needs the eval kit (scp it to work/mpbench_mp100_eval_kit) and the MP-Bench audio.
#
#   bash scripts/run_spk_mp100.sh                                  # runs/mt at step 1200
#   CKPT=work/spk/runs/mt/ckpt_final.pt bash scripts/run_spk_mp100.sh
#   LIMIT=2 bash scripts/run_spk_mp100.sh                          # smoke
#   AUDIO=/path/to/MP_Bench KIT=/path/to/kit bash scripts/run_spk_mp100.sh
#
# Log: work/logs/mp100_spk_<name>_<ckpt stem>.log. Full output: <run>/mp100_<stem>/; the small files
# (summary, per-task, the kit's score text, probs) are copied to results/spk/<name>/mp100_<stem>/.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
source scripts/spk_mt_env.sh
NAME=${NAME:-mt}
RUN=work/spk/runs/$NAME
KIT=${KIT:-work/mpbench_mp100_eval_kit}
AUDIO=${AUDIO:-/work/kcire3twcc/datasets/MP_Bench_data}
CKPT=${CKPT:-}
if [[ -z "$CKPT" ]]; then
  for c in "$RUN/ckpt_step001200.pt" "$RUN/ckpt_final.pt"; do [[ -f "$c" ]] && { CKPT=$c; break; }; done
fi
[[ -n "$CKPT" && -f "$CKPT" ]] || { echo "no checkpoint (looked for $RUN/ckpt_step001200.pt, ckpt_final.pt); set CKPT="; exit 1; }
STEM=$(basename "$CKPT" .pt)
LOG=work/logs/mp100_spk_${NAME}_${STEM}.log
exec > >(tee -a "$LOG") 2>&1
echo "======== $(date '+%F %T') $0 ckpt=$CKPT host=$(hostname) log=$LOG"
[[ -f "$KIT/subset/instances.tsv" && -d "$KIT/gold_speakers" ]] || { echo "kit not at $KIT (set KIT=)"; exit 1; }
[[ -d "$AUDIO" ]] || { echo "MP-Bench audio root $AUDIO is not a directory (set AUDIO=)"; exit 1; }
OUTD=$RUN/mp100_$STEM
$PY -u -m lyspk.mpbench_spk --ckpt "$CKPT" --kit "$KIT" --audio-root "$AUDIO" --out "$OUTD" --limit "${LIMIT:-0}" ${EXTRA:-}
RES=results/spk/$NAME/mp100_$STEM
mkdir -p "$RES"
cp "$OUTD"/summary.json "$OUTD"/per_task.json "$OUTD"/score_*.txt "$OUTD"/probs.npz "$RES"/
echo "copied to $RES"
