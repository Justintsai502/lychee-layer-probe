#!/usr/bin/env bash
# Data prep for run_spk_mt.sh, CPU only (no GPU is touched: CUDA_VISIBLE_DEVICES is emptied).
# Run it anywhere with the repo and the datasets, e.g. a CPU node or the login node, before taking a GPU:
#
#   bash scripts/prep_spk_mt.sh            # -> work/spk/mt/{pool}.jsonl, {pool}.turns.jsonl, val.turns.jsonl
#   SMOKE=1 bash scripts/prep_spk_mt.sh    # 12 recordings per pool -> work/spk/mt_smoke/
#   JOBS=3 ...                             # pools tokenized at once (default 5); threads are split between them
#
# 0. Step-Audio-2's speech tokenizer (download once)
# 1. lyspk.mt_data: multi-track manifests from mpmoshi's AMI / IF / SD manifests, test sessions held out
# 2. lyspk.stoken --device cpu: every participant's turns -> speech tokens, the five pools in parallel;
#    one JSON per recording, so a rerun continues where it stopped
# Log: work/logs/prep_spk_mt[_smoke].log, plus work/logs/prep_stoken_<pool>[_smoke].log per pool.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
SFX=""; [[ "${SMOKE:-0}" == "1" ]] && SFX=_smoke
exec > >(tee -a "work/logs/prep_spk_mt$SFX.log") 2>&1
echo "======== $(date '+%F %T') $0 SMOKE=${SMOKE:-0} host=$(hostname)"
source scripts/spk_mt_env.sh
export CUDA_VISIBLE_DEVICES=""
for v in AMI_ROOT IF_ROOT IF_SD4_ROOT; do [[ -d "${!v}" ]] || { echo "$v=${!v} is not a directory"; exit 1; }; done
echo "AMI_ROOT=$AMI_ROOT IF_ROOT=$IF_ROOT IF_SD4_ROOT=$IF_SD4_ROOT | python: $(command -v $PY) | env: ${CONDA_DEFAULT_ENV:-none}"
mkdir -p "$OUT"
M=$MP/data/manifests

if [[ ! -f "$TOK" ]]; then
  echo "== 0/2 speech tokenizer -> $TOK"
  $PY -c "from huggingface_hub import hf_hub_download as d; print(d('stepfun-ai/Step-Audio-2-mini', 'token2wav/speech_tokenizer_v2_25hz.onnx', local_dir='$TOKDIR'))"
fi

echo "== 1/2 multi-track manifests -> $OUT"
mk() {  # name, then lyspk.mt_data args
  local name=$1; shift
  [[ -f "$OUT/$name.jsonl" ]] && { echo "skip $name"; return; }
  $PY -u -m lyspk.mt_data "$@" --out "$OUT/$name.full.jsonl"
  if [[ "${SMOKE:-0}" == "1" ]]; then head -n 12 "$OUT/$name.full.jsonl" > "$OUT/$name.jsonl"; else mv "$OUT/$name.full.jsonl" "$OUT/$name.jsonl"; fi
}
ANN=$MP/data/ami/annotations
mk ami_train ami --manifest "$M/ami_train_ads.jsonl" --annotations "$ANN" --exclude "$M/ami_eval.jsonl"
mk ami_eval  ami --manifest "$M/ami_eval.jsonl" --annotations "$ANN"
mk if_train  scenes --manifest "$M/if_4spk_train.jsonl" --source if --root "$IF_ROOT" --exclude "$M/if_4spk_test.jsonl"
mk if_test   scenes --manifest "$M/if_4spk_test.jsonl" --source if --root "$IF_ROOT"
mk sd_train  scenes --manifest "$M/sd4spk_if_train.jsonl" --source sd --root "$IF_SD4_ROOT" --exclude "$M/if_4spk_test.jsonl"

JOBS=${JOBS:-5}
CORES=$(nproc)
export OMP_NUM_THREADS=$(( CORES / JOBS > 0 ? CORES / JOBS : 1 )) MKL_NUM_THREADS=$(( CORES / JOBS > 0 ? CORES / JOBS : 1 ))
echo "== 2/2 speech tokens on CPU: ${#POOLS[@]} pools, $JOBS at a time, $OMP_NUM_THREADS threads each ($CORES cores)"
pids=(); names=()
for n in "${POOLS[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 5; done
  $PY -u -m lyspk.stoken --manifest "$OUT/$n.jsonl" --out "$OUT/$n.turns.jsonl" --tokenizer "$TOK" \
    --turns-dir "$OUT/turns/$n" --device cpu > "work/logs/prep_stoken_$n$SFX.log" 2>&1 &
  pids+=($!); names+=("$n")
  echo "started $n (pid $!, log work/logs/prep_stoken_$n$SFX.log)"
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "done ${names[$i]}: $(tail -1 "work/logs/prep_stoken_${names[$i]}$SFX.log")"
  else echo "FAILED ${names[$i]}: see work/logs/prep_stoken_${names[$i]}$SFX.log"; fail=1; fi
done
(( fail == 0 )) || exit 1

# validation: IF test and AMI eval interleaved, so the first --eval-clips windows cover both
$PY - "$OUT/if_test.turns.jsonl" "$OUT/ami_eval.turns.jsonl" "$OUT/val.turns.jsonl" <<'PYEOF'
import itertools, sys
a, b = (open(p).read().splitlines() for p in sys.argv[1:3])
rows = [x for pair in itertools.zip_longest(a, b) for x in pair if x]
open(sys.argv[3], "w").write("\n".join(rows) + "\n")
print(f"wrote {sys.argv[3]}: {len(rows)} rows")
PYEOF
echo "prep done: $OUT (train with: ${SMOKE:+SMOKE=1 }bash scripts/run_spk_mt.sh)"
