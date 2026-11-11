#!/usr/bin/env bash
# Speaker head + Lychee's own dialogue loss on mpmoshi's multi-track corpora (AMI / IF / IF-self-distill).
# docs/SPK_HEAD_LYCHEE.md ("Dialogue loss"). Run on a GPU node from this repo:
#
#   SMOKE=1 bash scripts/run_spk_mt.sh     # 12 recordings per pool, 20 steps: paths, tokenizer, memory, speed
#   bash scripts/run_spk_mt.sh             # build manifests + speech tokens once, train, evaluate
#   NAME=mt_L8 LLM_LAYERS=8 bash scripts/run_spk_mt.sh
#   PUSH=1 ...                             # copy the run's small JSON files to results/spk/<name>/ and push
#
# Needs the mpmoshi checkout (MP, default /work/b12901015/mpmoshi-train) for its manifests and the vendored
# AMI annotations, and the wav roots AMI_ROOT / IF_ROOT / IF_SD4_ROOT (read from $MP/scripts/stage1.env
# when that file exists, as mpmoshi's own scripts do).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p work/logs
exec > >(tee -a work/logs/run_spk_mt.log) 2>&1

PY=${PY:-python}
MP=${MP:-/work/b12901015/mpmoshi-train}
[[ -f "$MP/scripts/stage1.env" ]] && { set -a; source "$MP/scripts/stage1.env"; set +a; }
: "${AMI_ROOT:?set AMI_ROOT (or MP with scripts/stage1.env)}"
: "${IF_ROOT:?set IF_ROOT}"
: "${IF_SD4_ROOT:?set IF_SD4_ROOT}"
NAME=${NAME:-mt}
LLM_LAYERS=${LLM_LAYERS:-14}
STEPS=${STEPS:-3000}
MIX=${MIX:-"0.4 0.4 0.2"}            # if sd ami, draw shares (mpmoshi: tt=40,sd=40,ami=20 by audio seconds)
OUT=work/spk/mt
TOKDIR=models/step_audio2_tok
TOK=$TOKDIR/token2wav/speech_tokenizer_v2_25hz.onnx
M=$MP/data/manifests
train=(--eval-every 250 --save-every 500)
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT=work/spk/mt_smoke; NAME=smoke_mt; STEPS=20
  train=(--eval-every 10 --save-every 0 --eval-clips 4 --log-every 1)
fi
mkdir -p "$OUT"

[[ -f third_party/lychee_fd_model/LycheeFD.py ]] || bash scripts/fetch_lychee_code.sh
echo "python: $(command -v $PY) | env: ${CONDA_DEFAULT_ENV:-none} | MP=$MP"

if [[ ! -f "$TOK" ]]; then
  echo "== 0/4 speech tokenizer -> $TOK"
  $PY -c "from huggingface_hub import hf_hub_download as d; print(d('stepfun-ai/Step-Audio-2-mini', 'token2wav/speech_tokenizer_v2_25hz.onnx', local_dir='$TOKDIR'))"
fi

echo "== 1/4 multi-track manifests -> $OUT"
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

echo "== 2/4 agent turns + speech tokens (skips rows already done)"
for n in if_train sd_train ami_train if_test ami_eval; do
  $PY -u -m lyspk.stoken --manifest "$OUT/$n.jsonl" --out "$OUT/$n.turns.jsonl" --tokenizer "$TOK" \
    --turns-dir "$OUT/turns/$n"
done
# validation: IF test and AMI eval interleaved, so the first --eval-clips windows cover both
$PY - "$OUT/if_test.turns.jsonl" "$OUT/ami_eval.turns.jsonl" "$OUT/val.turns.jsonl" <<'PYEOF'
import itertools, sys
a, b = (open(p).read().splitlines() for p in sys.argv[1:3])
rows = [x for pair in itertools.zip_longest(a, b) for x in pair if x]
open(sys.argv[3], "w").write("\n".join(rows) + "\n")
print(f"wrote {sys.argv[3]}: {len(rows)} rows")
PYEOF

echo "== 3/4 train -> work/spk/runs/$NAME (trunk layers $LLM_LAYERS, $STEPS steps, mix if/sd/ami $MIX)"
resume=()
[[ "${SMOKE:-0}" != "1" && -f "work/spk/runs/$NAME/ckpt_latest.pt" && "${FRESH:-0}" != "1" ]] && resume=(--resume "work/spk/runs/$NAME/ckpt_latest.pt")
$PY -u -m lyspk.train --dialogue --train "$OUT/if_train.turns.jsonl" "$OUT/sd_train.turns.jsonl" "$OUT/ami_train.turns.jsonl" \
  --train-weights $MIX --val "$OUT/val.turns.jsonl" --name "$NAME" --llm-layers "$LLM_LAYERS" --steps "$STEPS" \
  "${train[@]}" ${resume[@]+"${resume[@]}"} ${EXTRA:-}

echo "== 4/4 evaluate on all of $OUT/val.turns.jsonl"
$PY -u -m lyspk.evaluate --ckpt "work/spk/runs/$NAME/ckpt_final.pt" --manifest "$OUT/val.turns.jsonl"

if [[ "${PUSH:-0}" == "1" && "${SMOKE:-0}" != "1" ]]; then
  mkdir -p "results/spk/$NAME"
  cp work/spk/runs/$NAME/{train.jsonl,eval.jsonl,run_config.json,eval_ckpt_final.json} "results/spk/$NAME/"
  git add "results/spk/$NAME"
  git commit -m "Speaker head + dialogue loss results: $NAME"
  git push
  echo "pushed results/spk/$NAME"
fi
