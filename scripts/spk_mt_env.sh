# Sourced by prep_spk_mt.sh and run_spk_mt.sh: paths shared by data prep (CPU) and training (GPU).
PY=${PY:-python}
MP=${MP:-/work/b12901015/mpmoshi-train}
# Only the three audio roots come from mpmoshi's stage1.env. Sourcing all of it would also set PY to
# mpmoshi's moshi env (and HF_HOME etc.), which is not the env this repo runs in. Values set here win.
if [[ -f "$MP/scripts/stage1.env" ]]; then
  for v in AMI_ROOT IF_ROOT IF_SD4_ROOT; do
    if [[ -z "${!v:-}" ]]; then
      line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?$v=" "$MP/scripts/stage1.env" | tail -1 || true)
      [[ -n "$line" ]] && export "$v=$(echo "${line#*=}" | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
    fi
  done
fi
# Still unset: this account's locations (mpmoshi docs/AUDIO_AND_DATA.md, confirmed on the server).
D=/work/b12901015/datasets
: "${AMI_ROOT:=$D/ami}" "${IF_ROOT:=$D/synthesized_if_dataset_4spk}" "${IF_SD4_ROOT:=$D/self_distill_4spk_if}"
export AMI_ROOT IF_ROOT IF_SD4_ROOT
OUT=work/spk/mt
[[ "${SMOKE:-0}" == "1" ]] && OUT=work/spk/mt_smoke
TOKDIR=models/step_audio2_tok
TOK=$TOKDIR/token2wav/speech_tokenizer_v2_25hz.onnx
POOLS=(if_train sd_train ami_train if_test ami_eval)
