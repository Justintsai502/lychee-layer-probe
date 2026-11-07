#!/usr/bin/env bash
# Fetch the three Lychee-FD model files from the official repo, pinned to one commit,
# into third_party/lychee_fd_model/ (a package, so their relative imports resolve).
#
# The released weights (HIT-TMG/Lychee-FD) carry parameter names that match the
# LycheeFD class in Training_package/Code/Models/LycheeFD.py exactly
# (model / stoken_model / control_model / merge_model / encoder / adapter / lm_head),
# so that class loads them directly. The runtime module the repo's own inference code
# imports (lychee_fd.models.lychee_duplex) is not part of the public repo.
set -euo pipefail

COMMIT=${LYCHEE_COMMIT:-7bb1068c59b578e1f8e65fb762cd26708c14cc6d}
BASE="https://raw.githubusercontent.com/HITsz-TMG/Lychee-FD/${COMMIT}/Training_package/Code/Models"

cd "$(dirname "$0")/.."
DEST=third_party/lychee_fd_model
mkdir -p "$DEST"
for f in LycheeFD.py modeling_step_audio_2.py configuration_step_audio_2.py; do
  curl -fsSL -o "$DEST/$f" "$BASE/$f"
  echo "fetched $DEST/$f ($(wc -l < "$DEST/$f") lines)"
done
touch "$DEST/__init__.py"
echo "Lychee-FD model code at commit $COMMIT (Apache-2.0)" > "$DEST/SOURCE.txt"
