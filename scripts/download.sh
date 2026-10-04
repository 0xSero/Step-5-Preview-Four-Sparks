#!/bin/bash
# Download 0xSero/Step-5-Preview-Spark (~245 GB: ~222 GB EXL3 experts + body, ~23 GB BF16 body files) and verify it.
#   scripts/download.sh [MODEL_DIR]          default ~/models/Step-5-Preview-Spark
# Env: REVISION (pin a commit), HF_TOKEN (only if the repo is private for you), SKIP_SHA=1 (skip the sha256 pass),
#      BODY_FORMAT (hybrid; what verify.py requires: hybrid/bf16 need the body-bf16-*.safetensors files).
# Safe to re-run: hf download resumes and skips finished files. No file is split, so there is nothing to reassemble.
# Every Spark needs its own full copy: download once, then scripts/copy-to-peers.sh over the fabric.
set -euo pipefail
REPO=0xSero/Step-5-Preview-Spark
MODEL_DIR=${1:-${MODEL_DIR:-$HOME/models/Step-5-Preview-Spark}}
STATE_DIR=${STATE_DIR:-$HOME/.step5-sparks}
BODY_FORMAT=${BODY_FORMAT:-hybrid}
NEED_GB=${NEED_GB:-250}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$MODEL_DIR" "$STATE_DIR"

# Free space: ~245 GB of files plus some headroom for the hf download cache.
free_gb=$(df -Pk "$MODEL_DIR" | awk 'NR==2{print int($4/1048576)}')
have_gb=$(du -sk "$MODEL_DIR" 2>/dev/null | awk '{print int($1/1048576)}')
if [ $((free_gb + have_gb)) -lt "$NEED_GB" ]; then
  echo "warning: ${free_gb} GB free at $MODEL_DIR (+${have_gb} GB already there); the checkpoint needs ~245 GB" >&2
fi

# hf CLI: use one on PATH, else a private venv (DGX OS blocks pip --user, PEP 668).
HF=$(command -v hf || true)
if [ -z "$HF" ]; then
  [ -x "$STATE_DIR/venv/bin/hf" ] || { python3 -m venv "$STATE_DIR/venv" && "$STATE_DIR/venv/bin/pip" install -q -U "huggingface_hub[hf_xet]"; }
  HF=$STATE_DIR/venv/bin/hf
fi

echo "downloading $REPO -> $MODEL_DIR"
"$HF" download "$REPO" ${REVISION:+--revision "$REVISION"} --local-dir "$MODEL_DIR"

echo "verifying $MODEL_DIR"
SHA=--sha; [ "${SKIP_SHA:-0}" = 1 ] && SHA=
python3 "$HERE/verify.py" $SHA --body-format "$BODY_FORMAT" "$MODEL_DIR"
du -sh "$MODEL_DIR"
echo "done. Next: WORKERS=\"user@<spark2-fabric-ip> user@<spark3-fabric-ip> user@<spark4-fabric-ip>\" scripts/copy-to-peers.sh $MODEL_DIR"
