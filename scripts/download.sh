#!/bin/bash
# Download 0xSero/Step-5-Preview-Spark (341.4 GB = 318 GiB, 196 files: 293.4 GB EXL3 experts, ~12.6 GB EXL3 body,
# 35.8 GB BF16 tensors incl. the BF16 body files) and verify it.
#   scripts/download.sh [--no-verify] [MODEL_DIR]          default ~/models/Step-5-Preview-Spark
# The repo is public: no Hugging Face token or login is needed.
# Verification: layout, BF16 body files and every safetensors header/size, then the sha256 of every file listed in the
# repo's sha256-manifest.txt (reads all 341.4 GB). If the manifest is missing the script prints
# "no manifest: checksums NOT verified" and exits non-zero; --no-verify (or SKIP_SHA=1) skips the sha256 pass and
# accepts header/size checks only.
# Env: REVISION (pin a commit), HF_TOKEN (optional; only for a private fork or mirror), SKIP_SHA=1 (same as --no-verify),
#      BODY_FORMAT (hybrid; what verify.py requires: hybrid/bf16 need the body-bf16-*.safetensors files),
#      NEED_GB (400: free-space warning threshold, checkpoint plus headroom).
# Safe to re-run: hf download resumes and skips finished files. No file is split, so there is nothing to reassemble.
# Every Spark needs its own full copy: download once, then scripts/copy-to-peers.sh over the fabric.
# Needs python3 with venv support when no `hf` CLI is on PATH (DGX OS / Ubuntu: sudo apt install -y python3-venv).
set -euo pipefail
NO_VERIFY=${SKIP_SHA:-0}
ARGS=()
for a in "$@"; do
  case "$a" in
    --no-verify) NO_VERIFY=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) ARGS+=("$a") ;;
  esac
done
REPO=0xSero/Step-5-Preview-Spark
MODEL_DIR=${ARGS[0]:-${MODEL_DIR:-$HOME/models/Step-5-Preview-Spark}}
STATE_DIR=${STATE_DIR:-$HOME/.step5-sparks}
BODY_FORMAT=${BODY_FORMAT:-hybrid}
NEED_GB=${NEED_GB:-400}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$MODEL_DIR" "$STATE_DIR"

# Free space (decimal GB): 341.4 GB of files plus headroom for the hf download cache.
free_gb=$(df -Pk "$MODEL_DIR" | awk 'NR==2{print int($4*1024/1e9)}')
have_gb=$(du -sk "$MODEL_DIR" 2>/dev/null | awk '{print int($1*1024/1e9)}')
if [ $((free_gb + have_gb)) -lt "$NEED_GB" ]; then
  echo "warning: ${free_gb} GB free at $MODEL_DIR (+${have_gb} GB already there); the checkpoint is 341.4 GB, keep ~${NEED_GB} GB free" >&2
fi

# hf CLI: use one on PATH, else a private venv (DGX OS blocks pip --user, PEP 668).
HF=$(command -v hf || true)
if [ -z "$HF" ]; then
  if [ ! -x "$STATE_DIR/venv/bin/hf" ]; then
    python3 -m venv "$STATE_DIR/venv" || {
      rm -rf "$STATE_DIR/venv"
      echo "python3 -m venv failed: install venv support (sudo apt install -y python3-venv) and re-run" >&2; exit 1; }
    "$STATE_DIR/venv/bin/pip" install -q -U "huggingface_hub[hf_xet]"
  fi
  HF=$STATE_DIR/venv/bin/hf
fi

echo "downloading $REPO -> $MODEL_DIR"
"$HF" download "$REPO" ${REVISION:+--revision "$REVISION"} --local-dir "$MODEL_DIR"

echo "verifying $MODEL_DIR"
if [ "$NO_VERIFY" = 1 ]; then
  echo "--no-verify: sha256 pass skipped, checksums NOT verified (header/size checks only)"
  python3 "$HERE/verify.py" --body-format "$BODY_FORMAT" "$MODEL_DIR"
else
  python3 "$HERE/verify.py" --sha --body-format "$BODY_FORMAT" "$MODEL_DIR"
fi
du -sh "$MODEL_DIR"
echo "done. Next: WORKERS=\"user@<spark2-fabric-ip> user@<spark3-fabric-ip> user@<spark4-fabric-ip>\" scripts/copy-to-peers.sh $MODEL_DIR"
