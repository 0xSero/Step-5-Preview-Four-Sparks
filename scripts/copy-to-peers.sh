#!/bin/bash
# Copy the downloaded checkpoint to every worker Spark over the QSFP fabric, then verify it on each.
#   WORKERS="user@10.10.10.11 user@10.10.10.13 user@10.10.10.14" scripts/copy-to-peers.sh [MODEL_DIR]
# (example addresses). WORKERS are the peers' fabric addresses (fast link), not their WiFi/Tailscale names.
# Env: WORKER_MODEL_DIR (default: MODEL_DIR with the local $HOME swapped for each peer's), SSH_OPTS, SKIP_SHA=1,
#      BODY_FORMAT (hybrid), PARALLEL=1 (copy to all peers at once instead of one after another).
# Every Spark holds the full checkpoint (~245 GB incl. the BF16 body files; keep ~250 GB free per node).
# Resumable (rsync --partial). Alternative: run scripts/download.sh on each peer.
set -euo pipefail
read -r -a WK <<<"${WORKERS:-}"
[ ${#WK[@]} -gt 0 ] || { echo "set WORKERS=\"user@<spark2-fabric-ip> user@<spark3-fabric-ip> ...\"" >&2; exit 2; }
MODEL_DIR=${1:-${MODEL_DIR:-$HOME/models/Step-5-Preview-Spark}}
BODY_FORMAT=${BODY_FORMAT:-hybrid}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH_OPTS=${SSH_OPTS:-}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS"
[ -f "$MODEL_DIR/config.json" ] && [ -d "$MODEL_DIR/exl3/experts" ] || { echo "run scripts/download.sh first" >&2; exit 1; }
SHA=--sha; [ "${SKIP_SHA:-0}" = 1 ] && SHA=
LOCAL_N=$(find "$MODEL_DIR" -path "$MODEL_DIR/.cache" -prune -o -type f -name '*.safetensors' -print | wc -l | tr -d ' ')

copy_one() {  # $1 = ssh target
  local w=$1 whome wdir free r
  whome=$(timeout 20 $SSH "$w" 'echo $HOME') || { echo "[$w] cannot ssh non-interactively" >&2; return 1; }
  wdir=${WORKER_MODEL_DIR:-${MODEL_DIR/#$HOME/$whome}}
  timeout 20 $SSH "$w" "mkdir -p '$wdir' && command -v rsync >/dev/null && command -v python3 >/dev/null" \
    || { echo "[$w] needs rsync and python3 (sudo apt install rsync python3)" >&2; return 1; }
  free=$(timeout 20 $SSH "$w" "df -Pk '$wdir' | awk 'NR==2{print int(\$4/1048576)}'; du -sk '$wdir' | awk '{print int(\$1/1048576)}'" | awk '{s+=$1} END{print s+0}') || free=0
  [ "${free:-0}" -ge 250 ] || echo "[$w] warning: ~${free} GB available for $wdir; the checkpoint needs ~245 GB" >&2
  echo "[$w] rsync $MODEL_DIR/ -> $wdir/ (~245 GB)"
  # Skip the hf download cache; everything else (BF16 shards, body-bf16-*, exl3/, tokenizer, template) goes over.
  rsync -a --partial --info=progress2 -e "$SSH" --exclude '.cache/' "$MODEL_DIR/" "$w:$wdir/" || return 1
  echo "[$w] verifying"
  timeout 3600 $SSH "$w" "python3 - $SHA --body-format $BODY_FORMAT '$wdir'" < "$HERE/verify.py" || return 1
  r=$(timeout 60 $SSH "$w" "find '$wdir' -path '$wdir/.cache' -prune -o -type f -name '*.safetensors' -print | wc -l" | tr -d ' ')
  [ "$LOCAL_N" = "$r" ] || { echo "[$w] safetensors file count differs: local $LOCAL_N, peer $r" >&2; return 1; }
  echo "[$w] ready: $wdir ($r safetensors files)"
}

fail=0
if [ "${PARALLEL:-0}" = 1 ]; then
  pids=(); for w in "${WK[@]}"; do copy_one "$w" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  for w in "${WK[@]}"; do copy_one "$w" || fail=1; done
fi
[ $fail = 0 ] && echo "all ${#WK[@]} peers ready" || { echo "one or more peers failed; re-run (rsync resumes)" >&2; exit 1; }
