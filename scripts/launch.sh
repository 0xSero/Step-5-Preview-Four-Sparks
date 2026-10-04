#!/bin/bash
# Serve Step-5-Preview-Spark (EXL3 experts + hybrid EXL3/BF16 body, step5 vLLM plugin) with vLLM over N DGX Sparks.
# Default: 4 Sparks, tensor parallel 4 over the QSFP/RoCE fabric (one head + three workers).
#
# Run on the HEAD Spark (rank 0, serves the API). It starts ranks 1..N-1 on the WORKERS over ssh, then rank 0 here.
#
#   WORKERS="user@10.10.10.11 user@10.10.10.13 user@10.10.10.14" scripts/launch.sh   start all ranks, wait for the API
#   scripts/launch.sh stop                  stop every rank (head and all workers) and every memory guard
#   scripts/launch.sh status                containers, memory, memory guard on every node; API health
#   scripts/launch.sh logs [r0|r1|..]       follow a rank's container log (default r0; rI = I-th entry of WORKERS)
#
# Required:  WORKERS       space-separated ssh targets of the other Sparks, ideally their fabric (QSFP) addresses
#                          (the example addresses above are placeholders). Must work non-interactively (key auth).
#                          N = 1 + number of workers; rank I runs on the I-th worker.
# Optional:  TP (N) PP (1) tensor / pipeline parallel sizes, TP*PP must equal N
#            MODEL_DIR     checkpoint dir on the head (default ~/models/Step-5-Preview-Spark)
#            WORKER_MODEL_DIR  same dir on every worker (default: MODEL_DIR with the head's $HOME swapped for the worker's)
#            STATE_DIR     api key, kernel caches, memguard (default ~/.step5-sparks, every node, same rule as above)
#            HEAD_IP HEAD_IFNAME HEAD_HCAS          head fabric IPv4 / netdev / NCCL HCA list (default: auto-detected)
#            WORKER_IPS WORKER_IFNAMES WORKER_HCAS  same per worker, space-separated in WORKERS order, '-' = auto.
#                          A worker's IP defaults to the address in its ssh target when that is an IPv4.
#            PORT (8000) MPORT (29665) IMG API_KEY_FILE MEMGUARD_GIB (1) SSH_OPTS
#            DTYPE KV_BYTES MAX_LEN MAX_SEQS BATCH GPU_UTIL NSPEC GRAPH_MODE CAPS EAGER=1 VIDEO_FRAMES
#            BODY_FORMAT (hybrid) ROCE (1) V2_RUNNER (0) SHARED_STREAM_OFF (1) EXL3_INT8_GEMV (1)
#            EXTRA="..."      extra vLLM args (all ranks)      ENV_EXTRA="K=V K=V"  extra container env (all ranks)
#            CLEAR_COMPILE_CACHE=1   clear the torch.compile cache on every node even if the config did not change
#            WAIT=0        return right after starting the containers
#
# Never caps outputs; never touches GPU power or clocks.
set -euo pipefail

IMG=${IMG:-ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e}
SERVED=step-5-preview-spark
read -r -a WK <<<"${WORKERS:-}"
N=$(( ${#WK[@]} + 1 ))
TP=${TP:-$N}; PP=${PP:-1}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Step-5-Preview-Spark}
STATE_DIR=${STATE_DIR:-$HOME/.step5-sparks}
API_KEY_FILE=${API_KEY_FILE:-$STATE_DIR/api_key}
PORT=${PORT:-8000}; MPORT=${MPORT:-29665}; MEMGUARD_GIB=${MEMGUARD_GIB:-1}
KV_BYTES=${KV_BYTES:-20000000000}      # KV cache bytes per rank (BF16 KV): ~1.0M-token pool at TP4
MAX_LEN=${MAX_LEN:-262144}
MAX_SEQS=${MAX_SEQS:-4}
BATCH=${BATCH:-4096}                   # --max-num-batched-tokens (prefill chunk)
GPU_UTIL=${GPU_UTIL:-0.85}
DTYPE=${DTYPE:-float16}                # activations; matches the reference implementation (see README)
NSPEC=${NSPEC:-2}                      # MTP draft tokens (0 = speculative decoding off)
GRAPH_MODE=${GRAPH_MODE:-FULL_DECODE_ONLY}
BODY_FORMAT=${BODY_FORMAT:-hybrid}     # hybrid: EXL3 body for decode, BF16 body (body-bf16-*.safetensors) for prefill
ROCE=${ROCE:-1}                        # RoCEnante all-reduce / all-gather over the fabric (NCCL handles larger messages)
V2_RUNNER=${V2_RUNNER:-0}              # vLLM model runner: 0 = V1 (release), 1 = V2
SHARED_STREAM_OFF=${SHARED_STREAM_OFF:-1}
EXL3_INT8_GEMV=${EXL3_INT8_GEMV:-1}
EXL3_PREFILL=${EXL3_PREFILL:-st}; EXL3_PREFILL_MIN_ROWS=${EXL3_PREFILL_MIN_ROWS:-1}
HERE=$(cd "$(dirname "$0")" && pwd)

SSH_OPTS=${SSH_OPTS:-}
wssh() { local h=$1; shift; timeout "${T:-60}" ssh -o BatchMode=yes -o ConnectTimeout=10 $SSH_OPTS "$h" "$@"; }
need_workers() { [ ${#WK[@]} -gt 0 ] || { echo "set WORKERS=\"user@<spark2-fabric-ip> user@<spark3-fabric-ip> ...\"" >&2; exit 2; }; }
# A head-side path as a word for a worker's shell: paths under the head's $HOME map to the worker's $HOME.
rpath() { case "$1" in "$HOME"/*) printf '"$HOME"%q' "${1#"$HOME"}" ;; *) printf '%q' "$1" ;; esac; }
SD_R=$(rpath "$STATE_DIR")                                         # state dir as seen by a worker shell
MD_R=$( [ -n "${WORKER_MODEL_DIR:-}" ] && printf '%q' "$WORKER_MODEL_DIR" || rpath "$MODEL_DIR")
nth() { local a; read -r -a a <<<"${1:-}"; local v=${a[$2]:-}; [ "$v" = - ] && v=; printf '%s' "$v"; }   # $1 list, $2 index
ipv4_of_target() { local h=${1#*@}; [[ $h =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && printf '%s' "$h" || true; }

# Prints "<ifname> <ipv4> <hca,hca>" for one node. $1 (optional) = the node's fabric IPv4; without it, the first
# netdev with an IPv4 address behind an ACTIVE RoCE port is used (never WiFi, Ethernet or Tailscale).
# HCA order: the RDMA device of the fabric interface FIRST, then one other ACTIVE device on the same port function
# (same s0fN suffix). A fixed order breaks when a Spark carries its fabric address on the second NIC: NCCL and
# RoCEnante pair HCA 0 with HCA 0 across nodes, those land on different subnets and the all-reduce times out.
DETECT='
want=${1:-}; ifn=""; ip4=""
if [ -n "$want" ]; then
  ifn=$(ip -br -4 addr 2>/dev/null | awk -v ip="$want" "{for (i = 3; i <= NF; i++) if (index(\$i, ip \"/\") == 1) {print \$1; exit}}")
  [ -n "$ifn" ] && ip4=$want
else
  for d in /sys/class/infiniband/*; do
    [ -e "$d" ] || continue
    grep -q ACTIVE "$d"/ports/1/state 2>/dev/null || continue
    for n in "$d"/device/net/*; do
      n=$(basename "$n"); a=$(ip -4 -o addr show dev "$n" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -1)
      [ -n "$a" ] && { ifn=$n; ip4=$a; break 2; }
    done
  done
fi
fab=""; [ -n "$ifn" ] && fab=$(ls /sys/class/net/"$ifn"/device/infiniband/ 2>/dev/null | head -1)
hcas=$fab
if [ -n "$fab" ]; then
  for d in /sys/class/infiniband/*; do
    b=$(basename "$d"); [ "$b" = "$fab" ] && continue
    [ "${b: -4}" = "${fab: -4}" ] || continue
    grep -q ACTIVE "$d"/ports/1/state 2>/dev/null || continue
    hcas="$hcas,$b"; break
  done
fi
echo "${ifn:-none} ${ip4:-none} ${hcas:-none}"'

case "${1:-start}" in
stop)
  # Stops EVERY rank: the memory guard only kills the containers on its own node, so a guard kill of the head (or of
  # one worker) leaves the other ranks running and holding their memory until this is run.
  ids=$(timeout 30 docker ps -aq --filter name=st5-r 2>/dev/null || true)
  [ -n "$ids" ] && timeout 60 docker rm -f $ids >/dev/null 2>&1 || true
  bash "$STATE_DIR/memguard.sh" stop 2>/dev/null || true
  echo "head: stopped"
  i=1
  for w in "${WK[@]}"; do
    T=90 wssh "$w" "ids=\$(docker ps -aq --filter name=st5-r); [ -n \"\$ids\" ] && docker rm -f \$ids >/dev/null 2>&1; bash $SD_R/memguard.sh stop 2>/dev/null; true" \
      && echo "r$i $w: stopped" || echo "r$i $w: unreachable, stop it there (docker rm -f st5-r$i)" >&2
    i=$((i+1))
  done
  exit 0 ;;
status)
  st() {  # prints containers, memory and memguard state of the node it runs on; $1 = state dir word
    echo "docker ps -a --filter name=st5-r --format '{{.Names}} {{.Status}}'; awk '/MemAvailable/{printf \"MemAvailable %.1f GiB\", \$2/1048576}' /proc/meminfo; p=$1/memguard.pid; if [ -f \$p ] && kill -0 \$(cat \$p) 2>/dev/null; then echo ', memguard on'; else echo ', memguard OFF'; fi"
  }
  echo "== r0 (head)"; bash -c "$(st "$(rpath "$STATE_DIR")")" || true
  i=1; for w in "${WK[@]}"; do echo "== r$i $w"; T=20 wssh "$w" "$(st "$SD_R")" || echo "unreachable"; i=$((i+1)); done
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health || true); echo "health: $code"
  exit 0 ;;
logs)
  r=${2:-r0}; i=${r#r}
  [[ $i =~ ^[0-9]+$ ]] || { echo "usage: $0 logs [r0|r1|..]" >&2; exit 2; }
  if [ "$i" = 0 ]; then docker logs -f --tail 200 st5-r0
  else need_workers; [ "$i" -le ${#WK[@]} ] || { echo "no rank $r (${#WK[@]} workers)" >&2; exit 2; }
       T=86400 wssh "${WK[$((i-1))]}" "docker logs -f --tail 200 st5-r$i"; fi
  exit 0 ;;
start) ;;
*) echo "usage: $0 [start|stop|status|logs [rN]]" >&2; exit 2 ;;
esac

need_workers
[ $((TP * PP)) = "$N" ] || { echo "TP*PP ($TP*$PP) must equal the node count $N (head + ${#WK[@]} workers)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required on the head" >&2; exit 1; }
timeout 10 docker info >/dev/null 2>&1 || { echo "docker is not answering on the head (is the daemon up, is $USER in the docker group?)" >&2; exit 1; }

# ---- fabric: per-node interface, IPv4 and HCA order ------------------------------------------------------------
read -r h_if h_ip h_hca <<<"$(bash -c "$DETECT" _ "${HEAD_IP:-}")"
HEAD_IFNAME=${HEAD_IFNAME:-$h_if}; HEAD_IP=${HEAD_IP:-$h_ip}; HEAD_HCAS=${HEAD_HCAS:-$h_hca}
for v in "$HEAD_IFNAME" "$HEAD_IP" "$HEAD_HCAS"; do
  [ "$v" != none ] || { echo "head: could not detect the fabric interface/IP/HCAs; set HEAD_IP, HEAD_IFNAME, HEAD_HCAS" >&2; exit 1; }
done
echo "r0 head        $HEAD_IP on $HEAD_IFNAME, HCAs $HEAD_HCAS, model $MODEL_DIR"
W_IP=(); W_IF=(); W_HCA=()
for i in "${!WK[@]}"; do
  w=${WK[$i]}
  ip=$(nth "${WORKER_IPS:-}" "$i"); ip=${ip:-$(ipv4_of_target "$w")}
  det=$(printf '%s\n' "$DETECT" | T=20 wssh "$w" "bash -s -- $ip") \
    || { echo "cannot ssh to $w non-interactively (key auth from the head is required)" >&2; exit 1; }
  read -r d_if d_ip d_hca <<<"$det"
  W_IF[$i]=$(nth "${WORKER_IFNAMES:-}" "$i"); W_IF[$i]=${W_IF[$i]:-$d_if}
  W_IP[$i]=${ip:-$d_ip}
  W_HCA[$i]=$(nth "${WORKER_HCAS:-}" "$i"); W_HCA[$i]=${W_HCA[$i]:-$d_hca}
  for v in "${W_IF[$i]}" "${W_IP[$i]}" "${W_HCA[$i]}"; do
    [ "$v" != none ] && [ -n "$v" ] || { echo "r$((i+1)) $w: could not detect the fabric interface/IP/HCAs; set WORKER_IPS / WORKER_IFNAMES / WORKER_HCAS" >&2; exit 1; }
  done
  [ "${W_IP[$i]%.*}" = "${HEAD_IP%.*}" ] || echo "warning: r$((i+1)) fabric IP ${W_IP[$i]} is not in the head's /24 ($HEAD_IP)" >&2
  echo "r$((i+1)) $w  ${W_IP[$i]} on ${W_IF[$i]}, HCAs ${W_HCA[$i]}"
done

# ---- preflight: full checkpoint on every node (layout, safetensors headers, BF16 body files), image present ------
python3 "$HERE/verify.py" --body-format "$BODY_FORMAT" "$MODEL_DIR" || { echo "head: download the weights first (scripts/download.sh)" >&2; exit 1; }
for w in "${WK[@]}"; do
  T=900 wssh "$w" "python3 - --body-format $BODY_FORMAT $MD_R" < "$HERE/verify.py" \
    || { echo "$w: copy the weights first (scripts/copy-to-peers.sh)" >&2; exit 1; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || timeout 3600 docker pull "$IMG"
for w in "${WK[@]}"; do T=3600 wssh "$w" "docker image inspect $(printf %q "$IMG") >/dev/null 2>&1 || docker pull $(printf %q "$IMG")"; done

# ---- vLLM configuration ------------------------------------------------------------------------------------------
CAP=$((MAX_SEQS * (NSPEC + 1)))
CAPS=${CAPS:-$(python3 -c "c=$CAP;print(','.join(map(str,sorted(set([x for x in (1,2,4) if x<=c]+list(range(8,c+1,8))+[c])))))")}
COMP=$(printf '{"cudagraph_mode":"%s","cudagraph_capture_sizes":[%s]}' "$GRAPH_MODE" "$CAPS")

envs() {  # $1 host ip, $2 ifname, $3 hcas  ->  shell-quoted "-e K=V" list
  local a=(-e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP="$1"
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER="$V2_RUNNER" -e VLLM_USE_BREAKABLE_CUDAGRAPH=0
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    -e OMP_NUM_THREADS=16 -e MALLOC_ARENA_MAX=2 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_GID_INDEX=3
    -e NCCL_IB_HCA="$3" -e NCCL_IB_MERGE_NICS=1 -e NCCL_SOCKET_IFNAME="$2" -e GLOO_SOCKET_IFNAME="$2"
    -e NCCL_DEBUG=WARN -e NCCL_RAS_ENABLE=0
    -e XDG_CACHE_HOME=/cache -e B12X_COMPILE_CACHE_DIR=/cache/b12x
    -e ST_EXL3_PREFILL="$EXL3_PREFILL" -e ST_EXL3_PREFILL_MIN_ROWS="$EXL3_PREFILL_MIN_ROWS" -e STEP5_MODEL_DIR=/model
    -e STEP5_BODY_FORMAT="$BODY_FORMAT" -e EXL3_INT8_GEMV="$EXL3_INT8_GEMV"
    # REQUIRED (=1): with the shared expert on a side CUDA stream, the EXL3 MoE kernel and the shared-expert GEMMs
    # wait on each other and a rank deadlocks (decode or CUDA-graph capture hangs forever, no error). Running the
    # shared expert on the main stream costs little at these batch sizes and removes the hang.
    -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="$SHARED_STREAM_OFF")
  if [ "$ROCE" = 1 ]; then   # RoCEnante: small all-reduce / all-gather over RDMA verbs; bigger messages use NCCL
    a+=(-e VLLM_ENABLE_ROCE_ALLREDUCE=1 -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB -e VLLM_ROCE_ALLGATHER_MAX_SIZE=16MB
        -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e B12X_ROCE_TRAFFIC_CLASS=106 -e NCCL_IB_TC=106)
  fi
  local kv; for kv in ${ENV_EXTRA:-}; do a+=(-e "$kv"); done
  printf '%q ' "${a[@]}"
}
dock() {  # $1 model dir word, $2 state dir word (already shell-quoted)
  echo "-d --gpus all --network host --ipc host --privileged --ulimit memlock=-1 --ulimit nofile=1048576:1048576" \
    "--shm-size 32g -v $1:/model:ro -v $2/cache:/cache"
}
ARGS=(/model --served-model-name "$SERVED" --dtype "$DTYPE" --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --nnodes "$N" --master-addr "$HEAD_IP" --master-port "$MPORT"
  --kv-cache-memory-bytes "$KV_BYTES" --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH"
  --enable-chunked-prefill --enable-prefix-caching --safetensors-load-strategy lazy
  --limit-mm-per-prompt '{"image":4,"video":1}' --generation-config vllm)
if [ -n "${EAGER:-}" ]; then ARGS+=(--enforce-eager); else ARGS+=(--compilation-config "$COMP"); fi
[ "$NSPEC" -gt 0 ] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}")
[ -n "${VIDEO_FRAMES:-}" ] && ARGS+=(--media-io-kwargs "{\"video\":{\"num_frames\":$VIDEO_FRAMES}}")
[ -n "${EXTRA:-}" ] && { read -r -a ex <<<"$EXTRA"; ARGS+=("${ex[@]}"); }
q() { printf '%q ' "$@"; }

# torch.compile AOT cache: vLLM keys it on the model, not on the plugin settings (body format, MTP, env). A graph
# compiled for another body layout is then reused and fails at load (KeyError: 'weight'). Clear it on every node
# whenever this fingerprint changes. The files are root-owned (written by the container), so the clear runs in a
# throwaway container of the serving image instead of a host rm.
FP=$(printf '%s\n' "$IMG" "$MODEL_DIR" "$BODY_FORMAT" "$NSPEC" "$TP" "$PP" "$N" "$V2_RUNNER" "$SHARED_STREAM_OFF" \
  "$EXL3_INT8_GEMV" "$ROCE" "$GRAPH_MODE" "$CAPS" "$MAX_SEQS" "$BATCH" "$MAX_LEN" "${EAGER:-}" "${EXTRA:-}" "${ENV_EXTRA:-}" \
  | python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:16])')
cc_cmd() {  # $1 state dir word
  echo "mkdir -p $1/cache; f=$1/cache/fingerprint; if [ \"\$(cat \$f 2>/dev/null)\" != $FP ] || [ ${CLEAR_COMPILE_CACHE:-0} = 1 ]; then" \
    "docker run --rm --entrypoint rm -v $1/cache:/c $(printf %q "$IMG") -rf /c/vllm/torch_compile_cache" \
    "&& echo $FP > \$f && echo \"\$(hostname): torch.compile cache cleared (config fingerprint $FP)\"; fi"
}
mg_cmd() { echo "nohup setsid bash $1/memguard.sh $MEMGUARD_GIB >/dev/null 2>&1 < /dev/null &"; }   # $1 state dir word

# ---- state: api key, caches, memory guards (every node) ----------------------------------------------------------
mkdir -p "$STATE_DIR/cache"
[ -s "$API_KEY_FILE" ] || { (umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(32))" > "$API_KEY_FILE"); echo "new API key in $API_KEY_FILE"; }
SD_L=$(rpath "$STATE_DIR")
bash -c "$(cc_cmd "$SD_L")"
cp "$HERE/memguard.sh" "$STATE_DIR/memguard.sh"; bash -c "$(mg_cmd "$SD_L")"
for w in "${WK[@]}"; do
  T=30 wssh "$w" "mkdir -p $SD_R/cache && cat > $SD_R/memguard.sh" < "$HERE/memguard.sh"
  T=30 wssh "$w" "$(mg_cmd "$SD_R")"
  T=300 wssh "$w" "$(cc_cmd "$SD_R")"
done

# ---- ranks 1..N-1 on the workers, then rank 0 here ---------------------------------------------------------------
for i in "${!WK[@]}"; do
  r=$((i+1))
  T=120 wssh "${WK[$i]}" "docker rm -f st5-r$r >/dev/null 2>&1; docker run --name st5-r$r $(dock "$MD_R" "$SD_R") \
    $(envs "${W_IP[$i]}" "${W_IF[$i]}" "${W_HCA[$i]}") $(q "$IMG") $(q "${ARGS[@]}") --node-rank $r --headless" >/dev/null
done
timeout 60 docker rm -f st5-r0 >/dev/null 2>&1 || true
eval "docker run --name st5-r0 $(dock "$(printf %q "$MODEL_DIR")" "$SD_L") $(envs "$HEAD_IP" "$HEAD_IFNAME" "$HEAD_HCAS") \
  $(q "$IMG") $(q "${ARGS[@]}") --node-rank 0 --host 0.0.0.0 --port $PORT --api-key \"\$(cat $(q "$API_KEY_FILE"))\" \
  --reasoning-parser step3p5 --tool-call-parser step3p5 --enable-auto-tool-choice \
  --enable-prompt-tokens-details --enable-force-include-usage" >/dev/null
echo "started st5-r0 here and st5-r1..st5-r${#WK[@]} on the workers: TP$TP PP$PP, body $BODY_FORMAT, MTP k=$NSPEC," \
  "RoCE all-reduce $ROCE, KV $KV_BYTES B/rank, max len $MAX_LEN, seqs $MAX_SEQS"

# ---- wait for the API (first boot: weights, EXL3 layers, kernel compile, CUDA graphs) ----------------------------
[ "${WAIT:-1}" = 1 ] || exit 0
KEY=$(cat "$API_KEY_FILE"); t0=$(date +%s)
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:$PORT/v1/models || true)
  [ "$code" = 200 ] && { echo; echo "ready after $(( $(date +%s) - t0 )) s: http://$HEAD_IP:$PORT/v1  model $SERVED"; exit 0; }
  dead=""
  [ "$(docker inspect -f '{{.State.Running}}' st5-r0 2>/dev/null || echo false)" = true ] || dead="r0"
  for i in "${!WK[@]}"; do
    s=$(T=20 wssh "${WK[$i]}" "docker inspect -f '{{.State.Running}}' st5-r$((i+1)) 2>/dev/null" || echo unknown)
    [ "$s" = false ] && dead="$dead r$((i+1))"
  done
  if [ -n "$dead" ]; then
    echo; echo "rank(s) exited: $dead. Last log lines:" >&2
    docker logs --tail 40 st5-r0 2>&1 | sed 's/^/r0| /' >&2 || true
    for i in "${!WK[@]}"; do T=20 wssh "${WK[$i]}" "docker logs --tail 40 st5-r$((i+1)) 2>&1" | sed "s/^/r$((i+1))| /" >&2 || true; done
    echo "the other ranks are still running and holding memory: run scripts/launch.sh stop" >&2
    exit 1
  fi
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { echo "not ready after 45 min; check scripts/launch.sh logs" >&2; exit 1; }   # measured load ~12 min
  printf '.'; sleep 15
done
