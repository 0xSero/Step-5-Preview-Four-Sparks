# Step-5-Preview on four DGX Sparks

Scripts to serve StepFun's Step-5-Preview on four NVIDIA DGX Spark (GB10) nodes with vLLM, tensor parallel 4 over
the Sparks' QSFP/RoCE fabric, and to use it from the [Pi coding agent](https://github.com/earendil-works/pi).

Repository: `github.com/0xSero/Step-5-Preview-Four-Sparks`.

## What this is

The checkpoint is [`0xSero/Step-5-Preview-Spark`](https://huggingface.co/0xSero/Step-5-Preview-Spark), built from
[`SHSLab/Step-5-Preview-BF16`](https://huggingface.co/SHSLab/Step-5-Preview-BF16):

- **Routed experts** (352 per MoE layer) are EXL3 trellis (MUL1 codebook), one file per layer in `exl3/experts/`.
- **Body projections** (attention q/k/v/o, shared-expert and dense MLP projections) ship twice:
  - EXL3, one file per layer in `exl3/body/` (used for decode-sized batches);
  - BF16, in `body-bf16-*.safetensors` (~23 GB, used for prefill-sized batches).

  This is the **hybrid body** (`STEP5_BODY_FORMAT=hybrid`, the release default). It keeps prefill and the scored
  path close to BF16 while decode reads the smaller EXL3 weights.
- **Kept in BF16** (`model-0000N.safetensors`): embeddings, LM head, norms, router gate and bias, vision tower and
  projector, and the MTP draft layers.
- The sparse-attention indexer weights are not shipped: attention runs dense.
- Bits per weight: routed experts **K4 everywhere (4.01 bpw)**; EXL3 body **K8** for decode. Total size **~342 GB**
  (293.4 GB EXL3 experts, ~12.6 GB EXL3 body, 35.8 GB BF16 tensors including the BF16 body copy). Files are at most ~5 GiB each, so nothing is split or needs reassembly.

The server image is `ghcr.io/0xsero/step-5-preview-spark` (`@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e`, tag `s1`, built and attested by the local-ai-images GitHub workflow). It is vLLM with the EXL3
MoE path, the B12X Spark kernels (including RoCEnante, an RDMA all-reduce for small messages) and a `step5` plugin
that registers the model (text, vision, MTP drafter) and the `step5_exl3` quantization method. No
`--trust-remote-code` is needed. The served model name is `step-5-preview-spark`.

## Measured

Four DGX Spark (GB10), TP4 over RoCE, V1 model runner, fp16 activations, hybrid body, MTP with 2 draft tokens, RoCEnante all-reduce,
CUDA graphs `FULL_DECODE_ONLY`, 262,144 context. All decode runs used the server's default sampling and stopped
naturally (no output caps).

Final checkpoint on 4x DGX Spark TP4 (image recipe identical to the published digest), measured 2026-10-04 to 2026-10-06.
Speed varies between server launches of the same configuration (per-launch mean decode step ~75-81 vs ~90 ms observed)
and with context length (~81 ms/step at 1-2k generated tokens, ~102 at 20-30k); ranges below span several launches.

| Metric | Result |
|---|---|
| Prefill, 8k prompt | cold first request 1,106-1,327 tok/s, warm 1,365-1,565 tok/s over 4 launches (`NCCL_PROTO=Simple`; one launch without it: 1,353) |
| Prefill, 32k prompt | 1,445 tok/s (one launch) |
| Decode, 1 stream | ~25-30 tok/s per stream (paired benchmark, several launches), short of the 75 tok/s target |
| MTP acceptance (2 draft tokens) | per position 0.89 / 0.67, mean accepted length 2.56 |
| Decode, 4 streams | code 66.5 tok/s aggregate (19.0 per stream), prose 40.3 aggregate |
| KV cache pool | 999,279 tokens at 262,144 context, 20 GB KV per rank |
| Max context per request | 262,144: a 261,632-token prompt + answer recalls the passphrase (PASS, 4 runs, ~320-340 s); a 262,145-token prompt is rejected (HTTP 400) |
| Text, tool calls, reasoning, vision, video | text / tools / image / video PASS (`scripts/smoke.py`) |
| Load time | ~12-13 min (697-802 s measured, page cache warm; first boot from cold disk is longer) |

Quality: full-vocabulary token-wise KL divergence against the BF16 reference on a held-out panel of 64 windows x
2,048 tokens (65,536 scored positions), 95% bootstrap confidence intervals, measured on the final checkpoint and
release configuration in the serving runtime with the decode-path EXL3 body on every token, captured in eager mode
without MTP or CUDA graphs on prompt positions (closest offline match to serving, not the serving path itself). Target:
top-1 >= 93% and mean KL ~0.07: point estimates meet it; the top-1 interval crosses 93%. Windows 0-7 of the panel were
also used to choose between candidate builds; the 56 never-used windows give KL 0.0737 / top-1 93.38%. A serving-path
cross-check (same checkpoint, 4x RTX PRO 6000, CUDA graphs + MTP, prompt positions through the BF16 prefill body) gives
KL 0.0716 (0.0652-0.0788) / top-1 93.34%, consistent with the number below.
For scale: the unquantized model with only a different summation order already scores KL 0.027 / top-1 96.1% against
this teacher (8 windows).

| Panel | Mean KL (nats) | Top-1 agreement | dNLL vs BF16 |
|---|---:|---:|---:|
| held-out 64 x 2,048 | 0.0735 (0.0674-0.0800) | 93.3% (92.8-93.8%) | +0.013 nats |

## Hardware and requirements

- 4x DGX Spark (DGX OS). One acts as the **head** (rank 0, serves the API on port 8000); the other three are
  **workers** (ranks 1-3), listed in `WORKERS`.
- A QSFP fabric with RoCE up that puts all four nodes on one IPv4 subnet (for example a QSFP switch). On each node at
  least one RDMA port must be `ACTIVE` (`rdma link`) with an IPv4 address on that subnet on its netdev (example
  addresses in this README: head 10.10.10.12, workers 10.10.10.11, 10.10.10.13, 10.10.10.14).
- Passwordless ssh from the head to every worker (key auth; use the workers' fabric addresses).
- Docker with the NVIDIA runtime on all four nodes, your user in the `docker` group.
- **~250 GB free NVMe on every node.** Each node holds the full ~245 GB checkpoint.
- `python3`, `rsync`, `curl` on all nodes. Node.js + npm on whatever machine runs Pi.

## Quick start

All commands run on the head unless noted. `WORKERS` lists the ssh targets of the other three Sparks, in rank order.

```bash
git clone https://github.com/0xSero/Step-5-Preview-Four-Sparks && cd Step-5-Preview-Four-Sparks
export WORKERS="user@10.10.10.11 user@10.10.10.13 user@10.10.10.14"   # example: the workers' fabric addresses
```

1. **Download the weights** (~245 GB) to `~/models/Step-5-Preview-Spark`. The script resumes, then checks the
   layout, the BF16 body files and every safetensors header (and sha256, if the repo carries `sha256-manifest.txt`;
   `SKIP_SHA=1` skips it):

   ```bash
   scripts/download.sh
   ```

2. **Copy to every worker** over the fabric and verify there (`PARALLEL=1` copies to all three at once):

   ```bash
   scripts/copy-to-peers.sh
   ```

   (Or run `scripts/download.sh` on each worker.)

3. **Pull the image** on all nodes. `launch.sh` pulls it where missing; to do it ahead of time:

   ```bash
   docker pull ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e
   ```

4. **Launch** all four ranks. The script detects each node's fabric interface and HCA order, checks the checkpoint
   on every node, writes an API key to `~/.step5-sparks/api_key`, starts a memory guard on every node, clears stale
   compile caches, starts ranks 1-3 on the workers and rank 0 here, and waits until the API answers (first boot
   about 12 min):

   ```bash
   scripts/launch.sh            # also: scripts/launch.sh status | logs [r0|r1|r2|r3] | stop
   ```

5. **Smoke test** (text, tool call, image, video; each prints PASS/FAIL, video prints SKIP without OpenCV):

   ```bash
   python3 scripts/smoke.py http://127.0.0.1:8000
   ```

6. **Bench** (optional):

   ```bash
   python3 scripts/bench.py --out bench.json --conc 1 2 4
   ```

7. **Connect Pi** (on any machine that can reach the head):

   ```bash
   pi/install.sh http://<head-ip>:8000/v1
   export STEP5_API_KEY=$(ssh <head> cat ~/.step5-sparks/api_key)
   pi --model step5-sparks/step-5-preview-spark
   ```

The endpoint is OpenAI-compatible: `http://<head-ip>:8000/v1`, model `step-5-preview-spark`, bearer auth with the
key above. Images go in as `image_url` parts (up to 4 per prompt), video as one `video_url` part.

## Pi

`pi/install.sh` installs Pi (`npm install -g @earendil-works/pi-coding-agent`) if `pi` is not on PATH, then:

- adds or replaces only the `step5-sparks` provider in `~/.pi/agent/models.json` (`$PI_CODING_AGENT_DIR` if set);
  other providers are untouched and a timestamped backup is written first. See `pi/models.fragment.json`.
- appends `step5-sparks/step-5-preview-spark` to `enabledModels` in `settings.json` only if you use that list;
  `SET_DEFAULT=1` also makes it the startup model.
- installs `pi/step5-sparks.ts` as a Pi extension. Pi always sends an output-token limit; the extension removes
  `max_tokens` / `max_completion_tokens` for this provider so answers run to their natural end.

The model entry: reasoning on, text + image input, 262,144 context, tools via the server's `step3p5` parser.
Pi thinking levels map to the chat template's `reasoning_effort` as minimal/low -> `low`, medium -> `medium`,
high/xhigh/max -> `high`. The key is read from `$STEP5_API_KEY` at request time and is never written to Pi's config.

Checked with Pi 1.0.0 against the served model: a `pi -p` task (write a script, run it with bash, report the output)
completed with both tool calls and the correct answer.

## Configuration

`scripts/launch.sh` reproduces the release configuration. Main settings (override via environment):

| Setting | Value |
|---|---|
| Nodes | head + `WORKERS` (default layout: 4 Sparks); `TP` defaults to the node count, `PP` to 1, `TP*PP` must equal it |
| Model runner | V1 (`VLLM_USE_V2_MODEL_RUNNER=0`; `V2_RUNNER=1` switches) |
| Activations | `--dtype float16` (DTYPE): the reference implementation runs fp16; bf16 changes the selected experts for ~30% of tokens per MoE layer in an 8-layer comparison (fp16: 5-10%) |
| Body format | `hybrid` (BODY_FORMAT -> `STEP5_BODY_FORMAT`): EXL3 body for decode, BF16 body files for prefill; `exl3` runs without the BF16 body files |
| Speculative decoding | MTP (`{"method":"mtp","num_speculative_tokens":2}`), NSPEC=2; `NSPEC=0` turns it off |
| All-reduce | RoCEnante on (ROCE=1): `VLLM_ENABLE_ROCE_ALLREDUCE=1`, all-reduce up to 2 MB, all-gather up to 16 MB, traffic class 106 (`B12X_ROCE_TRAFFIC_CLASS`, `NCCL_IB_TC`); larger messages and `ROCE=0` use NCCL over RoCE |
| Shared experts | `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (SHARED_STREAM_OFF), required, see Troubleshooting |
| EXL3 GEMV | `EXL3_INT8_GEMV=1` (int8 activations + fp16 residual: about fp16 accuracy at close to int8 speed; mode 0 is exact but ~1.6x slower GEMV, mode 2 loses accuracy at 1-2 rows) |
| KV cache | model dtype (FP16), `--kv-cache-memory-bytes 20000000000` per rank (KV_BYTES): ~1.0M-token pool |
| Context / batch | `--max-model-len 262144` (MAX_LEN), `--max-num-seqs 4` (MAX_SEQS), `--max-num-batched-tokens 4096` (BATCH) |
| GPU memory | `--gpu-memory-utilization 0.85` (GPU_UTIL) |
| CUDA graphs | `FULL_DECODE_ONLY` (GRAPH_MODE), capture sizes from MAX_SEQS x (NSPEC+1) (CAPS); `EAGER=1` disables graphs |
| EXL3 prefill | batched expert prefill kernel on (EXL3_PREFILL=st, EXL3_PREFILL_MIN_ROWS=1) |
| Multimodal | `--limit-mm-per-prompt {"image":4,"video":1}`; video frames sampled uniformly (vLLM default 32, VIDEO_FRAMES overrides) |
| Parsers | `step3p5` reasoning and tool-call parsers, `--enable-auto-tool-choice` |
| Sampling | `--generation-config vllm` (server defaults; clients set temperature etc.) |
| Fabric | NCCL over RoCE (`NCCL_IB_HCA` per node, fabric NIC first; GID index 3), Gloo/NCCL sockets and `VLLM_HOST_IP` on the fabric IPs, `NCCL_RAS_ENABLE=0`, `NCCL_PROTO=Simple` (prefill all-reduces: +13 % prefill at 8k) |
| Containers | `--ulimit nofile=1048576:1048576`, `--ulimit memlock=-1`, host network and IPC |
| Host | `MALLOC_ARENA_MAX=2`, memory guard floor 1 GiB (MEMGUARD_GIB) |
| Ports | API 8000 (PORT), rank rendezvous 29665 (MPORT) |
| Extra | `EXTRA="..."` extra vLLM args, `ENV_EXTRA="K=V ..."` extra container env (both on every rank) |

Fabric overrides: `HEAD_IP`, `HEAD_IFNAME`, `HEAD_HCAS`, and per worker `WORKER_IPS`, `WORKER_IFNAMES`, `WORKER_HCAS`
(space-separated in `WORKERS` order, `-` = auto-detect).

Paths: `MODEL_DIR` (default `~/models/Step-5-Preview-Spark`, same path under each worker's home unless
`WORKER_MODEL_DIR`), `STATE_DIR` (default `~/.step5-sparks` on every node: API key, kernel and compile caches,
memory guard log). The checkpoint is mounted read-only at `/model` in every container (`st5-r0` on the head,
`st5-r1`..`st5-r3` on the workers).

Video: the checkpoint has no temporal module. Each sampled frame goes through the image encoder as one global
view and the frame embeddings are concatenated, so video understanding is frame-by-frame.

## Troubleshooting

- **Memory guard.** A GB10 Spark shares one 128 GB pool between CPU and GPU. If the model drives `MemAvailable` to
  zero the host does not OOM-kill cleanly; it wedges until the hardware watchdog reboots it. `scripts/memguard.sh`
  runs on every node and kills that node's `st5-r*` container when `MemAvailable` falls below 1 GiB. Log:
  `~/.step5-sparks/memguard.log` on that node. A guard only sees its own node: when it kills one rank (for example
  the head), the other ranks keep running, stalled and still holding their memory. Run `scripts/launch.sh stop`,
  which removes the containers and stops the guards on the head **and** every worker. If a guard fires, lower
  `KV_BYTES` or `GPU_UTIL` before raising the floor. `scripts/launch.sh status` shows each node's memory and
  whether its guard is running.
- **Hang in decode or CUDA-graph capture (shared-expert stream).** With the shared expert on a side CUDA stream, the
  EXL3 MoE kernel and the shared-expert GEMMs wait on each other and a rank deadlocks: no error, GPU busy or idle,
  the request never finishes, or graph capture never ends. `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (the default)
  runs the shared expert on the main stream and removes it. Do not set `SHARED_STREAM_OFF=0`.
- **`KeyError: 'weight'` or other load errors after changing settings (compile cache).** vLLM keys its
  torch.compile cache on the model, not on the plugin settings, so a graph compiled for another body format or MTP
  setting can be reused. `launch.sh` fingerprints the configuration and clears
  `STATE_DIR/cache/vllm/torch_compile_cache` on every node when it changes. The files are owned by root (written
  from the container), so the clear runs in a throwaway container. Force it with `CLEAR_COMPILE_CACHE=1`, or by hand:
  `docker run --rm --entrypoint rm -v ~/.step5-sparks/cache:/c <image> -rf /c/vllm/torch_compile_cache`.
- **All-reduce timeouts at start (per-node HCA order).** NCCL and RoCEnante pair the first HCA in `NCCL_IB_HCA` on
  one node with the first on the others. If a Spark carries its fabric address on its second NIC, a fixed list
  pairs devices on different subnets and the first all-reduce times out. `launch.sh` builds the list per node with
  the RDMA device behind that node's fabric interface first (printed at start). If detection is wrong, set
  `HEAD_HCAS` / `WORKER_HCAS` (see `ip -br a`, `rdma link`, `ls /sys/class/net/<ifname>/device/infiniband/`).
- **Fabric pinning.** vLLM, NCCL and Gloo must bind to the QSFP addresses, not WiFi, Ethernet or Tailscale. The
  launcher sets `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` and `NCCL_IB_HCA` per node. A worker's
  fabric IP is taken from its ssh target when that is an IPv4 address, otherwise detected; override with
  `WORKER_IPS` / `WORKER_IFNAMES`. It warns if a node's fabric IP is not in the head's /24.
- **Disk space.** Every node needs the full ~245 GB checkpoint plus headroom: keep ~250 GB free per node.
  `download.sh` and `copy-to-peers.sh` warn when there is less.
- **Preflight fails with `no body-bf16-*.safetensors`**: the BF16 body files are missing (partial download or an
  older revision). Re-run `scripts/download.sh`, then `scripts/copy-to-peers.sh`. To serve without them, use
  `BODY_FORMAT=exl3` (EXL3 body everywhere; lower prefill quality, not the release setting).
- **Preflight fails with `missing ...` or `size ... implied by its header`**: the download is incomplete or
  truncated; re-run `scripts/download.sh` (it resumes), then `scripts/copy-to-peers.sh`.
- **WiFi-only nodes.** If a Spark's only internet path is WiFi, downloads are slow; download once on the node with
  the best uplink and use `copy-to-peers.sh` over the fabric. Clients (Pi) can reach the head over any network;
  only the rank-to-rank traffic needs the fabric.
- **First boot is slow**: weights load, EXL3 layers, B12X kernel compile (cached in `STATE_DIR/cache` after the
  first run), torch.compile and CUDA graph capture. `launch.sh` waits up to 45 minutes and prints the last log
  lines of every rank if one exits. `scripts/launch.sh logs r2` follows the second worker.
- **One rank exited**: `launch.sh` prints the logs and leaves the others running; run `scripts/launch.sh stop`
  before starting again.
- **Debugging without CUDA graphs**: `EAGER=1 scripts/launch.sh` (slower decode, faster start).
- **Video check skipped**: `smoke.py` builds its test clip with OpenCV; `pip install opencv-python-headless numpy`
  on the machine running the smoke test. The server does not need it.
- **Pi shows no model**: `STEP5_API_KEY` must be set in the shell that starts Pi; run `/model` to reload.
- **`pi -p` hangs in a script**: print mode also reads stdin when it is not a terminal; add `< /dev/null`.

## Repository layout

| Path | What it does |
|---|---|
| `scripts/download.sh` | HF download, then `verify.py` |
| `scripts/verify.py` | checkpoint layout, BF16 body files, safetensors header/size and optional sha256 checks |
| `scripts/copy-to-peers.sh` | rsync the checkpoint to every worker over the fabric, verify on each |
| `scripts/launch.sh` | start / stop / status / logs for all ranks |
| `scripts/memguard.sh` | host memory guard (started by `launch.sh` on every node) |
| `scripts/smoke.py` | text, tool-call, image and video checks, no output caps |
| `scripts/bench.py` | prefill / decode / long-context speed bench, no output caps |
| `pi/install.sh` | install Pi and register the provider |
| `pi/models.fragment.json`, `pi/step5-sparks.ts` | Pi provider entry and request adapter |

## Licence and attribution

- Step-5-Preview: StepFun. The source weights (`SHSLab/Step-5-Preview-BF16`) and this quantized checkpoint are
  under the StepFun Community License of the source; read it before use.
- vLLM: Apache-2.0. Spark serving stack (vLLM fork, B12X kernels): Local Inference Lab, Apache-2.0. The `step5`
  vLLM plugin derives from vLLM model code: Apache-2.0.
- ExLlamaV3 / EXL3 trellis encoder and kernels: turboderp, MIT.
- Pi coding agent: Earendil Works, MIT.
- The scripts in this repository: MIT (see `LICENSE`).
