# Qwen3.5-122B Hybrid v2 — Deployment Runbook

This runbook was produced while deploying and validating [albond's optimization stack](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4#optimization-5-vllm-pr-38325-swapab-sm120-fp8-gemm) on two DGX Spark GB10 units — an **NVIDIA DGX Spark Founders Edition** and an **ASUS Ascent GX10**. It documents the full build process, optimization layers, hardware-specific quirks (including an ASUS GX10 PD firmware wedge that caused a 33% throughput gap), and validated bench results on both units.

Among other things it resolves [issue #15 (pinned torch nightly rolled off PyPI)](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/issues/15) — the `torch==2.12.0.dev20260408+cu130` nightly is gone. This runbook documents the working stable cu130 triple (`torch==2.12.0+cu130 / torchvision==0.27.0+cu130 / torchaudio==2.11.0+cu130`) and the build path adjustments needed to use it.

Base recipe: [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)

> **Note:** This is a sanitized copy for sharing. IPs replaced with SPARK1_IP/SPARK2_IP/LITELLM_HOST_IP.
> Internal hostnames replaced with LITELLM_HOST/NAS_HOST. User paths replaced with USER.

> **Node legend:** `SPARK1_IP` = spark1 — NVIDIA DGX Spark Founders Edition (GB10). `SPARK2_IP` = spark2 — ASUS Ascent GX10 (GB10). `LITELLM_HOST_IP` = LiteLLM gateway host.


**Current state (2026-06-18):** Production — spark1 at **~52 tok/s**, spark2 at **~53 tok/s** (post-PD-wedge cold-drain fix).
**Target ceiling:** ~51.6 tok/s/node (albond reference, GX10 at 0.90 GMU) — **both nodes now exceed this at 0.85 GMU**
**Achieved:** 52 tok/s on Founders, 53 tok/s on GX10 (post-wedge-fix)
**Source:** [albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)
**Build path:** `eugr/spark-vllm-docker` at commit `49d6d9f` → `build-and-copy.sh` (NOT install.sh)
**Last updated:** 2026-06-18 (vLLM pinned to 2a69949bd; GB10 CUDA graph OOM analysis documented; crash-loop recovery runbook added; spark1 baseline memory discrepancy under investigation)
**ob1 memories:** *(internal, omitted)*
**GMU note:** 0.85 is the correct production value on clean boot. 0.85 × 121.69 = 103.44 GiB — after model load (66.93 GiB), 36.5 GiB remains for KV cache + CUDA graph capture (sufficient). 0.80 leaves only 30.4 GiB after model load — CUDA graph capture OOM-kills EngineCore silently. Use 0.80 only as a diagnostic probe to isolate `request_memory()` failures in degraded state. 0.90 fails on both nodes. If `request_memory()` fails at 0.85, the node has crash-loop CUDA residue — reboot it (lesson 11), not lower GMU.
**vLLM version pin:** `2a69949bd` (`0.19.1.dev0+g2a69949bd.d20260616`) — pinned in `Dockerfile` and `build-and-copy.sh`. Do NOT track `main`; newer builds have stricter `request_memory()` that fails at 0.85 on GB10.
**Key fix:** Starlette 1.x breaks vLLM 0.19 router — must pin `starlette<1.0` + `fastapi<0.116`.
**Torch triple (stable cu130, verified in container):**
```
torch==2.12.0+cu130        # stable — nightly dev20260408 no longer on PyPI
torchvision==0.27.0+cu130  # note: lags torch by one minor; 2.12.0 not on stable index
torchaudio==2.11.0+cu130   # same lag — vLLM doesn't use torchaudio, safe to leave as-is
# index: https://download.pytorch.org/whl/cu130
```

## Architecture

```
LiteLLM (LITELLM_HOST:4000) — latency-based routing
    ├── spark1 (SPARK1_IP:8000) — TP=1 single-node, qwen35-hybrid-v2
    └── spark2  (SPARK2_IP:8000) — TP=1 single-node, qwen35-hybrid-v2

Each node is independent. No NCCL cross-node communication.
LiteLLM load-balances across both for fast/reasoning/code pools.
```

### Hardware

| Node | Hardware | GPU/RAM | Storage | Max GMU | Notes |
|------|----------|---------|---------|---------|-------|
| spark1 | NVIDIA DGX Spark (Founders Edition) | GB10 128GB unified | 4TB NVMe | 0.85* | Confirmed baseline ~99 GiB free (NVIDIA firmware overhead). 0.90 fails. *See note. |
| spark2 | ASUS Ascent GX10 | GB10 128GB unified | 1TB NVMe | 0.87 | Higher driver/firmware overhead (~106 GiB free after Phase 0) |

Both run the same GB10 SoC with 128GB unified memory. Storage difference is irrelevant for inference
(model + caches fit in ~80GB). **Nodes are not interconnected** — no DAC, no RoCE, no NCCL cross-node link.
Each node is a fully independent inference endpoint.

**IMPORTANT:** The ASUS GX10 cannot run at 0.90 GMU — the docker launch will fail with
`ValueError: Free memory (106.0 GiB) < desired (109.46 GiB)`. Use 0.85 GMU on spark2.

**\*spark1 Max GMU note (2026-06-18):** The NVIDIA Founders Edition baseline shows only ~99 GiB free at clean boot
vs ~103-118 GiB on the GX10. Root cause under investigation (hardware firmware difference between NVIDIA and ASUS
boards, both running same kernel 6.17.0-1021-nvidia). At ~99 GiB free, 0.85 × 121.69 = 103.44 GiB FAILS the
`request_memory()` check. Current workaround: spark1 needs either (a) the memory baseline restored to ~103+ GiB
or (b) a different GMU value. 0.80 passes the check but causes CUDA graph OOM during capture. Do not use 0.80.
If spark1's memory baseline is not restored, contact NVIDIA or open a NVSM ticket.
The DGX Spark Founders Edition has ~12 GiB more GPU headroom and can run 0.90.

> **Note:** Phase 4-alt (distributed TP=2) is documented for historical reference only.
> The nodes are no longer DAC-connected. Production deployment is independent
> single-node TP=1 on each node (Phase 4). TP=2 is not available.

## What This Build Does

Three optimizations stacked on Intel AutoRound INT4 baseline:

| Layer | Optimization | Effect |
|---|---|---|
| Patch 01 | Hybrid INT4+FP8 — MoE experts stay INT4 Marlin, dense layers (shared expert, attention) use FP8 CUTLASS block-128 | +8.8% |
| Patch 03 | INT8 LM Head v2 — baked into Docker image at build time | +speedup |
| Launch flag | MTP-2 speculative decoding (2 tokens/step, ~95% acceptance) | +25% |
| Launch flag | FlashInfer attention backend (native SM121 kernels) | +16% |

Single-node result (post-wedge-fix, 2026-06-16): **~52 tok/s** on Founders, **~53 tok/s** on GX10, at 262K context (0.85 GMU).
Upstream ceiling: 51.6 tok/s (albond reference, GX10 at 0.90 GMU) — both nodes now match or exceed this at 0.85 GMU.
*Prior reading (pre-fix): GX10 was stuck at ~34 tok/s due to a PD firmware wedge pinning the GPU at 702 MHz / 10W. See [Known Firmware Issues](#known-firmware-issues) for diagnosis and the cold-drain fix.*

### Optimization Stack (cumulative, from upstream spec):
| # | Optimization | Effect | Status |
|---|---|---|---|
| 1 | FlashInfer attention backend | +16% | ✅ Active |
| 2 | Hybrid INT4+FP8 dense layers | +8.8% | ✅ Active (144 FP8 scales) |
| 3 | MTP-2 speculative decoding | +25% | ✅ Active |
| 4 | INT8 LM Head v2 + @triton.autotune | +33% + 1.2% | ✅ Active |
| 5 | PR #38325 swapAB SM120 FP8 GEMM | +0.76% | ✅ Active |

**TP=2 dual-node (NOT available — nodes no longer DAC-connected):** Would give ~56 tok/s per-request but requires direct node interconnect.
Two independent TP=1 provides ~95 tok/s aggregate for 2 concurrent requests.

## Known Firmware Issues

### ASUS GX10 PD wedge → low-clock state (2026-06-16)

**Affected hardware:** ASUS Ascent GX10 nodes only (spark2). Founders Edition (spark1) is not affected. ASUS GX10 lags DGX Spark Founders on PD firmware fixes — every GX10 in the fleet is exposed to this wedge class until ASUS catches up.

**Symptom signature** (all must match — no single symptom is diagnostic alone):

- vLLM throughput 30-50% below spark1 despite identical container, identical checkpoint, identical CMD args
- `nvidia-smi` shows 90%+ GPU utilization under sustained load
- SM clock pinned at **611 MHz** or **708 MHz** under load (vs. expected 2400-2600 MHz)
- Power draw <15W under sustained load (vs. expected 35-45W)
- `nvidia-smi -lgc <max>` accepts the command but the runtime clock is unchanged
- `nvidia-smi -q -d PERFORMANCE` shows NO throttle reasons active (no SW Power Cap, no HW Slowdown, no Thermal)
- SW Power Capping counter near zero on the wedged node (vs. spark1's hundreds of millions of µs)

**Root cause:** USB-C Power Delivery firmware wedge on the GB10 chassis. Documented bug class with NVIDIA-confirmed PD firmware fixes shipped for DGX Spark Founders but not yet for ASUS GX10. Wedge is triggered by OOM-kills during heavy memory load, sleep/wake cycles, or container crash-loops.

**Diagnostic confirmation — the clock-lock-ignored test (run BEFORE cold-drain to capture evidence):**

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2418,3003   # accepts on a wedged GX10
# Now run any sustained inference load. If clock stays <1 GHz under load
# despite this lock command reporting success, wedge is confirmed.
sudo nvidia-smi -rgc
```

The accept-but-not-enforce behavior is mechanistically conclusive: the cap lives BELOW the nvidia-smi userspace layer, which can only mean firmware enforcement.

**Capture forensic evidence BEFORE drain** (current state is reproducible; once drained, ground truth is gone):

```bash
scp scripts/spark2-capture.sh spark2:/tmp/
ssh spark2 'bash /tmp/spark2-capture.sh'
# Stash the resulting tarball off-node:
scp spark2:~/spark2-wedge-evidence-*.tar.gz NAS_HOST:/path/to/storage/wedge-evidence/
```

**Primary fix — AC cold drain:**

1. Stop vLLM cleanly: `ssh spark2 'docker stop vllm-qwen35-v2'`
2. Halt OS: `ssh spark2 'sudo poweroff'`
3. Wait until the unit is fully off (LEDs off, no fan).
4. **Unplug the power brick from the wall.** USB-C end can stay attached to spark2 — the wedge is in the brick + chassis-side PMIC handshake, so breaking AC at the wall forces the brick to fully de-energize and re-negotiate PD power class on reconnect.
5. **Wait 90-120 seconds.** The 60s value cited in the forum thread is a minimum; multiple community reports cite "60s didn't take, 2 min did". Cap discharge is via internal bleed resistors — no harm in waiting longer.
6. Reconnect AC at the wall. Power on spark2.
7. Wait for boot + `systemctl is-active nvidia-persistenced` returns `active`.
8. Restart vLLM: `ssh spark2 'docker start vllm-qwen35-v2'`. Wait for JIT compile + cudagraph capture (~5-10 min on cold start).
9. Validate: `scp scripts/spark2-post-drain.sh spark2:/tmp/ && ssh spark2 'bash /tmp/spark2-post-drain.sh'`. The script prints `VERDICT.txt` indicating WEDGE RESOLVED or WEDGE PERSISTS.

**If cold drain doesn't resolve:** File ASUS support ticket. Submit BIOS string (`GX10DGX.0104.2026.0326.1657` or current), pre-drain tarball, post-drain tarball (proves drain was attempted), and reference NVIDIA Developer Forum "Another ASUS GX10 problem" (2026-05-24, https://forums.developer.nvidia.com/t/another-asus-gx10-problem/371201). Request PD firmware ≥5.8 for GX10.

**Do NOT stage PD firmware components separately** — bricks the unit per https://forums.developer.nvidia.com/t/how-i-recovered-from-bricked-asus-gx10-apt-updates-using-debian-image-and-reinstall-bios-no-data-loss/372063. Use combined SoC+BIOS+EC capsule only.

**Operational prevention:**

- Run spark2 on a smart plug (Z-Wave / UniFi / Kasa) so future cold-drains are a phone tap, not crawling under the desk.
- Track ASUS firmware releases. When ASUS ships an updated PD payload, apply to all GX10 units in lockstep.
- Reduce wedge triggers: avoid OOM-kills (Phase 0 + GMU 0.85 ceiling on GX10 keeps headroom), avoid sleep/wake cycles (DGX OS default is multi-user.target), monitor for container crash-loops.

**Note on SW Power Cap counter:** A high counter on spark1 (1.17B µs and climbing) is positive evidence — it means spark1 routinely tries to draw more than its envelope and gets SW-capped to ~40W, i.e., the GPU is healthy and trying to work hard. Do NOT interpret high SW Power Cap engagement as a problem. The wedged node's near-zero counter is the abnormal signal.

## Pre-Flight Checklist

### Phase 0: Minimize System Footprint (CRITICAL)

DGX OS 7 (both Founders Edition AND ASUS GX10) ships with many services that
auto-enable via socket activation, target wants, or generator units —
`systemctl disable` does NOT prevent them from re-starting on next boot.

**MASK is required for these services on DGX OS — `disable` alone is insufficient.**

Without Phase 0, CUDA reports only ~105 GiB free in containers (need 109.52 GiB
for 0.90 GMU). With Phase 0 + reboot, CUDA reports ~117-118 GiB free.

**Run once on each node (these settings persist across reboots):**

```bash
# Switch to headless multi-user target (no graphical login)
sudo systemctl set-default multi-user.target

# Disable services we don't want
sudo systemctl disable gdm3 plymouth snapd snapd.socket snapd.service \
  snap.firmware-updater.firmware-notifier cups cups-browsed avahi-daemon \
  bluetooth ModemManager gnome-remote-desktop NetworkManager-wait-online \
  dgx-dashboard-admin dgx-dashboard fwupd lldpd multipathd rasdaemon \
  rdma-ndd rtkit-daemon smartmontools wpa_supplicant rpc-statd rpcbind \
  udisks2 upower pulse-agent rsyslog

# MASK services that re-enable on boot via socket activation or wants
# (CRITICAL — disable alone does NOT prevent these from re-starting)
sudo systemctl mask gdm3 plymouth snapd snapd.socket dgx-dashboard \
  dgx-dashboard-admin rdma-ndd rpcbind rpcbind.socket rpc-statd rtkit-daemon \
  smartd smartmontools wpa_supplicant rsyslog fwupd lldpd multipathd \
  rasdaemon udisks2 upower pulse-agent

# Stop monitoring exporters during deployment (re-enable after)
sudo systemctl stop prometheus-node-exporter 2>/dev/null
docker stop nvidia_gpu_exporter node_exporter 2>/dev/null

# Disable tailscale if not needed
sudo systemctl disable tailscaled
sudo systemctl stop tailscaled

# Apply memory sysctls (reserve 5GB from page cache starvation)
sudo tee /etc/sysctl.d/90-dgx-spark-memory.conf > /dev/null <<'EOF'
vm.min_free_kbytes = 5242880
vm.vfs_cache_pressure = 200
vm.swappiness = 1
vm.dirty_background_ratio = 2
vm.dirty_ratio = 5
vm.panic_on_oom = 0
vm.max_map_count = 2097152
vm.zone_reclaim_mode = 0
vm.overcommit_memory = 1
EOF
sudo sysctl -p /etc/sysctl.d/90-dgx-spark-memory.conf

# Disable swap (prevents hidden paging on unified memory — inference workloads
# should never swap; if OOM occurs, fail fast rather than degrade silently)
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab  # Persist across reboots

# Reboot to apply mask + headless target + free GPU memory
sudo reboot
```

**After reboot, verify ONLY these services are running** (anything else is unwanted):
```
containerd.service          
docker.service              
NetworkManager.service      
nvidia-persistenced.service 
```

If any of `rdma-ndd`, `rpcbind`, `rpc-statd`, `rtkit-daemon`, `smartd`,
`wpa_supplicant`, `rsyslog` come back after reboot, mask them again — they
have socket-activated triggers that bypass `disable`.

**Validate CUDA-visible memory** in a fresh container:
```bash
cat > /tmp/memcheck.py << 'EOF'
import torch
f, t = torch.cuda.mem_get_info()
print(f'Free: {f/1024**3:.2f} GiB, Total: {t/1024**3:.2f} GiB, Need 0.90: {t*0.9/1024**3:.2f} GiB')
EOF
docker run --rm --gpus all --net=host --ipc=host --memory=0 --memory-swap=0 \
  -v /tmp/memcheck.py:/tmp/memcheck.py --entrypoint python3 \
  vllm-qwen35-v2 /tmp/memcheck.py

# Expected: Free ~117-118 GiB on Founders Edition (spark1)
# Expected: Free ~106 GiB on ASUS GX10 (spark2, hardware ceiling)
```

# Reboot to apply multi-user.target and free GPU memory
sudo reboot
```

**After reboot, verify clean baseline:**
```bash
# Should show multi-user.target
systemctl get-default

# Should show <8GB used (no GPU-holding desktop processes)
free -g | head -2

# GPU compute apps — NOTE: on GB10 this returns EMPTY even when 20+ GiB is consumed
# (unified memory arch; nvidia-smi process tracking is not supported).
# A clean result here does NOT guarantee free GPU memory on GB10.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# GB10-specific gate: check which PIDs hold the nvidia device file.
# nvidia-persistenced (usually PID ~1580) is expected. Anything else = zombie CUDA context.
PERSISTENCED=$(pgrep nvidia-persistenced)
UNEXPECTED=$(sudo fuser /dev/nvidia0 2>/dev/null | tr ' ' '\n' | grep -v "^${PERSISTENCED}$" | grep -v '^$')
if [ -n "$UNEXPECTED" ]; then
  echo "BLOCKED: PIDs holding GPU device: $UNEXPECTED — stop those processes before launching"
  echo "If they are stale container PIDs: docker stop <container> && docker rm <container>"
  echo "If they persist after docker rm: reboot the node to fully clear CUDA context residue"
fi
```

> **Why this matters:** With desktop services running, CUDA reports only
> ~105 GiB free on a 121.63 GiB device. The 0.90 GMU requires 109.46 GiB.
> Without desktop: ~118+ GiB free → 0.90 passes easily.
>
> **DGX Web Dashboard** (`dgx-dashboard-admin`) stays enabled — it uses
> minimal memory and provides hardware monitoring via browser.
>
> **GB10 memory reality (2026-06-18):** The CUDA driver permanently reserves ~22 GiB on GB10 for driver
> structures (page tables, kernel management). Total CUDA-visible: 121.69 GiB. The ASUS GX10 (spark2) shows
> ~103+ GiB free at clean boot, making 0.85 GMU viable. The NVIDIA Founders Edition (spark1) shows only ~99 GiB
> free — root cause under investigation. After any crash-loop, residue accumulates above the baseline; reboot
> fully clears it (lesson 11). Do NOT lower GMU to 0.80 as a workaround — it passes `request_memory()` but
> causes silent EngineCore OOM-kill during CUDA graph capture (see lesson 10).

### Standard Pre-Flight Checks

```bash
# 1. Verify both nodes are reachable
ssh spark1 'hostname && nvidia-smi --query-gpu=memory.total --format=csv,noheader'
ssh spark2 'hostname && nvidia-smi --query-gpu=memory.total --format=csv,noheader'

# 2. Check current containers (these will be stopped)
ssh spark1 'docker ps --format "{{.Names}} {{.Status}}"'
ssh spark2 'docker ps --format "{{.Names}} {{.Status}}"'

# 3. Check disk space (need ~170GB for models + build)
ssh spark1 'df -h /home'
ssh spark2 'df -h /home'

# 4. Flush page cache before launch (MANDATORY on every deploy)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Phase 1: Build via eugr/spark-vllm-docker (CORRECT path)

**DO NOT use install.sh for building.** Use `eugr/spark-vllm-docker` directly.
install.sh is useful for model prep (Steps 0-2) but its Docker build path
misses the `@triton.autotune` decorator and produces ~47 tok/s instead of 51+.

### Steps 0-2: Model Preparation (host-side, run once)

```bash
ssh spark1
cd ~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

# Step 0: Download (cached = instant)
# Step 1: Build hybrid checkpoint (~20 min, ~71 GB output)
# Step 2: Add MTP weights
# install.sh handles these correctly:
bash install.sh --no-launch   # Only does Steps 0-2 if images missing
```

### Step 3: Build vllm-sm121 base image (30-60 min)

```bash
cd ~
rm -rf spark-vllm-docker
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
git checkout 49d6d9fefd7cd05e63af8b28e4b514e9d30d249f

# Remove temp patches (target main, not v0.19.0)
sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/\&\& rm pr35568.diff/d' Dockerfile
sed -i '/# TEMPORARY PATCH for broken compilation/,/\&\& rm pr38919.diff/d' Dockerfile

# Pin stable torch 2.12.0+cu130 (nightly dev20260408 is GONE from PyPI)
# Critical: same version in BOTH stages to avoid ABI mismatch
sed -i 's|uv pip install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/cu130|uv pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 torchaudio==2.11.0+cu130 triton --index-url https://download.pytorch.org/whl/cu130|g' Dockerfile

# Cherry-pick PR #38325 (swapAB SM120 FP8 GEMM, +0.76%)
PROJECT_DIR=~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
cp "${PROJECT_DIR}/patches/05-pr38325-swapab/pr38325-swapab-fp8-sm120.diff" local-pr38325.diff
# Inject after VLLM_PRS block (find the 'fi' line number, +1)
sed -i '206a \
\
COPY local-pr38325.diff /tmp/local-pr38325.diff\
RUN git apply -v /tmp/local-pr38325.diff && rm /tmp/local-pr38325.diff' Dockerfile

# Build (-j 8 is safe on GB10 during Docker build, only JIT inference OOMs)
./build-and-copy.sh -t vllm-sm121 --vllm-ref 2a69949bd --tf5 -j 8
```

**Time:** ~15 min on Founders (fast NVMe), ~90 min on GX10.

### Step 4: Build vllm-qwen35-v2 thin layer (~2 sec)

```bash
cd ~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4

# IMPORTANT: Add starlette fix to Dockerfile.v2 first (one-time)
# The --tf5 flag pulls transformers 5.x which drags in Starlette 1.x
# which breaks vLLM 0.19's router (_IncludedRouter.path removed)
grep -q 'starlette' docker/Dockerfile.v2 || \
  sed -i '/^ENTRYPOINT/i \
# Fix: Starlette 1.x breaks vLLM 0.19 router\
RUN pip install --no-deps "starlette>=0.40.0,<1.0" "fastapi>=0.115.0,<0.116.0"\
' docker/Dockerfile.v2

docker build -t vllm-qwen35-v2 -f docker/Dockerfile.v2 .
```

## Phase 2: Build on spark2 (option A: transfer images, option B: build in place)

### Option A: Transfer from spark1 (faster if network is good)

```bash
# Copy hybrid checkpoint to spark2 (~71 GB)
rsync -avP --info=progress2 \
    ~/models/qwen35-122b-hybrid-int4fp8/ \
    spark2:~/models/qwen35-122b-hybrid-int4fp8/

# Transfer Docker image to spark2
docker save vllm-qwen35-v2 | ssh spark2 'docker load'

# Verify on spark2
ssh spark2 'docker images | grep vllm-qwen35-v2'
ssh spark2 'ls ~/models/qwen35-122b-hybrid-int4fp8/model.safetensors.index.json'
```

### Option B: Run install.sh on spark2 too (independent build, ~1-2 hrs)

```bash
ssh spark2
cd ~ && git clone https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4.git
cd DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
bash install.sh --no-launch
```

## Phase 3: Stop Current Containers

```bash
# Stop Qwen3.5-122B containers on both nodes
ssh spark1 'docker stop qwen35-122b-nvfp4 && docker rm qwen35-122b-nvfp4' 2>/dev/null
ssh spark2 'docker stop qwen35-122b-nvfp4 && docker rm qwen35-122b-nvfp4' 2>/dev/null

# Verify clean
ssh spark1 'docker ps -a --filter name=qwen --filter name=vllm --format "{{.Names}}"'
ssh spark2 'docker ps -a --filter name=qwen --filter name=vllm --format "{{.Names}}"'

# Verify memory freed
ssh spark1 'free -h'
ssh spark2 'free -h'
```

## Phase 4: Launch Single-Node Instances (Production Topology)

> **Validated config:** Upstream Step 5 from install.sh. Minimal flags only —
> the critical optimizations are `--attention-backend FLASHINFER` and
> `--speculative-config`. Everything else is optional per upstream docs.

Each node runs independently with TP=1 + MTP-2. Launch on both nodes:

```bash
# MANDATORY: Flush page cache before EVERY launch
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Launch (identical on both nodes)
docker run -d --name vllm-qwen35-v2 \
  --gpus all --net=host --ipc=host \
  -e MAX_JOBS=2 \
  -e FLASHINFER_NVCC_THREADS=1 \
  -v ~/models:/models \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.triton:/root/.cache/triton \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen35-hybrid-v2 \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**GMU selection:**
- `0.90` — FAILS on both nodes (107.82 GiB free < 109.52 needed). Do NOT use.
- `0.85` — validated production ceiling for both Founders and GX10
- `0.80` — **DO NOT USE in production.** Passes `request_memory()` but causes silent EngineCore OOM-kill during CUDA graph capture (30.4 GiB remaining after model load is insufficient for 51-size PIECEWISE graph capture). Useful only as a diagnostic probe.

**IMPORTANT: `--enable-prefix-caching` CRASHES on Qwen3.5** (DeltaNet hybrid attention
incompatibility in vLLM 0.19). Do NOT add this flag despite community examples using it
(they may have a different vLLM build or newer patch).

**MANDATORY env vars (GB10-specific):**
- `MAX_JOBS=2` — throttles parallel nvcc compile jobs during FlashInfer JIT
- `FLASHINFER_NVCC_THREADS=1` — single nvcc thread per kernel compilation

Without these two, FlashInfer JIT spawns ~20-32 parallel `cicc` compiler processes
(each 1-4GB), exhausting unified memory and triggering OOM-killer (`signal 9`).
The upstream README doesn't mention these because it assumes bare-metal launch
(not containerized) where the JIT happens outside Docker. In our containerized
deploy, these are MANDATORY.

**Cache volume mounts (persist compiled kernels across restarts):**
- `~/.cache/vllm` — torch.compile AOT cache
- `~/.cache/flashinfer` — FlashInfer JIT compiled kernels  
- `~/.triton` — Triton JIT kernels

First boot: ~25-45 min (weights + JIT + graph capture). Second boot: ~7-10 min.

**Optional production flags** (add after baseline is confirmed stable):
```bash
  --enable-auto-tool-choice \           # tool calling support
  --tool-call-parser qwen3_coder \      # Qwen3 tool format
  --enable-chunked-prefill \            # better TTFT on long prompts
  --load-format fastsafetensors         # faster checkpoint loading
```

**First boot is slow (~13-25 min):**
- Weight loading: ~10 min (14 shards × ~54s each)
- torch.compile + CUDA graph capture: ~2.5-15 min (depends on cached state)
- FlashInfer JIT kernels: compiled on first run, cached in container

**SSH will become unresponsive during CUDA graph compilation** (all 20 cores
saturated, memory at ceiling). This is NORMAL. Do not hard-reset unless it
exceeds 30 minutes. The compilation caches persist — second boot is ~7 min.

Monitor startup:
```bash
ssh <node> 'docker logs -f vllm-qwen35-v2'
# Wait for: "Application startup complete."
```

### Quick relaunch (base64 for pwsh quoting safety)

If `/tmp/run-upstream.sh` is wiped (reboot), recreate from base64:

```bash
# 0.85 GMU + JIT throttle (validated config):
echo IyEvYmluL2Jhc2gKZG9ja2VyIHJ1biAtZCAtLW5hbWUgdmxsbS1xd2VuMzUtdjIgXAogIC0tZ3B1cyBhbGwgLS1uZXQ9aG9zdCAtLWlwYz1ob3N0IFwKICAtZSBNQVhfSk9CUz0yIFwKICAtZSBGTEFTSElORkVSX05WQ0NfVEhSRUFEUz0xIFwKICAtdiB+L21vZGVsczovbW9kZWxzIFwKICAtdiB+Ly5jYWNoZS92bGxtOi9yb290Ly5jYWNoZS92bGxtIFwKICAtdiB+Ly5jYWNoZS9mbGFzaGluZmVyOi9yb290Ly5jYWNoZS9mbGFzaGluZmVyIFwKICAtdiB+Ly50cml0b246L3Jvb3QvLmNhY2hlL3RyaXRvbiBcCiAgdmxsbS1xd2VuMzUtdjIgXAogIHNlcnZlIC9tb2RlbHMvcXdlbjM1LTEyMmItaHlicmlkLWludDRmcDggXAogIC0tc2VydmVkLW1vZGVsLW5hbWUgcXdlbjM1LWh5YnJpZC12MiBcCiAgLS1wb3J0IDgwMDAgXAogIC0tbWF4LW1vZGVsLWxlbiAyNjIxNDQgXAogIC0tZ3B1LW1lbW9yeS11dGlsaXphdGlvbiAwLjg1IFwKICAtLXJlYXNvbmluZy1wYXJzZXIgcXdlbjMgXAogIC0tYXR0ZW50aW9uLWJhY2tlbmQgRkxBU0hJTkZFUiBcCiAgLS1zcGVjdWxhdGl2ZS1jb25maWcgJ3sibWV0aG9kIjoibXRwIiwibnVtX3NwZWN1bGF0aXZlX3Rva2VucyI6Mn0nCg== \
  | base64 -d > /tmp/run-upstream.sh && chmod +x /tmp/run-upstream.sh

# Deploy:
docker rm -f vllm-qwen35-v2 2>/dev/null
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
bash /tmp/run-upstream.sh
```

## Phase 4-alt: Launch Dual-Node TP=2 Cluster (Historical Reference — Not Available)

> **NOT APPLICABLE:** Nodes are no longer DAC-connected. This section is retained
> for reference only. TP=2 distributed deployment requires direct high-bandwidth
> interconnect between nodes.

> **Not currently in production.** Use this for experiments requiring larger context
> or distributed tensor parallelism. Throughput may be lower than 2x single-node
> due to cross-node NCCL communication overhead.

**Order matters: WORKER (rank 1) FIRST, then HEAD (rank 0) after ~15s.**

### 4a. Launch WORKER on spark2 (rank 1)

```bash
ssh spark2

docker run -d --name vllm-qwen35-v2 \
  --gpus all --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -v ~/models:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.triton:/root/.cache/triton \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen35-hybrid-v2 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank 1 \
  --master-addr SPARK1_IP --master-port 29501 \
  --headless

echo "Worker launched. Waiting 15s for init..."
sleep 15
docker logs --tail 5 vllm-qwen35-v2
```

### 4b. Launch HEAD on spark1 (rank 0)

```bash
ssh spark1

docker run -d --name vllm-qwen35-v2 \
  --gpus all --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -v ~/models:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.triton:/root/.cache/triton \
  vllm-qwen35-v2 \
  serve /models/qwen35-122b-hybrid-int4fp8 \
  --served-model-name qwen35-hybrid-v2 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank 0 \
  --master-addr SPARK1_IP --master-port 29501
```

### 4c. Monitor Startup

Expect ~15-20 min total: model load (~10 min) + torch.compile (~72s) + FlashInfer JIT + CUDA graph warmup.

```bash
# Watch head logs
ssh spark1 'docker logs -f vllm-qwen35-v2'

# In another terminal, watch worker
ssh spark2 'docker logs -f vllm-qwen35-v2'

# Health check (once ready)
curl -s http://SPARK1_IP:8000/health
curl -s http://SPARK1_IP:8000/v1/models | python3 -m json.tool
```

## Phase 5: Verify

```bash
# Health check both nodes
curl -s http://SPARK1_IP:8000/health
curl -s http://SPARK2_IP:8000/health

# Model listing
curl -s http://SPARK1_IP:8000/v1/models | python3 -m json.tool
curl -s http://SPARK2_IP:8000/v1/models | python3 -m json.tool

# Quick smoke test (each node)
curl -s http://SPARK1_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-hybrid-v2",
    "messages": [{"role": "user", "content": "What is 2+2? Be concise."}],
    "max_tokens": 64
  }' | python3 -m json.tool

curl -s http://SPARK2_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-hybrid-v2",
    "messages": [{"role": "user", "content": "What is 2+2? Be concise."}],
    "max_tokens": 64
  }' | python3 -m json.tool
```

## Phase 6: Update LiteLLM Routing

Both nodes should be in the LiteLLM LB pool for fast/reasoning/code. Config on LITELLM_HOST:

```yaml
# In LITELLM_CONFIG_PATH (e.g. /home/USER/litellm/config.yaml)
# spark1 entries
- model_name: fast
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK1_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark1-qwen35-122b-fast

- model_name: reasoning
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK1_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark1-qwen35-122b-reasoning

- model_name: code
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK1_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark1-qwen35-122b-code

# spark2 entries (identical, different api_base)
- model_name: fast
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK2_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark2-qwen35-122b-fast

- model_name: reasoning
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK2_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark2-qwen35-122b-reasoning

- model_name: code
  litellm_params:
    model: openai/qwen35-hybrid-v2
    api_base: http://SPARK2_IP:8000/v1
    api_key: "dummy"
  model_info:
    id: spark2-qwen35-122b-code
```

After updating, restart LiteLLM and verify routing:
```bash
ssh LITELLM_HOST 'cd LITELLM_COMPOSE_DIR && docker compose restart litellm'

# Test through LiteLLM gateway
curl -s http://LITELLM_HOST_IP:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32}'
```

## Phase 7: Benchmark

```bash
# Bench script is already present on both nodes from deployment:
# ~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/bench_qwen35.sh

# Run benchmark (pause LiteLLM first — Operational Lesson #6)
ssh spark2 'cd ~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 && bash bench_qwen35.sh spark2-isolated'
ssh spark1 'cd ~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 && bash bench_qwen35.sh spark1-isolated'
```

Expected results per node (single-stream, LiteLLM paused, warm-cache run 2+):
- Q&A 256: ~52-54 tok/s
- Code 512: ~53-55 tok/s
- JSON 1024: ~52-54 tok/s
- Math 64: ~48-50 tok/s
- LongCode 2048: ~55-57 tok/s
- Mean: ~53-54 tok/s

## Phase 7.5: Validated Deployment Findings (2026-06-16)

### Performance Results

| Configuration | tok/s | Context | GMU | Notes |
|---|---|---|---|---|
| Vanilla NVFP4 (pre-deploy, prior image) | ~17 | 32K | 0.85 | Old baseline — FP4 kernels broken on SM121 |
| Hybrid INT4+FP8 + MTP-2, throttled JIT only | ~26.8 | 32K | 0.80 | First MTP-2 success (low GMU) |
| Production config, spark1 pre-v2 image | ~47.5 | 262K | 0.85 | Validated on spark1, isolated bench (2026-06-15) |
| spark2 with PD firmware wedge (unfixed) | ~34 | 262K | 0.85 | ASUS GX10 stuck at 702 MHz / 10W — see Known Firmware Issues |
| **spark1 post-v2 image (current)** | **~52** | **262K** | **0.85** | **albond harness, single-stream warm-cache, 2026-06-16** |
| **spark2 post-wedge fix (current)** | **~54** | **262K** | **0.85** | **Cold-drain cleared PD wedge; now exceeds Founders node** |
| Upstream ceiling (albond README) | ~51.6 | 262K | 0.90 | albond's reference — Founders Edition, 0.90 GMU |
| TP=2 dual-node distributed (not available) | ~56 | 262K+ | 0.85 | Requires direct node interconnect — nodes not currently connected |

### Post-Wedge Canonical Bench (2026-06-16, albond harness, LiteLLM paused, warm-cache)

| Test | spark2 (post-wedge) | spark1 | albond ref (GX10, upstream) |
|---|---|---|---|
| Q&A 256 | **53.6** | 52.0 | 51.3 |
| Code 512 | **55.4** | 53.8 | 52.8 |
| JSON 1024 | **53.9** | 53.5 | 51.1 |
| Math 64 | **50.0** | 48.4 | 47.8 |
| LongCode 2048 | **57.0** | 55.5 | 54.9 |
| **Mean** | **53.3** | **52.0** | **51.6** |

Both nodes now match or exceed albond's published upstream ceiling. The prior 34 vs 52 gap was entirely due to the ASUS GX10 PD firmware wedge pinning the GPU at 702 MHz / 10W (see Known Firmware Issues section). After 10-minute cold drain, spark2 recovered to full clock state and outperforms spark1 by ~1.3 tok/s mean.

### MTP-2 vs MTP-3 Isolated Comparison (2026-06-16)

Re-tested `--speculative-config` token counts after observing that prior comparisons were contaminated by concurrent LiteLLM traffic. Both nodes were drained from the LiteLLM `fast`/`reasoning`/`code` pools for the duration of each measurement (verified zero in-flight requests via `vllm:num_requests_running == 0.0`), then immediately restored.

Bench harness: 4 phases per (node × config) = 16 total measurements:

1. **single-stream probe** — 5 fixed prompts × 256 max_tokens, sequential
2. **loaded probe** — 8 parallel requests × 128 max_tokens (aggregate tok/s under load)
3. **NVIDIA tool-eval-bench `--short` sequential** — 15 scenarios, single client
4. **NVIDIA tool-eval-bench `--short --parallel 4`** — 15 scenarios, 4 concurrent

| Phase | spark1 MTP-2 | spark1 MTP-3 | spark2 MTP-2 | spark2 MTP-3 |
|---|---|---|---|---|
| single-stream tok/s | 43.12 | 45.08 | **49.65** | 45.31 |
| loaded N=8 aggregate tok/s | 54.11 | 58.64 | **79.91** | 60.13 |
| `--short` sequential score | 73/100 (11p/4f) 84.9s | 60/100 (9p/6f) 73.5s | 80/100 (12p/3f) 88.0s | 87/100 (13p/2f) 84.4s |
| `--short --parallel 4` score | 73/100 (11p/4f) **59.2s** | 87/100 (13p/2f) 72.6s | 80/100 (12p/3f) 71.7s | 80/100 (12p/3f) 67.8s |

**Verdict: MTP-2 chosen on both nodes.** spark2 was the decisive node — MTP-3 lost 9.6% single-stream and **32.9%** on loaded N=8 throughput. spark1 showed mixed throughput (MTP-3 nominally higher but loaded delta was only +8% and well within run variance). Quality scores via tool-eval-bench `--short` ran 60-87/100 across same-config repeats — too noisy at N=15 scenarios to differentiate the configs.

The earlier "40% MTP-2 regression without INT8 LM Head" claim documented in the optimization stack table was a contaminated measurement (made while the node was serving LiteLLM traffic). Isolated MTP-2 throughput is solidly competitive at 43-50 tok/s single-stream and 54-80 tok/s aggregate at N=8.

Toggle script (both nodes): `python3 ~/toggle_mtp.py {2|3}` patches `docker-compose.yml` and re-validates. Recreate the container with `cd ~/vllm && docker compose up -d --force-recreate`. Allow ~10 minutes for cold boot (spec-decode artifact rebuild dominates).

Raw result JSONs: see `eqi12:/tmp/bench_results_{spark1,spark2}-mtp{2,3}.json` (operator's eqi12 ingest node).

### Community Env Var A/B Testing (2026-06-15)

All tested on spark1 in isolation (LiteLLM paused). Baseline: 47.5 tok/s (pre-v2 image; current v2 image is ~52 tok/s).

| Env Var | Result | Verdict |
|---|---|---|
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | 46.2 tok/s (required dropping to 0.84 GMU to boot) | **Skip** — no gain, memory-tight |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **OOM CRASH** at 0.85 GMU (exit 255) | **Harmful** — changes allocator breaks GB10 memory ceiling |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | 47.3 tok/s (within noise) | **Skip** — no measurable difference |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | Cancelled (triggers recompilation for zero gain) | **Skip** |
| All 4 combined (0.80 GMU) | Not completed — OOM during CUDA graph compile | **Harmful at any useful GMU** |

**Conclusion:** Community env vars provide **zero benefit** on GB10 with this workload.
The documented upstream config (no extra env vars) is optimal. Do not add these.

### TP=1 vs TP=2 Analysis

| Metric | TP=2 Distributed | Two Independent TP=1 (current) |
|---|---|---|
| Per-request speed | ~56 tok/s | ~52-54 tok/s (current v2) |
| Aggregate throughput (2 concurrent) | ~56 tok/s (1 instance) | ~104-108 tok/s |
| KV cache | ~54 GiB (doubled) | ~27 GiB per node |
| Cross-node latency | ~1-2ms per layer (NCCL/TCP) | None |
| Operational complexity | Ray/NCCL, worker-first ordering | Simple independent processes |
| Failure mode | Both nodes down = full outage | Graceful degradation |

**Decision (2026-06-15):** TP=1 independent — +9 tok/s not worth the complexity for agentic workloads
that benefit more from concurrent capacity than single-stream latency.

### Image Verification (CORRECTED)

**INT8 LM Head v2 IS present in our image.** Earlier claim of "Gap 1" was wrong.

```bash
# Verified:
docker run --rm --entrypoint bash vllm-qwen35-v2:latest \
  -c 'grep -q DGX_SPARK_INT8_LMHEAD_V2 /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py && echo FOUND'
# → FOUND

# Image built 2026-06-14, includes:
# - patches/01-hybrid-int4-fp8/inc.py (INC quantization)
# - patches/03-int8-lm-head/patch_int8_lmhead.py (INT8 LM Head v2)
# - vllm-ref 2a69949bd (0.19.1.dev0+g2a69949bd.d20260616, pinned 2026-06-18), transformers 5.10.2, NCCL 2.30.4, Triton 3.7.0
# - SM121 CUDA arch (12.1a)
```

### ✅ Performance Gap Closed (2026-06-16)

Both nodes now at or above the upstream ceiling (~51.6 tok/s albond reference) at 0.85 GMU.
spark1: ~52 tok/s, spark2: ~53 tok/s. The prior gap (spark2 at 34 tok/s) was entirely due to
the PD firmware wedge — see Known Firmware Issues. The notes below describe the prior gap analysis
and are retained for diagnostic reference.

**Prior gap analysis (pre-wedge-fix):**
The 3.5 tok/s gap between 47.5 tok/s production and 51 tok/s upstream was from GMU:
- **Production: 0.85 GMU** — safe headroom for system memory variability
- **Upstream: 0.90 GMU** — bare-metal, assumes zero system overhead

This is an acceptable trade-off. The 0.85 GMU config boots reliably without OOM risk.
Raising to 0.90 gains ~3.5 tok/s but introduces startup fragility (OOM if any system
process allocates unexpectedly during model load).

**Community env vars DO NOT help** — tested exhaustively (see A/B testing above).
The gap is purely memory budget, not kernel efficiency.

### Critical Operational Lessons

1. **`MAX_JOBS=2` + `FLASHINFER_NVCC_THREADS=1` are MANDATORY** — not optional.
   FlashInfer JIT spawns ~20-32 parallel `cicc` processes (each 1-4GB). Without
   throttling, they exhaust 128GB unified memory → `signal 9` (OOM-kill).

2. **Page cache flush before EVERY launch** — `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.
   Without this, prior crashed runs leave bloated page cache at the memory ceiling.

3. **Phase 0 (disable desktop services) is REQUIRED for 0.90 GMU** — gdm3/gnome-shell
   hold ~16GB GPU memory on GB10. With desktop: only 105 GiB free (need 109.46 for 0.90).
   Without desktop: 118+ GiB free.

4. **Cache volume mounts save 30+ min per restart:**
   - `~/.cache/vllm` — torch.compile AOT artifacts
   - `~/.cache/flashinfer` — FlashInfer JIT compiled kernels
   - `~/.triton` — Triton JIT kernels
   
5. **Hard reset > waiting** when SSH locks up for >10 min. Nodes in OOM-kill loops
   don't recover gracefully. Reset gives clean kernel, empty page cache, fresh GPU.
   JIT cache volumes survive reboot.

6. **LiteLLM traffic impacts benchmarks.** Always pause LiteLLM routing to spark
   backends before running isolated benchmarks:
   ```bash
   # On LITELLM_HOST: comment out spark api_base lines, restart litellm
    sed -i 's|api_base: http://172.16.200|# PAUSED api_base: http://172.16.200|g' \
      LITELLM_CONFIG_PATH
    cd LITELLM_COMPOSE_DIR && docker compose restart litellm
   # After bench: remove "# PAUSED " prefix and restart again
   ```

7. **PowerShell mangles JSON in ssh commands.** Always use base64 encoding for
   launch scripts that contain JSON (e.g., `--speculative-config`).

8. **Backup JIT caches after successful first boot:**
   ```bash
   tar czf ~/backups/jit-cache-$(date +%Y%m%d).tar.gz \
     ~/.cache/vllm ~/.cache/flashinfer ~/.triton
   ```
   Restoring these skips the 30-45 min compilation entirely on fresh deploys.

9. **Pin `VLLM_REF` to `2a69949bd` — do not track `main`.** vLLM nightlies after this commit have a stricter `request_memory()` check that fails at 0.85 GMU on GB10 because the CUDA driver permanently reserves ~22 GiB (leaving ~99 GiB free, below the 103.44 GiB that 0.85 × 121.69 requires). The working image is `0.19.1.dev0+g2a69949bd.d20260616.cu132`. Pin in both `Dockerfile` (`ARG VLLM_REF=2a69949bd`) and `build-and-copy.sh` (`VLLM_REF="2a69949bd"`). Do NOT update without re-testing the memory check passes.

10. **Use `--gpu-memory-utilization 0.85` on clean boot.** 0.85 × 121.69 = 103.44 GiB — the right production value. After model load (66.93 GiB), 36.5 GiB remains for KV cache + CUDA graph capture (~512-token batch, 51 graph sizes), which fits. Do NOT use 0.80 in production: it leaves only 30.4 GiB after model load, insufficient for CUDA graph capture → silent EngineCore OOM-kill. 0.80 can be used as a diagnostic probe (it passes `request_memory()` in degraded state but reveals CUDA graph failure as the next blocker). 0.90 fails on both nodes. nvidia-smi does not report memory on GB10 — read free/total from vLLM startup logs.

11. **If vLLM fails to start more than once, reboot before retrying.** Each failed startup attempt leaves CUDA context residue that `docker rm` alone cannot clear. The residue accumulates: first failure ~22 GiB driver overhead, after 2-3 crash-loop cycles ~24-26 GiB consumed, eventually nothing fits. Only a clean OS reboot fully resets the GPU memory allocator to baseline. Diagnosis: if `docker logs` shows `num_gpu_blocks=0 with num_gpu_blocks_override=512` followed by a silent EngineCore death (no Python traceback, container restarts), the node is in residue-degraded state. Fix:
    ```bash
    ssh <node> 'sudo reboot'
    # wait ~2 min for SSH to return
    ssh <node> 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
    ssh <node> 'docker compose -f /home/USER/vllm/docker-compose.yml up -d'
    ```
    After clean reboot the container should load without hitting the `num_gpu_blocks=0` override.

## Rollback

```bash
# Stop v2 containers on both nodes
ssh spark1 'docker stop vllm-qwen35-v2 && docker rm vllm-qwen35-v2'
ssh spark2 'docker stop vllm-qwen35-v2 && docker rm vllm-qwen35-v2'

# Restart original Qwen3.5-122B NvFP4 (if old model/image still available)
# spark1: old image vllm/vllm-openai:cu130-nightly, model at ~/models/qwen35-122b-nvfp4
# spark2: same image, model at ~/models/Qwen3.5-122B-A10B-NVFP4

# Revert LiteLLM: change openai/qwen35-hybrid-v2 → openai/spark-v1 in config
```

## Key Constraints

- **`MAX_JOBS=2` and `FLASHINFER_NVCC_THREADS=1` are MANDATORY** — without them, FlashInfer JIT OOM-kills the node
- **Page cache flush before EVERY launch** — `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
- **Phase 0 (disable desktop services) is REQUIRED for 0.90 GMU** — gdm3 holds ~16GB GPU memory
- **GMU 0.85 is the validated safe value** for containerized deploy. 0.90 may work after Phase 0 reboot.
- **Max-model-len 262144** — validated working at 0.85 GMU with JIT throttle + Phase 0
- **FlashInfer cache volumes** must be mounted to persist JIT kernels across restarts (~30 min savings)
- **INT8 LM Head v2 IS present** in current `vllm-qwen35-v2:latest` image (verified 2026-06-15)
- **Image was correctly built via install.sh** — includes PR #38325 swapAB FP8 optimization

## INC Fix (SM121 Specific)

The `vllm-qwen35-v2:latest` image contains a patched `inc.py` that adapts the hybrid INT4+FP8 quantization config for SM121's module layout:

- SM121 vLLM has `auto_gptq.py` but NOT `gptq.py` or `gptq_marlin.py`
- The patch aliases AutoGPTQ classes to the expected GPTQMarlin/GPTQ names
- Also adds `hf_config=None` to two method signatures that receive it from newer HuggingFace

If rebuilding the image, the authoritative upstream inc.py (`~/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/patches/01-hybrid-int4-fp8/inc.py`) targets standard vLLM and must be adapted with these 4 changes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cicc died due to signal 9 (Kill signal)` during FlashInfer JIT | **Add `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` env vars.** System DRAM exhausted by parallel nvcc. See Phase 7.5. |
| Node SSH-unreachable for >10 min | **Hard-reset the node, don't wait.** See Critical Operational Lessons #5. Empirically faster than trying to recover gracefully. |
| Repeated OOM after retry on same node | Memory fragmentation accumulates. **Hard-reset**, then relaunch with throttled config. |
| Health returns nothing after 25 min (cold) / 10 min (warm) | Check `docker logs` for OOM. Lower GMU to 0.78. |
| INCConfig `hf_config` error | Image not patched. Rebuild with SM121-adapted inc.py (see INC Fix section) |
| `gptq_marlin` / `gptq` ModuleNotFoundError | SM121 build only has `auto_gptq.py`. Rebuild image with aliased imports. |
| NCCL timeout on worker connect (TP=2 only) | Verify master-addr is correct (SPARK1_IP), check firewall for port 29501 |
| Garbage output | Ensure patched image (vllm-qwen35-v2), not vanilla vLLM |
| Engine init failed, no completion tokens | Container exited during warmup. Check `docker logs` for OOM-killer signal 9. Drop GMU. |
| `/tmp/run-mtp2.sh` not found | Wiped on reboot. Recreate from base64 in Phase 4 "Quick relaunch" section. |
| Throughput stuck at ~26 tok/s, expected ~47 | Missing MTP-2 (`--speculative-config`), or JIT cache not populated (first boot is slower). Run bench after warmup. |
| Throughput stuck at ~29-34 tok/s on spark2, spark1 gets ~47-52 | **First check:** PD firmware wedge (see Known Firmware Issues). Run `nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader` during load — if <1000 MHz, cold-drain is required. **Second check:** stale torch_compile cache — wipe `~/.cache/vllm` with sudo, relaunch. |
| Throughput stuck at ~36 tok/s, expected ~51 | GMU 0.80 vs 0.90 gap — this is expected. See Performance Gap section. |
| Two nodes show very different memory usage on htop | Older node likely has fragmentation/page cache from prior crashes. **Hard-reset the worse node** to bring them in line. |
| `weight_scale_inv not found` | Patch 01 not applied — rebuild Docker image |
| `ValueError: Free memory on device cuda:0 (X/121.69 GiB) less than desired GPU memory utilization (0.85, 103.44 GiB)` | GB10 CUDA driver permanently reserves ~22 GiB. 0.85 × 121.69 = 103.44 GiB fails if crash-loop residue has accumulated. vLLM nightly builds after `2a69949bd` have stricter `request_memory()` that enforces this at startup. Fix: (1) stop + rm container; (2) ensure compose uses `--gpu-memory-utilization 0.80`; (3) restart. If free GPU memory is still < 97 GiB after restart, reboot the node to clear CUDA context residue. nvidia-smi always reports memory N/A on GB10 — read free/total from vLLM's own startup log line. |
| OOM during FlashInfer JIT | Lower GMU to 0.80, let compilation finish, raise back after cache populated |
| Only ~17 tok/s (half expected) | Missing `--attention-backend FLASHINFER` flag |
| Stale Triton cache | `docker exec vllm-qwen35-v2 rm -rf /root/.cache/triton` and restart |
| nvidia_gpu_exporter 0 metrics | Recreate with `--privileged --gpus all` flags |
| nvidia_gpu_exporter panic (invalid metric name) | GB10 has non-standard `clocks_event_reasons` fields. Add `--query-field-names=timestamp,name,uuid,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,power.limit,clocks.current.sm,clocks.current.memory,fan.speed,pstate` |
| JIT cache permission denied on wipe | Files owned by root (created inside container). Use `sudo rm -rf ~/.cache/vllm ~/.cache/flashinfer ~/.triton` |
