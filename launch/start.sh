#!/usr/bin/env bash
set -euo pipefail
set -x

# ----------------------------------------------------------------------------- #
# Configurable options (all overridable via environment variables)
# ----------------------------------------------------------------------------- #
AUTO_TRAIN=${AUTO_TRAIN:-1}                  # 1=auto-submit training; 0=only start the cluster and keep alive (for debugging)
RUN_SCRIPT=${RUN_SCRIPT:-run_math_sdpo.sh}   # training script filename
TRAIN_RECIPE_SUBDIR=${TRAIN_RECIPE_SUBDIR:-sdpo} # recipe subdirectory under STABLE_RL_DIR
if [ -z "${NNODES:-}" ]; then
    if [ -n "${HOST_NUM:-}" ]; then
        NNODES="${HOST_NUM}"
    elif [ -n "${NODE_IP_LIST:-}" ]; then
        NNODES=0
        for item in ${NODE_IP_LIST//,/ }; do
            [ -n "${item}" ] && NNODES=$((NNODES + 1))
        done
        [ "${NNODES}" -gt 0 ] || NNODES=6
    else
        NNODES=6
    fi
fi
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-${HOST_GPU_NUM:-8}} # GPUs per node
RAY_READY_TIMEOUT=${RAY_READY_TIMEOUT:-1800} # timeout (seconds) for waiting the ray cluster to become ready
CEPH_READY_TIMEOUT=${CEPH_READY_TIMEOUT:-300} # timeout (seconds) for waiting the ceph mount to become ready
MPI_HOSTFILE=${MPI_HOSTFILE:-/etc/taiji/hostfile}
START_RAY_WORKERS_WITH_MPIRUN=${START_RAY_WORKERS_WITH_MPIRUN:-0}
ulimit -n 65536 || true

# Preserve the previously stable NCCL path: use socket/TCP instead of forcing RoCE/RDMA.
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export VLLM_USE_V1=1
export SDPO_TOOLCHAIN_DIR=/apdcephfs_gy4/share_303378103/user/audenhuang/FlowSD/recipe/sdpo/toolchain
export PATH=${SDPO_TOOLCHAIN_DIR}:${PATH}
export CC=${SDPO_TOOLCHAIN_DIR}/gcc
export CXX=${SDPO_TOOLCHAIN_DIR}/g++
export TRITON_CC=${SDPO_TOOLCHAIN_DIR}/gcc
export CUDAHOSTCXX=${SDPO_TOOLCHAIN_DIR}/g++
export LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LIBRARY_PATH:+:${LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export COMPILER_PATH=/usr/libexec/gcc/x86_64-TencentOS-linux/12${COMPILER_PATH:+:${COMPILER_PATH}}
export LDFLAGS="-L/usr/lib/gcc/x86_64-TencentOS-linux/12 -L/usr/lib64${LDFLAGS:+ ${LDFLAGS}}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

fix_lgcc_linker_path() {
    local gcc_libdir=/usr/lib/gcc/x86_64-TencentOS-linux/12
    local system_libdir=/usr/lib64
    if [ ! -f "${gcc_libdir}/libgcc.a" ]; then
        echo "[start.sh] WARN: libgcc.a not found under ${gcc_libdir}; skip linker hotfix."
        return 0
    fi
    ln -sf "${gcc_libdir}/libgcc.a" "${system_libdir}/libgcc.a" || true
    ln -sf "${gcc_libdir}/libgcc_eh.a" "${system_libdir}/libgcc_eh.a" || true
    ln -sf "${gcc_libdir}/libgcc_s.so" "${system_libdir}/libgcc_s.so" || true
    if ld -shared -o "/tmp/sdpo_lgcc_test_$$.so" -lgcc >/dev/null 2>&1; then
        rm -f "/tmp/sdpo_lgcc_test_$$.so"
        echo "[start.sh] linker hotfix OK: ld can resolve -lgcc from ${system_libdir}."
    else
        echo "[start.sh] WARN: linker hotfix failed: ld still cannot resolve -lgcc."
    fi
}
fix_lgcc_linker_path

# Directory of this script (= platform-uploaded lightweight launch dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Scheme 1: only start.sh is uploaded; code/data/model are read from the mounted ceph path.
CEPH_ROOT="${CEPH_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang}"

# Load W&B credentials from a private file without exposing the API key in xtrace.
WANDB_ENV_FILE=${WANDB_ENV_FILE:-${CEPH_ROOT}/.secrets/wandb.env}
# Explicit task-level settings take precedence over defaults in the credential file.
_requested_wandb_mode=${WANDB_MODE:-}
_requested_wandb_project=${WANDB_PROJECT:-}
_requested_wandb_entity=${WANDB_ENTITY:-}
_requested_wandb_dir=${WANDB_DIR:-}
if [ -f "${WANDB_ENV_FILE}" ]; then
    set +x
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
    set -x
    echo "[start.sh] loaded W&B credentials from ${WANDB_ENV_FILE} (key redacted)"
else
    echo "[start.sh] WARN: W&B credential file not found: ${WANDB_ENV_FILE}"
fi
[ -z "${_requested_wandb_mode}" ] || WANDB_MODE=${_requested_wandb_mode}
[ -z "${_requested_wandb_project}" ] || WANDB_PROJECT=${_requested_wandb_project}
[ -z "${_requested_wandb_entity}" ] || WANDB_ENTITY=${_requested_wandb_entity}
[ -z "${_requested_wandb_dir}" ] || WANDB_DIR=${_requested_wandb_dir}
export WANDB_API_KEY
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-verl-grpo}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_DIR=${WANDB_DIR:-${CEPH_ROOT}/output/wandb}
unset _requested_wandb_mode _requested_wandb_project _requested_wandb_entity _requested_wandb_dir

wait_for_ceph() {
    local waited=0
    echo "[start.sh] waiting for ceph path: ${CEPH_ROOT}/FlowSD (timeout=${CEPH_READY_TIMEOUT}s)"
    while [ "${waited}" -lt "${CEPH_READY_TIMEOUT}" ]; do
        if [ -d "${CEPH_ROOT}/FlowSD" ]; then
            echo "[start.sh] ceph ready: ${CEPH_ROOT}"
            return 0
        fi
        echo "[start.sh] ceph not ready yet: ${CEPH_ROOT}/FlowSD (${waited}s/${CEPH_READY_TIMEOUT}s)"
        echo "[start.sh] visible ceph mounts:"
        mount | grep -Ei 'ceph|apdcephfs|fuse|300719894' || true
        echo "[start.sh] visible candidate dirs:"
        ls -la /apdcephfs 2>/dev/null || true
        ls -la /apdcephfs/share_300719894/user 2>/dev/null || true
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

if ! wait_for_ceph; then
    echo "[start.sh] ERROR: ceph is still unavailable after waiting: ${CEPH_ROOT}/FlowSD"
    echo "[start.sh] SCRIPT_DIR=${SCRIPT_DIR} CEPH_ROOT=${CEPH_ROOT}"
    echo "[start.sh] mount debug:"
    mount | grep -Ei 'ceph|apdcephfs|fuse|300719894' || true
    echo "[start.sh] df debug:"
    df -h | grep -Ei 'ceph|apdcephfs|300719894' || true
    sleep infinity
fi

WORK_DIR="${CEPH_ROOT}"
STABLE_RL_DIR="${STABLE_RL_DIR:-${WORK_DIR}/FlowSD}"
cd "${WORK_DIR}"

# ----------------------------------------------------------------------------- #
# Data / model paths (used for preflight checks; same defaults as the run scripts, and exported downstream)
# All defaults live on the ceph shared disk /apdcephfs/share_300719894/... (the container's only mounted disk)
# ----------------------------------------------------------------------------- #
export DATA_ROOT=${DATA_ROOT:-"${CEPH_ROOT}/data"}
export MODEL_ROOT=${MODEL_ROOT:-"${CEPH_ROOT}/models"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${CEPH_ROOT}/output"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/rl/train.parquet"}
export TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/rl/aime-2024.parquet"}
export MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B"}

# ----------------------------------------------------------------------------- #
# GPU monitor (enabled on both master/worker, split into per-hostname files)
# ----------------------------------------------------------------------------- #
LOG_DIR="${WORK_DIR}/gpu_logs"
mkdir -p "${LOG_DIR}"
GPU_MONITOR_PID=""

if [ -f "${WORK_DIR}/gpu_monitor.py" ]; then
    GPU_LOG="${LOG_DIR}/gpu_$(hostname)_$(date +%Y%m%d_%H%M%S).csv"
    echo "[start.sh] launching GPU monitor -> ${GPU_LOG}"
    python3 "${WORK_DIR}/gpu_monitor.py" --log "${GPU_LOG}" --no-clear -i 10 \
        > "${LOG_DIR}/gpu_monitor_$(hostname).log" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "[start.sh] GPU monitor PID=${GPU_MONITOR_PID}"
else
    echo "[start.sh] WARN: gpu_monitor.py not found under WORK_DIR=${WORK_DIR}; skip GPU monitor."
fi

# Clean up the monitor process on exit
cleanup() {
    if [ -n "${GPU_MONITOR_PID}" ]; then
        echo "[start.sh] cleaning up GPU monitor (PID=${GPU_MONITOR_PID})"
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ----------------------------------------------------------------------------- #
# master / worker detection
#   NODE_IP_LIST looks like "ip:xx,ip:xx,...", the first IP is the master
#   master starts Ray head locally and bootstraps workers via mpirun when available.
# ----------------------------------------------------------------------------- #
NODE_IP_LIST="${NODE_IP_LIST:-}"
RAY_PORT=${RAY_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
LOCAL_IPS="$(hostname -I 2>/dev/null) $(hostname -i 2>/dev/null)"

get_local_node_ip() {
    if [ -n "${NODE_IP_LIST}" ]; then
        local item ip
        for item in ${NODE_IP_LIST//,/ }; do
            ip="${item%%:*}"
            if echo " ${LOCAL_IPS} " | grep -Fqw "${ip}"; then
                echo "${ip}"
                return 0
            fi
        done
    fi
    hostname -I 2>/dev/null | awk '{print $1}'
}

LOCAL_NODE_IP="$(get_local_node_ip)"
if [ -n "${NODE_IP_LIST}" ]; then
    MASTER_IP=$(echo "${NODE_IP_LIST}" | cut -d',' -f1 | cut -d':' -f1)
else
    MASTER_IP="${LOCAL_NODE_IP}"
fi

IS_MASTER=0
if [ -z "${NODE_IP_LIST}" ]; then
    IS_MASTER=1                                  # single node / no NODE_IP_LIST: treat as master
elif echo " ${LOCAL_IPS} " | grep -Fqw "${MASTER_IP}"; then
    IS_MASTER=1
fi
echo "[start.sh] NODE_IP_LIST=${NODE_IP_LIST} MASTER_IP=${MASTER_IP} LOCAL_NODE_IP=${LOCAL_NODE_IP} LOCAL_IPS=[${LOCAL_IPS}] IS_MASTER=${IS_MASTER}"

# ----------------------------------------------------------------------------- #
# worker: keep alive by default; fallback can self-join when mpirun bootstrap is disabled.
# ----------------------------------------------------------------------------- #
wait_for_master_port() {
    python3 - "${MASTER_IP}" "${RAY_PORT}" "${RAY_READY_TIMEOUT}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"[start.sh] master ray port ready: {host}:{port}")
            sys.exit(0)
    except OSError as exc:
        print(f"[start.sh] waiting master ray port {host}:{port}: {exc}", flush=True)
        time.sleep(5)
print(f"[start.sh] ERROR: timeout waiting master ray port {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

node_ip_count() {
    if [ -z "${NODE_IP_LIST}" ]; then
        echo 1
        return 0
    fi
    local count=0 item
    for item in ${NODE_IP_LIST//,/ }; do
        [ -n "${item}" ] && count=$((count + 1))
    done
    echo "${count:-1}"
}

start_ray_workers_via_mpirun() {
    local target_nodes
    target_nodes="$(node_ip_count)"
    if [ "${target_nodes}" -le 1 ]; then
        echo "[start.sh] single-node run; skip mpirun worker bootstrap."
        return 0
    fi
    if [ "${START_RAY_WORKERS_WITH_MPIRUN}" != "1" ]; then
        echo "[start.sh] START_RAY_WORKERS_WITH_MPIRUN=${START_RAY_WORKERS_WITH_MPIRUN}; skip mpirun worker bootstrap."
        return 0
    fi
    if ! command -v mpirun >/dev/null 2>&1; then
        echo "[start.sh] WARN: mpirun not found; rely on platform to execute start.sh on worker pods."
        return 1
    fi
    if [ ! -f "${MPI_HOSTFILE}" ]; then
        echo "[start.sh] WARN: MPI hostfile not found: ${MPI_HOSTFILE}; rely on platform worker start_cmd."
        return 1
    fi

    echo "[start.sh] bootstrapping Ray workers via mpirun: hostfile=${MPI_HOSTFILE}, np=${target_nodes}, master=${MASTER_IP}:${RAY_PORT}"
    set +e
    NODE_IP_LIST="${NODE_IP_LIST}" MASTER_IP="${MASTER_IP}" RAY_PORT="${RAY_PORT}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" RAY_READY_TIMEOUT="${RAY_READY_TIMEOUT}" \
    mpirun --allow-run-as-root --hostfile "${MPI_HOSTFILE}" --map-by ppr:1:node -np "${target_nodes}" \
        -x NODE_IP_LIST -x MASTER_IP -x RAY_PORT -x N_GPUS_PER_NODE -x RAY_READY_TIMEOUT \
        bash -lc '
set -euo pipefail
local_ips="$(hostname -I 2>/dev/null || true) $(hostname -i 2>/dev/null || true) ${LOCAL_IP:-} ${NODE_IP:-}"
rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-}}}"
IFS="," read -ra node_items <<< "${NODE_IP_LIST}"
local_node_ip=""
match_source="local-ip-match"
for item in "${node_items[@]}"; do
    ip="${item%%:*}"
    if echo " ${local_ips} " | grep -Fqw "${ip}"; then
        local_node_ip="${ip}"
        break
    fi
done
if [ -z "${local_node_ip}" ] && [ -n "${rank}" ] && [ "${rank}" -ge 0 ] 2>/dev/null && [ "${rank}" -lt "${#node_items[@]}" ]; then
    local_node_ip="${node_items[$rank]%%:*}"
    match_source="mpi-rank-${rank}"
fi
if [ -z "${local_node_ip}" ]; then
    local_node_ip="${LOCAL_IP:-${NODE_IP:-}}"
    match_source="env-fallback"
fi
echo "[start.sh/mpirun] host=$(hostname) rank=${rank:-NA} local_ips=[${local_ips}] selected_ip=${local_node_ip:-NA} source=${match_source} master=${MASTER_IP}"
if [ -z "${local_node_ip}" ]; then
    echo "[start.sh/mpirun] ERROR: cannot determine local node ip" >&2
    exit 2
fi
if [ "${local_node_ip}" = "${MASTER_IP}" ] || echo " ${local_ips} " | grep -Fqw "${MASTER_IP}"; then
    echo "[start.sh/mpirun] skip head node ${local_node_ip}"
    exit 0
fi
python3 - "${MASTER_IP}" "${RAY_PORT}" "${RAY_READY_TIMEOUT}" <<'"'"'PY'"'"'
import socket
import sys
import time
host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"[start.sh/mpirun] master ray port ready: {host}:{port}", flush=True)
            sys.exit(0)
    except OSError as exc:
        print(f"[start.sh/mpirun] waiting master ray port {host}:{port}: {exc}", flush=True)
        time.sleep(5)
print(f"[start.sh/mpirun] ERROR: timeout waiting master ray port {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
timeout 30s ray stop --force || true
rm -rf /tmp/ray
nvidia-smi -L || true
ray start --node-ip-address="${local_node_ip}" --address="${MASTER_IP}:${RAY_PORT}" --num-gpus="${N_GPUS_PER_NODE}"
echo "[start.sh/mpirun] worker joined Ray cluster: local=${local_node_ip}, master=${MASTER_IP}:${RAY_PORT}, num_gpus=${N_GPUS_PER_NODE}"
'
    local rc=$?
    set -e
    if [ "${rc}" -ne 0 ]; then
        echo "[start.sh] WARN: mpirun worker bootstrap exited with rc=${rc}; wait_ray_ready will verify cluster state."
        return "${rc}"
    fi
    return 0
}

if [ "${IS_MASTER}" -ne 1 ]; then
    if [ "${START_RAY_WORKERS_WITH_MPIRUN}" = "1" ]; then
        echo "[start.sh] this pod is a WORKER; master will bootstrap Ray here via mpirun. Keep alive to avoid duplicate ray start."
        sleep infinity
    fi
    echo "[start.sh] this pod is a WORKER; starting local Ray worker and joining ${MASTER_IP}:${RAY_PORT}."
    if [ -z "${LOCAL_NODE_IP}" ]; then
        echo "[start.sh] ERROR: cannot determine LOCAL_NODE_IP for worker."
        sleep infinity
    fi
    if ! wait_for_master_port; then
        echo "[start.sh] ERROR: master Ray port not ready; keep alive for debugging."
        sleep infinity
    fi
    timeout 30s ray stop --force || true
    rm -rf /tmp/ray
    echo "[start.sh] worker nvidia-smi summary:"
    nvidia-smi -L || true
    ray start --node-ip-address="${LOCAL_NODE_IP}" --address="${MASTER_IP}:${RAY_PORT}" --num-gpus="${N_GPUS_PER_NODE}"
    echo "[start.sh] worker joined Ray cluster: local=${LOCAL_NODE_IP}, master=${MASTER_IP}:${RAY_PORT}, num_gpus=${N_GPUS_PER_NODE}"
    sleep infinity
fi

# ================================ MASTER below ================================ #

# the run script requires cwd=FlowSD repo root, and runtime_env's working_dir=. is relative to that cwd
if [ ! -d "${STABLE_RL_DIR}" ]; then
    echo "[start.sh] ERROR: STABLE_RL_DIR not found: ${STABLE_RL_DIR}"
    sleep infinity
fi
cd "${STABLE_RL_DIR}"

# ----------------------------------------------------------------------------- #
# Secure WANDB key injection (without polluting the git-tracked runtime_env.yaml)
#   create a local copy .runtime_env.local.yaml (already in .gitignore) and write the real key into it;
#   then point the run script at it via RUNTIME_ENV. The runtime_env.yaml in git always stays a placeholder.
# ----------------------------------------------------------------------------- #
RUNTIME_ENV_SRC="${RUNTIME_ENV_SRC:-${STABLE_RL_DIR}/runtime_env.yaml}"
RUNTIME_ENV_RUN="${STABLE_RL_DIR}/.runtime_env.local.yaml"
if [ ! -f "${RUNTIME_ENV_SRC}" ]; then
    echo "[start.sh] ERROR: runtime env file not found: ${RUNTIME_ENV_SRC}"
    sleep infinity
fi
cp "${RUNTIME_ENV_SRC}" "${RUNTIME_ENV_RUN}"
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_API_KEY}" != "your key" ]; then
    # Read secrets from the environment inside Python so xtrace never prints them.
    set +x
    python3 - "${RUNTIME_ENV_RUN}" <<'PY_WANDB'
import json
import os
import sys
from pathlib import Path

p = Path(sys.argv[1])
text = p.read_text()
keys = ("WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR")
for key in keys:
    value = os.environ.get(key)
    if not value:
        continue
    line = f"  {key}: {json.dumps(value)}"
    lines = text.splitlines()
    for i, existing in enumerate(lines):
        if existing.startswith(f"  {key}:"):
            lines[i] = line
            break
    else:
        lines.append(line)
    text = "\n".join(lines) + "\n"
p.write_text(text)
PY_WANDB
    set -x
    echo "[start.sh] injected W&B online configuration into ${RUNTIME_ENV_RUN} (key redacted)"
else
    echo "[start.sh] WARN: WANDB_API_KEY is unavailable; W&B online logging will fail."
fi
export RUNTIME_ENV="${RUNTIME_ENV_RUN}"

# ----------------------------------------------------------------------------- #
# Start Ray head locally, then actively bootstrap Ray workers via mpirun.
# ----------------------------------------------------------------------------- #
echo "[start.sh] starting Ray head locally on ${MASTER_IP}:${RAY_PORT} ..."
timeout 30s ray stop --force || true
rm -rf /tmp/ray
echo "[start.sh] head nvidia-smi summary:"
nvidia-smi -L || true
ray start --head \
    --node-ip-address="${MASTER_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-gpus="${N_GPUS_PER_NODE}"
echo "[start.sh] Ray head started: ${MASTER_IP}:${RAY_PORT}, dashboard=${MASTER_IP}:${DASHBOARD_PORT}, num_gpus=${N_GPUS_PER_NODE}"
start_ray_workers_via_mpirun || true

# Poll and wait until the cluster nodes are ready
wait_ray_ready() {
    local need=$1 timeout=$2 waited=0 alive=0
    while [ "${waited}" -lt "${timeout}" ]; do
        alive=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive')))" 2>/dev/null || echo 0)
        case "${alive}" in
            ''|*[!0-9]*) alive=0 ;;
        esac
        if [ "${alive}" -ge "${need}" ]; then
            echo "[start.sh] ray cluster ready: ${alive}/${need} nodes"
            return 0
        fi
        echo "[start.sh] waiting ray nodes: ${alive}/${need} ... (${waited}s/${timeout}s)"
        sleep 10
        waited=$((waited + 10))
    done
    echo "[start.sh] WARN: ray not fully ready after ${timeout}s (${alive}/${need}); proceeding anyway."
    return 1
}
if ! wait_ray_ready "${NNODES}" "${RAY_READY_TIMEOUT}"; then
    echo "[start.sh] ERROR: Ray cluster did not become ready; keep alive for debugging."
    sleep infinity
fi

# ----------------------------------------------------------------------------- #
# AUTO_TRAIN disabled: after starting the cluster just keep alive (for manual debugging/submission)
# ----------------------------------------------------------------------------- #
if [ "${AUTO_TRAIN}" != "1" ]; then
    echo "[start.sh] AUTO_TRAIN=${AUTO_TRAIN}; Ray cluster is ready; automatic training submission is disabled. Keeping the job alive."
    sleep infinity
fi

# ----------------------------------------------------------------------------- #
# preflight: if data/model is missing, skip submitting training, fall back to keep-alive with a hint (avoid empty runs)
# ----------------------------------------------------------------------------- #
preflight_ok=1
if [ "${SKIP_TRAIN_PREFLIGHT:-0}" = "1" ]; then
    echo "[start.sh] SKIP_TRAIN_PREFLIGHT=1; skip generic training data/model checks."
else
    for f in "${TRAIN_FILE}" "${TEST_FILE}"; do
        if [ ! -f "${f}" ]; then echo "[start.sh] MISSING data file: ${f}"; preflight_ok=0; fi
    done
    if [ ! -d "${MODEL_PATH}" ]; then echo "[start.sh] MISSING model dir: ${MODEL_PATH}"; preflight_ok=0; fi
fi

if [ "${preflight_ok}" -ne 1 ]; then
    echo "[start.sh] preflight FAILED: data or model assets are unavailable."
    echo "[start.sh] Prepare the required assets first; see recipe/sdpo/data/download_all.sh"
    echo "[start.sh] Ray cluster is ready and will remain alive for manual submission."
    sleep infinity
fi

# ----------------------------------------------------------------------------- #
# Submit training (detached: the run script uses ray job submit --no-wait internally)
# ----------------------------------------------------------------------------- #
echo "[start.sh] submitting training: recipe/${TRAIN_RECIPE_SUBDIR}/${RUN_SCRIPT} (NNODES=${NNODES}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE})"
if ! NNODES="${NNODES}" N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" RUNTIME_ENV="${RUNTIME_ENV}" bash "recipe/${TRAIN_RECIPE_SUBDIR}/${RUN_SCRIPT}"; then
    echo "[start.sh] ERROR: training submission script failed; keep alive to retain ray cluster for debugging."
    sleep infinity
fi

echo "[start.sh] training submitted. keep alive to retain ray cluster & GPU monitor."
sleep infinity
