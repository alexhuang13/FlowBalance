#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536 || true

########################
# CONFIG
########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Worker node IPs. Supported formats:
#   NODE_IP_LIST="ip0,ip1,ip2,ip3"
#   NODE_IP_LIST="ip0:8,ip1:8,ip2:8,ip3:8"
NODE_IP_LIST="${NODE_IP_LIST:-}"
if [[ -z "${NODE_IP_LIST}" ]]; then
  echo "ERROR: NODE_IP_LIST is empty; cannot start a multi-node Ray cluster."
  exit 1
fi

IFS=',' read -ra NODE_ENTRIES <<< "${NODE_IP_LIST}"
WORKERS=()
for entry in "${NODE_ENTRIES[@]}"; do
  ip="${entry%%:*}"
  if [[ -n "${ip}" ]]; then
    WORKERS+=("${ip}")
  fi
done
if (( ${#WORKERS[@]} == 0 )); then
  echo "ERROR: no valid worker IP parsed from NODE_IP_LIST=${NODE_IP_LIST}"
  exit 1
fi

SSH_HOSTS=()
if [[ -n "${NODE_SSH_HOSTS:-}" ]]; then
  IFS=',' read -ra SSH_HOSTS <<< "${NODE_SSH_HOSTS}"
fi

MASTER_IP="${WORKERS[0]}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
RAY_READY_TIMEOUT="${RAY_READY_TIMEOUT:-300}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-${#WORKERS[@]}}"

# Optional in-script training submission.
# Examples:
#   AUTO_TRAIN=1 TRAIN_SCRIPT=recipe/flowsd/run_math_flowsd.sh ... bash scripts/start_ray_cluster.sh
#   TRAIN_SCRIPT=run_math_flowsd.sh ... bash scripts/start_ray_cluster.sh
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${RUN_SCRIPT:-}}"
AUTO_TRAIN="${AUTO_TRAIN:-0}"
if [[ -n "${TRAIN_SCRIPT}" ]]; then
  AUTO_TRAIN=1
fi
KEEP_ALIVE_AFTER_SUBMIT="${KEEP_ALIVE_AFTER_SUBMIT:-1}"

########################
# FUNCTIONS
########################

start_head() {
  echo "==> Starting Ray head on master (${MASTER_IP})"
  ray stop --force || true
  rm -rf /tmp/ray 2>/dev/null || true

  ray start --head \
    --node-ip-address="${MASTER_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-gpus="${N_GPUS_PER_NODE}"
}

start_worker() {
  local worker_ip=$1
  local worker_idx=$2
  local ssh_target="${worker_ip}"
  if (( worker_idx < ${#SSH_HOSTS[@]} )); then
    ssh_target="${SSH_HOSTS[worker_idx]}"
  fi

  echo "==> Starting Ray worker on ${worker_ip} (ssh ${ssh_target})"
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 "${ssh_target}" bash <<EOF
set -e
ray stop --force || true
rm -rf /tmp/ray 2>/dev/null || true
ray start --node-ip-address="${worker_ip}" --address="${MASTER_IP}:${RAY_PORT}" --num-gpus="${N_GPUS_PER_NODE}"
EOF
}

wait_ray_ready() {
  local need_nodes=$1
  local need_gpus=$2
  local timeout=$3
  local waited=0
  local alive=0
  local gpus=0

  while (( waited < timeout )); do
    read -r alive gpus < <(python3 - <<'PY' 2>/dev/null || echo "0 0"
import ray
try:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    alive = sum(1 for n in ray.nodes() if n.get("Alive"))
    gpus = int(ray.cluster_resources().get("GPU", 0))
    print(alive, gpus)
except Exception:
    print(0, 0)
finally:
    if ray.is_initialized():
        ray.shutdown()
PY
)
    alive="${alive:-0}"
    gpus="${gpus:-0}"
    if (( alive >= need_nodes && gpus >= need_gpus )); then
      echo "==> Ray cluster ready: nodes=${alive}/${need_nodes}, GPUs=${gpus}/${need_gpus}"
      return 0
    fi
    echo "==> Waiting for Ray: nodes=${alive}/${need_nodes}, GPUs=${gpus}/${need_gpus} (${waited}s/${timeout}s)"
    sleep 10
    waited=$((waited + 10))
  done

  echo "ERROR: Ray cluster not ready after ${timeout}s: nodes=${alive}/${need_nodes}, GPUs=${gpus}/${need_gpus}"
  return 1
}

resolve_train_script() {
  local script=$1
  if [[ "${script}" = /* ]]; then
    echo "${script}"
    return 0
  fi
  if [[ "${script}" == */* ]]; then
    echo "${REPO_ROOT}/${script}"
    return 0
  fi
  for candidate in \
    "${REPO_ROOT}/recipe/flowsd/${script}" \
    "${REPO_ROOT}/recipe/sdpo/${script}" \
    "${REPO_ROOT}/recipe/antisd/${script}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "${REPO_ROOT}/${script}"
}

submit_training() {
  local script_path
  script_path="$(resolve_train_script "${TRAIN_SCRIPT}")"
  if [[ ! -f "${script_path}" ]]; then
    echo "ERROR: TRAIN_SCRIPT not found: ${TRAIN_SCRIPT} -> ${script_path}"
    exit 1
  fi

  echo "==> Submitting training: ${script_path}"
  echo "==> NNODES=${NNODES}, N_GPUS_PER_NODE=${N_GPUS_PER_NODE}, RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:${DASHBOARD_PORT}}"
  cd "${REPO_ROOT}"
  NNODES="${NNODES}" \
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
  RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:${DASHBOARD_PORT}}" \
  WORKING_DIR="${WORKING_DIR:-${REPO_ROOT}}" \
  RUNTIME_ENV="${RUNTIME_ENV:-${REPO_ROOT}/recipe/sdpo/runtime_env.yaml}" \
  bash "${script_path}"
}

########################
# MAIN
########################

if (( NNODES != ${#WORKERS[@]} )); then
  echo "WARN: NNODES=${NNODES} but NODE_IP_LIST has ${#WORKERS[@]} nodes; Ray readiness will wait for NNODES."
fi

start_head

NUM_WORKERS=${#WORKERS[@]}
if (( NUM_WORKERS > 1 )); then
  for idx in $(seq 1 $(( NUM_WORKERS - 1 ))); do
    start_worker "${WORKERS[$idx]}" "${idx}"
  done
fi

echo
echo "Ray cluster started (${NUM_WORKERS} nodes)"
echo "Dashboard: http://${MASTER_IP}:${DASHBOARD_PORT}"

if [[ "${AUTO_TRAIN}" == "1" ]]; then
  wait_ray_ready "${NNODES}" "$(( NNODES * N_GPUS_PER_NODE ))" "${RAY_READY_TIMEOUT}"
  submit_training
  if [[ "${KEEP_ALIVE_AFTER_SUBMIT}" == "1" ]]; then
    echo "==> Training submitted. Keeping Ray cluster alive."
    sleep infinity
  fi
fi
