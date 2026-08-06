#!/usr/bin/env bash
set -uxo pipefail

YOUR_CODE=${YOUR_CODE:-"python3 ${PWD}/scripts/test.py"}

########################
# CONFIG
########################

# Worker node IPs
WORKERS=($(echo $NODE_IP_LIST | sed 's/:.//g' | sed "s/,/ /g"))

# Master node IP (this machine)
MASTER_IP=${WORKERS[0]}

########################
# FUNCTIONS
########################

start_head() {
  echo "==> Starting on master ($MASTER_IP)"
${YOUR_CODE}
}

start_worker() {
  local worker_ip=$1
  echo "==> Starting on worker $worker_ip"

  ssh "$worker_ip" bash <<EOF
${YOUR_CODE}
EOF
}

########################
# MAIN
########################

start_head &

NUM_WORKERS=${#WORKERS[@]}
if (( NUM_WORKERS > 1 )); then
for w in "${WORKERS[@]:1}"; do
  start_worker "$w" &
done
fi

echo
echo "Done"
