#!/usr/bin/env bash
# Compatibility shim for launch/start.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../flowopsd/run_math_flowopsd_eta_sweep.sh" "$@"
