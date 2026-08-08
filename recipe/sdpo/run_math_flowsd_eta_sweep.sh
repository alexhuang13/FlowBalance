#!/usr/bin/env bash
# Compatibility shim for launch/start.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../flowsd/run_math_flowsd_eta_sweep.sh" "$@"
