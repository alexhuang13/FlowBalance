#!/usr/bin/env bash
# Compatibility shim for launch/start.sh, which invokes recipe/sdpo/${RUN_SCRIPT}.
# The actual FlowOPSD implementation lives under recipe/flowopsd/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../flowopsd/run_math_flowopsd.sh" "$@"
