#!/usr/bin/env bash
# Compatibility shim for launch/start.sh, which invokes recipe/sdpo/${RUN_SCRIPT}.
# The actual FlowSD implementation lives under recipe/flowsd/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../flowsd/run_math_flowsd.sh" "$@"
