#!/usr/bin/env bash
# Compatibility shim for launch/start.sh, which invokes recipe/sdpo/${RUN_SCRIPT}.
# The actual Anti-SD implementation lives under recipe/antisd/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../antisd/run_math_antisd.sh" "$@"
