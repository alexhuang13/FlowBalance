#!/usr/bin/env bash
# Compatibility shim for launch/start.sh. Canonical RLSD lives in recipe/rlsd.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../rlsd/run_math_rlsd.sh" "$@"
