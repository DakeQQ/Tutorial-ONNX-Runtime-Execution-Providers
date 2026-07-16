#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-webgpu}"
if [[ $# -gt 0 ]]; then
    shift
fi

find_python() {
    local check_code="$1"
    local candidate
    local candidates=()
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        candidates=("$PYTHON_BIN")
    else
        candidates=(python3 python3.12 python3.13 python3.11 python3.14 python3.10)
    fi

    for candidate in "${candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c "$check_code" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

case "$MODE" in
    native-webgpu)
        if ! PYTHON_BIN="$(find_python "import platform, sys; raise SystemExit(platform.python_implementation() != 'CPython' or not (3, 11) <= sys.version_info[:2] < (3, 15) or sys.maxsize <= 2**32)")"; then
            echo "ERROR: install 64-bit CPython 3.11, 3.12, 3.13, or 3.14 for native WebGPU." >&2
            exit 1
        fi
        exec "$PYTHON_BIN" launch_demo.py native-webgpu "$@"
        ;;
    webgpu|webnn|wasm)
        if ! PYTHON_BIN="$(find_python "import sys; raise SystemExit(sys.version_info < (3, 10))")"; then
            echo "ERROR: the browser launcher requires Python 3.10 or newer for its local server." >&2
            exit 1
        fi
        exec "$PYTHON_BIN" launch_demo.py "$MODE" "$@"
        ;;
    *)
        echo "Usage: bash run_demo.sh [native-webgpu|webgpu|webnn|wasm] [options]" >&2
        exit 2
        ;;
esac
