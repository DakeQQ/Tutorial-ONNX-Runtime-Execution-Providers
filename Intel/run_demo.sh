#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NEW_VENV=0
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python3 was not found. Install 64-bit CPython 3.11, 3.12, or 3.13 first." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import platform, struct, sys; raise SystemExit(0 if sys.implementation.name == "cpython" and (3, 11) <= sys.version_info[:2] <= (3, 13) and struct.calcsize("P") == 8 and platform.machine().lower() in {"x86_64", "amd64"} else 1)'; then
    echo "ERROR: onnxruntime-openvino 1.24.1 requires x86-64 CPython 3.11, 3.12, or 3.13." >&2
    echo "       Python 3.10 and 3.14 do not have a compatible published wheel." >&2
    exit 1
fi

if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import platform, struct, sys; raise SystemExit(0 if sys.implementation.name == "cpython" and (3, 11) <= sys.version_info[:2] <= (3, 13) and struct.calcsize("P") == 8 and platform.machine().lower() in {"x86_64", "amd64"} else 1)'; then
    echo "[1/4] Existing .venv uses an unsupported Python; recreating it..."
    rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "[1/4] Creating .venv..."
    "$PYTHON_BIN" -m venv .venv
    NEW_VENV=1
else
    echo "[1/4] Reusing the supported .venv interpreter."
fi

PY=.venv/bin/python
if ! "$PY" -c 'import importlib.metadata as m, re; norm=lambda name: re.sub(r"[-_.]+", "-", name).lower(); wanted={"onnxruntime-openvino":"1.24.1","onnx":"1.22.0","numpy":"2.3.5"}; names={norm(d.metadata["Name"]) for d in m.distributions() if d.metadata["Name"]}; owners={norm(name) for name in m.packages_distributions().get("onnxruntime", [])}; forbidden={"onnxruntime","onnxruntime-gpu","onnxruntime-directml","openvino"}; ok=owners=={"onnxruntime-openvino"} and names.isdisjoint(forbidden) and all(m.version(name)==version for name,version in wanted.items()); raise SystemExit(0 if ok else 1)' 2>/dev/null; then
    echo "[2/4] Preparing a clean, matched environment..."
    if ((NEW_VENV == 0)); then
        rm -rf .venv
        "$PYTHON_BIN" -m venv .venv
        PY=.venv/bin/python
    fi
    echo "[3/4] Installing the pinned ONNX Runtime/OpenVINO stack..."
    "$PY" -m pip install -r requirements.txt
else
    echo "[2/4] Existing environment matches the pinned stack."
    echo "[3/4] No package changes needed."
fi

if ! "$PY" -m pip check >/dev/null; then
    echo "ERROR: Python package dependencies are inconsistent. Delete .venv and rerun." >&2
    exit 1
fi

echo "[4/4] Running the strict demo..."
"$PY" Test.py "$@"
