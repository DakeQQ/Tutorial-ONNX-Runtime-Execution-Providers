#!/usr/bin/env python3
"""One-click ONNX Runtime CoreML execution-provider proof for macOS.

The launcher creates an isolated environment, installs the pinned Apple Silicon
stack, generates a deterministic static Conv model, and starts a worker that
requires CoreML graph execution with ONNX Runtime CPU fallback disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time
import traceback
import venv
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENV = SCRIPT_DIR / ".venv-coreml"
DEFAULT_ARTIFACTS = SCRIPT_DIR / ".coreml-smoke"
DEFAULT_CACHE = SCRIPT_DIR / ".coreml-cache"
REQUIREMENTS = SCRIPT_DIR / "requirements.txt"
COREML_EP = "CoreMLExecutionProvider"
MODEL_FORMATS = {
    "mlprogram": "MLProgram",
    "neuralnetwork": "NeuralNetwork",
}
COMPUTE_UNITS = {
    "all": "ALL",
    "cpu": "CPUOnly",
    "cpu-gpu": "CPUAndGPU",
    "cpu-ane": "CPUAndNeuralEngine",
}
EXPECTED_VERSIONS = {
    "onnx": "1.22.0",
    "onnxruntime": "1.27.0",
}
ORT_DISTRIBUTIONS = frozenset(
    {
        "onnxruntime",
        "onnxruntime-azure",
        "onnxruntime-directml",
        "onnxruntime-gpu",
        "onnxruntime-migraphx",
        "onnxruntime-openvino",
        "onnxruntime-rocm",
        "onnxruntime-silicon",
        "onnxruntime-training",
        "onnxruntime-vitisai",
        "onnxruntime-windowsml",
    }
)


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _macos_version() -> tuple[int, ...]:
    version = platform.mac_ver()[0]
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def _validate_host() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "This one-click inference path requires macOS. CoreML is unavailable on "
            "Linux/Windows; iOS uses the CocoaPods or custom-build route in the README."
        )
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "The pinned onnxruntime 1.27.0 PyPI wheel is Apple Silicon only. "
            "Use a native arm64 Python process on an M-series Mac."
        )
    if _macos_version() < (14, 0):
        raise RuntimeError(
            "The pinned onnxruntime 1.27.0 wheel targets macOS 14.0 or newer."
        )
    if platform.python_implementation() != "CPython":
        raise RuntimeError("Use CPython for the pinned ONNX Runtime wheel.")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Use 64-bit CPython 3.11, 3.12, 3.13, or 3.14.")
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("The pinned ONNX Runtime wheel does not support free-threaded CPython.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Use a 64-bit Python installation.")


def _expected_versions() -> dict[str, str]:
    numpy_version = "2.4.6" if sys.version_info[:2] == (3, 11) else "2.5.1"
    return {**EXPECTED_VERSIONS, "numpy": numpy_version}


def _installed_ort_distributions() -> set[str]:
    installed: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            canonical = re.sub(r"[-_.]+", "-", name).lower()
            if canonical in ORT_DISTRIBUTIONS:
                installed.add(canonical)
    return installed


def _validate_worker_packages() -> None:
    expected = _expected_versions()
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Pinned package mismatch: expected {expected}, found {actual}.")
    installed_ort = _installed_ort_distributions()
    if installed_ort != {"onnxruntime"}:
        raise RuntimeError(
            "Install exactly one ONNX Runtime distribution in this environment; "
            f"found {sorted(installed_ort)}."
        )


def _environment_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    check = (
        "import importlib.metadata as m, re, sys; "
        f"expected={_expected_versions()!r}; "
        f"ort_candidates={set(ORT_DISTRIBUTIONS)!r}; "
        "actual={name:m.version(name) for name in expected}; "
        "all_names={re.sub(r'[-_.]+','-',name).lower() "
        "for dist in m.distributions() for name in [dist.metadata.get('Name')] if name}; "
        "installed_ort=all_names & ort_candidates; "
        "raise SystemExit(not (actual == expected and "
        "installed_ort == {'onnxruntime'} and "
        "(3,11) <= sys.version_info[:2] < (3,15) and sys.maxsize > 2**32))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", check],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_environment(venv_dir: Path, refresh: bool) -> Path:
    _validate_host()
    python = _venv_python(venv_dir)
    if not refresh and _environment_ready(python):
        print(f"[1/3] Reusing pinned environment: {venv_dir}")
        return python

    if venv_dir.exists():
        if not venv_dir.is_dir() or not (venv_dir / "pyvenv.cfg").is_file():
            raise RuntimeError(
                "Refusing to remove --venv-dir because it is not an existing Python "
                f"virtual environment: {venv_dir}"
            )
        print(f"[1/3] Recreating non-matching environment: {venv_dir}")
        shutil.rmtree(venv_dir)
    else:
        print(f"[1/3] Creating isolated environment: {venv_dir}")

    try:
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    except Exception as exc:
        raise RuntimeError("Could not create the CoreML virtual environment.") from exc
    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS}")

    print("[2/3] Installing pinned ONNX Runtime + CoreML packages...")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    if not _environment_ready(python):
        raise RuntimeError("The isolated environment does not match the pinned CoreML stack.")
    return python


def _generate_model(model_path: Path, np: Any, onnx: Any) -> str:
    from onnx import TensorProto, helper, numpy_helper

    weights = np.linspace(-0.2, 0.2, 4 * 3 * 3 * 3, dtype=np.float32).reshape(4, 3, 3, 3)
    bias = np.asarray([-0.15, -0.05, 0.05, 0.15], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["conv"],
                name="coreml_smoke_conv",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Relu", ["conv"], ["relu"], name="coreml_smoke_relu"),
            helper.make_node(
                "GlobalAveragePool",
                ["relu"],
                ["output"],
                name="coreml_smoke_pool",
            ),
        ],
        "coreml_strict_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 1, 1])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph,
        producer_name="ort-coreml-one-click",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10

    import hashlib

    cache_material = (
        EXPECTED_VERSIONS["onnxruntime"].encode("ascii")
        + b"\0"
        + model.SerializeToString()
    )
    cache_key = hashlib.sha256(cache_material).hexdigest()[:48]
    metadata = model.metadata_props.add()
    metadata.key = "COREML_CACHE_KEY"
    metadata.value = cache_key
    onnx.checker.check_model(model)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = model.SerializeToString()
    if not model_path.is_file() or model_path.read_bytes() != serialized:
        model_path.write_bytes(serialized)
    return cache_key


def _input_and_reference(np: Any) -> tuple[Any, Any]:
    input_data = np.linspace(-1.0, 1.0, 1 * 3 * 16 * 16, dtype=np.float32).reshape(1, 3, 16, 16)
    weights = np.linspace(-0.2, 0.2, 4 * 3 * 3 * 3, dtype=np.float32).reshape(4, 3, 3, 3)
    bias = np.asarray([-0.15, -0.05, 0.05, 0.15], dtype=np.float32)
    padded = np.pad(input_data, ((0, 0), (0, 0), (1, 1), (1, 1)))
    conv = np.zeros((1, 4, 16, 16), dtype=np.float32)
    for out_channel in range(4):
        conv[:, out_channel] = bias[out_channel]
        for in_channel in range(3):
            for kernel_y in range(3):
                for kernel_x in range(3):
                    conv[:, out_channel] += (
                        padded[
                            :,
                            in_channel,
                            kernel_y : kernel_y + 16,
                            kernel_x : kernel_x + 16,
                        ]
                        * weights[out_channel, in_channel, kernel_y, kernel_x]
                    )
    reference = np.maximum(conv, np.float32(0.0)).mean(axis=(2, 3), keepdims=True)
    return input_data, reference


def _profile_provider_counts(profile_path: str | os.PathLike[str]) -> Counter[str]:
    with Path(profile_path).open("r", encoding="utf-8") as stream:
        events = json.load(stream)
    counts: Counter[str] = Counter()
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[str(provider)] += 1
    return counts


def _assignment_provider_counts(session: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    getter = getattr(session, "get_provider_graph_assignment_info", None)
    if getter is None:
        return counts
    for assignment in getter():
        nodes = list(assignment.get_nodes())
        counts[str(assignment.ep_name)] += max(1, len(nodes))
    return counts


def _run_timed(
    session: Any,
    feeds: dict[str, Any],
    warmups: int,
    runs: int,
) -> tuple[Any, float, float]:
    output = None
    for _ in range(warmups):
        output = session.run(None, feeds)[0]

    samples_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = session.run(None, feeds)[0]
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    if output is None:
        raise RuntimeError("Inference produced no output.")
    return output, statistics.median(samples_ms), statistics.fmean(samples_ms)


def _provider_options(args: argparse.Namespace) -> dict[str, str]:
    options = {
        "ModelFormat": MODEL_FORMATS[args.model_format],
        "MLComputeUnits": COMPUTE_UNITS[args.compute_units],
        "RequireStaticInputShapes": "1",
        "EnableOnSubgraphs": "0",
        "SpecializationStrategy": (
            "FastPrediction" if args.specialization == "fast-prediction" else "Default"
        ),
        "ProfileComputePlan": "1" if args.profile_compute_plan else "0",
        "AllowLowPrecisionAccumulationOnGPU": (
            "1" if args.allow_low_precision_gpu else "0"
        ),
    }
    if not args.no_cache:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        options["ModelCacheDirectory"] = str(args.cache_dir.resolve())
    return options


def _worker(args: argparse.Namespace) -> int:
    _validate_host()
    _validate_worker_packages()
    if args.profile_compute_plan and args.model_format != "mlprogram":
        raise RuntimeError("--profile-compute-plan requires --model-format mlprogram.")
    if args.profile_compute_plan and _macos_version() < (14, 4):
        raise RuntimeError("--profile-compute-plan requires macOS 14.4 or newer.")
    if args.specialization == "fast-prediction" and _macos_version() < (15, 0):
        raise RuntimeError(
            "--specialization fast-prediction requires Core ML 8 (macOS 15 or newer)."
        )
    if args.allow_low_precision_gpu and args.compute_units not in {"all", "cpu-gpu"}:
        raise RuntimeError("--allow-low-precision-gpu requires GPU-capable compute units.")

    import numpy as np
    import onnx
    import onnxruntime as ort

    print("=" * 76)
    print("ONNX Runtime + Apple CoreML EP strict proof test")
    print("=" * 76)
    print(f"macOS / process     : {platform.mac_ver()[0]} / {platform.machine()}")
    print(f"Python              : {platform.python_version()} ({sys.executable})")
    print(f"ONNX Runtime        : {ort.__version__}")
    print(f"Available providers : {ort.get_available_providers()}")
    if COREML_EP not in ort.get_available_providers():
        raise RuntimeError(
            "CoreMLExecutionProvider is absent. Use the pinned macOS arm64 onnxruntime "
            "wheel; provider availability alone is not an execution proof."
        )

    model_path = args.artifacts_dir / "coreml_smoke.onnx"
    cache_key = _generate_model(model_path, np, onnx)
    input_data, reference = _input_and_reference(np)
    provider_options = _provider_options(args)
    print(f"Model               : {model_path} (static Conv partition)")
    print(f"Model cache key     : {cache_key}")
    print(f"Model format        : {provider_options['ModelFormat']}")
    print(f"Requested units     : {provider_options['MLComputeUnits']}")
    print(f"Persistent cache    : {provider_options.get('ModelCacheDirectory', 'disabled')}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_prefix = args.artifacts_dir / "coreml-profile"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")

    session = None
    profile_ended = False
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[(COREML_EP, provider_options)],
        )
        if COREML_EP not in session.get_providers():
            raise RuntimeError(f"CoreML was not registered in the session: {session.get_providers()}")
        assignment_counts = _assignment_provider_counts(session)
        output, median_ms, mean_ms = _run_timed(
            session,
            {"input": input_data},
            args.warmups,
            args.runs,
        )
        profile_path = session.end_profiling()
        profile_ended = True
        profile_counts = _profile_provider_counts(profile_path)

        coreml_assigned = assignment_counts.get(COREML_EP, 0)
        coreml_profiled = profile_counts.get(COREML_EP, 0)
        cpu_profiled = profile_counts.get("CPUExecutionProvider", 0)
        if coreml_assigned == 0 and coreml_profiled == 0:
            raise RuntimeError(
                "Inference completed, but assignment/profile data did not prove CoreML "
                f"execution. Assignment={dict(assignment_counts)}, profile={dict(profile_counts)}"
            )
        if cpu_profiled:
            raise RuntimeError(
                "The strict session unexpectedly profiled ONNX Runtime CPU nodes: "
                f"{dict(profile_counts)}"
            )

        tolerance = 5e-3
        np.testing.assert_allclose(output, reference, rtol=tolerance, atol=tolerance)
        max_error = float(np.max(np.abs(output - reference)))
        print(f"Session providers    : {session.get_providers()}")
        print(f"Graph assignment     : {dict(assignment_counts)}")
        print(f"Profiled providers   : {dict(profile_counts)}")
        print(f"CoreML median / mean : {median_ms:.3f} / {mean_ms:.3f} ms")
        print(f"Max |CoreML-NumPy|   : {max_error:.8g} (limit {tolerance:g})")
        print(
            "\nPASS: CoreMLExecutionProvider executed the complete non-trivial "
            "partition with ONNX Runtime CPU EP fallback disabled."
        )
        print(
            "Compute-unit boundary: this proves the CoreML EP path, not which Core ML "
            "device ran every operation. Use --profile-compute-plan on supported macOS "
            "versions or Xcode Instruments for CPU/GPU/ANE placement evidence."
        )
        print("Note: this tiny graph validates configuration; it is not a benchmark.")
        return 0
    finally:
        if session is not None and not profile_ended:
            try:
                session.end_profiling()
            except Exception:
                pass
        del session
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pinned environment and strictly prove CoreML EP execution on macOS."
    )
    parser.add_argument(
        "--model-format",
        choices=tuple(MODEL_FORMATS),
        default="mlprogram",
        help="Core ML model representation (default: mlprogram).",
    )
    parser.add_argument(
        "--compute-units",
        choices=tuple(COMPUTE_UNITS),
        default="all",
        help="Core ML compute-unit policy (default: all).",
    )
    parser.add_argument("--warmups", type=int, default=5, help="Warm-up inferences.")
    parser.add_argument("--runs", type=int, default=30, help="Measured inferences.")
    parser.add_argument(
        "--specialization",
        choices=("default", "fast-prediction"),
        default="default",
        help="Core ML specialization strategy.",
    )
    parser.add_argument(
        "--profile-compute-plan",
        action="store_true",
        help="Ask MLProgram to log Core ML per-operation device placement.",
    )
    parser.add_argument(
        "--allow-low-precision-gpu",
        action="store_true",
        help="Allow lower-precision GPU accumulation.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable persistent Core ML cache.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--venv-dir", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--refresh", action="store_true", help="Recreate the pinned environment.")
    parser.add_argument("--verbose", action="store_true", help="Print a traceback after failure.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be >= 0 and --runs must be >= 1")
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    args.venv_dir = args.venv_dir.expanduser().resolve()
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.worker:
            print("[3/3] Running strict CoreML proof...")
            return _worker(args)

        python = _ensure_environment(args.venv_dir, args.refresh)
        worker_args = [argument for argument in sys.argv[1:] if argument != "--refresh"]
        os.execv(
            str(python),
            [str(python), str(Path(__file__).resolve()), "--worker", *worker_args],
        )
        return 0
    except Exception as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())