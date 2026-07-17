#!/usr/bin/env python3
"""One-click ONNX Runtime + Intel OpenVINO EP smoke test and benchmark.

The script creates a small static ONNX model locally, discovers Intel devices,
runs a CPU reference, and then runs the requested CPU/GPU/NPU/AUTO target.
No model download or ML framework is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "openvino_ep_smoke.onnx"
CACHE_ROOT = ROOT / ".openvino_cache"
TARGETS = ("CPU", "GPU", "NPU")
DEVICE_TOKEN = re.compile(r"^(?:CPU|NPU|GPU(?:\.\d+)?)$")
EXPECTED_NODE_TYPES = Counter({"MatMul": 2, "Add": 2, "Relu": 1})


def normalize_distribution_name(name: str) -> str:
    """Normalize a Python distribution name for reliable ownership checks."""
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_installed_stack() -> None:
    """Reject package combinations known to overwrite imports or native libs."""
    distributions = {
        normalize_distribution_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    known_ort_packages = {
        name
        for name in distributions
        if name
        in {
            "onnxruntime",
            "onnxruntime-gpu",
            "onnxruntime-directml",
            "onnxruntime-openvino",
        }
    }
    ort_package_owners = {
        normalize_distribution_name(name)
        for name in importlib.metadata.packages_distributions().get("onnxruntime", [])
    }
    if known_ort_packages != {"onnxruntime-openvino"} or ort_package_owners != {
        "onnxruntime-openvino"
    }:
        raise RuntimeError(
            "Install exactly onnxruntime-openvino==1.24.1; found known ONNX Runtime "
            f"distributions {sorted(known_ort_packages) or ['none']} and owners of the "
            f"onnxruntime import {sorted(ort_package_owners) or ['none']}. Recreate .venv."
        )
    if distributions.get("onnxruntime-openvino") != "1.24.1":
        raise RuntimeError("This demo requires onnxruntime-openvino==1.24.1. Recreate .venv.")

    openvino_version = distributions.get("openvino")
    if platform.system() == "Windows" and openvino_version != "2025.4.1":
        raise RuntimeError("Windows requires openvino==2025.4.1. Recreate .venv.")
    if platform.system() == "Linux" and openvino_version is not None:
        raise RuntimeError(
            "Do not install the standalone openvino wheel beside onnxruntime-openvino on Linux; "
            "their duplicate native libraries conflict. Recreate .venv."
        )


def add_openvino_windows_dlls() -> None:
    """Expose the matching OpenVINO wheel DLLs before creating a session on Windows."""
    if platform.system() != "Windows":
        return
    try:
        import onnxruntime.tools.add_openvino_win_libs as utils

        utils.add_openvino_libs_to_path()
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Windows needs the matching OpenVINO runtime. Run: "
            "python -m pip install openvino==2025.4.1"
        ) from exc


def import_runtime() -> tuple[Any, Any]:
    validate_installed_stack()
    add_openvino_windows_dlls()
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Missing packages. Activate the tutorial virtual environment and run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return onnx, ort


def build_model(onnx: Any, path: Path) -> None:
    """Create a deterministic, static float32 graph supported by CPU/GPU/NPU."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(20260715)
    input_size = 512
    hidden_size = 1024
    output_size = 256

    w1 = (rng.standard_normal((input_size, hidden_size)) * 0.025).astype(np.float32)
    b1 = (rng.standard_normal(hidden_size) * 0.01).astype(np.float32)
    w2 = (rng.standard_normal((hidden_size, output_size)) * 0.025).astype(np.float32)
    b2 = (rng.standard_normal(output_size) * 0.01).astype(np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "w1"], ["mm1"], name="hidden_matmul"),
            helper.make_node("Add", ["mm1", "b1"], ["add1"], name="hidden_bias"),
            helper.make_node("Relu", ["add1"], ["relu1"], name="hidden_relu"),
            helper.make_node("MatMul", ["relu1", "w2"], ["mm2"], name="output_matmul"),
            helper.make_node("Add", ["mm2", "b2"], ["output"], name="output_bias"),
        ],
        "openvino_ep_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size])],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="openvino-ep-one-click-demo",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def discover_openvino_devices(ort: Any) -> tuple[list[str], dict[str, str]]:
    """Query devices through the OpenVINO API bundled in the pinned ORT wheel."""
    info = {"source": "ONNX Runtime OpenVINO device API"}
    try:
        devices = list(ort.capi._pybind_state.get_available_openvino_device_ids())
    except Exception as exc:  # Driver/plugin failures vary by platform and release.
        info["error"] = f"{type(exc).__name__}: {exc}"
        return [], info

    if platform.system() == "Windows":
        try:
            from openvino import get_version

            info["version"] = get_version()
        except Exception as exc:
            info["version_error"] = f"{type(exc).__name__}: {exc}"
    else:
        info["version"] = "2025.4.1 (bundled by onnxruntime-openvino 1.24.1)"
    return devices, info


def normalize_target(raw: str) -> str:
    target = raw.strip().upper()
    if target == "AUTO":
        raise ValueError(
            "Bare AUTO is not a reliable qualification target in this pinned release. "
            "Use an explicit list such as AUTO:GPU,CPU or AUTO:NPU,CPU."
        )
    if target in TARGETS or DEVICE_TOKEN.fullmatch(target):
        return target

    mode, separator, device_list = target.partition(":")
    devices = device_list.split(",") if separator else []
    if (
        mode not in {"AUTO", "MULTI", "HETERO"}
        or len(devices) < (1 if mode == "AUTO" else 2)
        or not all(DEVICE_TOKEN.fullmatch(device) for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            f"Unsupported target {raw!r}. Use CPU, GPU, GPU.0, NPU, "
            "AUTO:GPU,NPU,CPU, HETERO:GPU,CPU, or MULTI:GPU,CPU."
        )
    return target


def cache_path(target: str) -> Path:
    """Return a target-specific path that cannot escape the cache root."""
    safe_name = re.sub(r"[^A-Z0-9._-]+", "_", target)
    return CACHE_ROOT / safe_name


def provider_options(target: str, cache_dir: Path) -> dict[str, str]:
    base = target.split(":", 1)[0].split(".", 1)[0]
    config_device = (
        base if base in {"CPU", "GPU", "NPU", "AUTO", "MULTI", "HETERO"} else "CPU"
    )
    options = {"device_type": target}
    if config_device in {"CPU", "GPU", "NPU", "AUTO"}:
        config: dict[str, dict[str, str]] = {
            config_device: {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": str(cache_dir),
            }
        }
        options["load_config"] = json.dumps(config)
    return options


def ensure_target_is_available(target: str, devices: list[str]) -> None:
    """Reject missing hardware instead of allowing a meta-device to skip it."""
    if not devices:
        raise RuntimeError(
            "OpenVINO did not enumerate any device. Repair the runtime/driver installation "
            "before creating an inference session."
        )

    available = {device.upper() for device in devices}
    requested_devices = target.split(":", 1)[1].split(",") if ":" in target else [target]

    def is_available(device: str) -> bool:
        if device == "GPU":
            return any(item == "GPU" or item.startswith("GPU.") for item in available)
        return device in available

    missing = [device for device in requested_devices if not is_available(device)]
    if missing:
        raise RuntimeError(
            f"Requested {target}, but {missing} are absent from OpenVINO's enumerated "
            f"devices {sorted(available)}. Install/repair those drivers before inference."
        )


def ensure_openvino_session(session: Any) -> None:
    """Catch ORT's provider-construction fallback to CPU after a loader error."""
    if "OpenVINOExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "OpenVINO provider registration failed and ORT created a CPU-only session. "
            "Read the loader warning above and repair the matched runtime/driver stack."
        )


def verify_graph_assignment(session: Any) -> tuple[int, list[str]]:
    """Directly prove that all five smoke nodes were assigned to OpenVINO EP."""
    assignments = session.get_provider_graph_assignment_info()
    if not assignments:
        raise RuntimeError("ONNX Runtime returned no recorded graph-assignment information.")

    providers = {assignment.ep_name for assignment in assignments}
    nodes = [node for assignment in assignments for node in assignment.get_nodes()]
    node_types = Counter(node.op_type for node in nodes)
    if providers != {"OpenVINOExecutionProvider"} or node_types != EXPECTED_NODE_TYPES:
        details = [
            f"{assignment.ep_name}: {[node.op_type for node in assignment.get_nodes()]}"
            for assignment in assignments
        ]
        raise RuntimeError(
            "Unexpected graph assignment; expected all five smoke nodes on OpenVINO EP, "
            f"received {details}."
        )
    return len(nodes), [node.name or node.op_type for node in nodes]


def create_session(ort: Any, model_path: Path, target: str, cache_dir: Path) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    # Let OpenVINO perform the hardware-aware graph optimization recommended by
    # the ONNX Runtime OpenVINO EP guide.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # This graph is fully supported. Fail on ORT CPU assignment, suppress the
    # OpenVINO NPU-to-CPU fallback, and record assignment for a direct assertion.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    providers = [("OpenVINOExecutionProvider", provider_options(target, cache_dir))]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def create_cpu_reference(ort: Any, model_path: Path) -> Any:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def benchmark(
    session: Any, data: np.ndarray, warmup: int, runs: int
) -> tuple[np.ndarray, float, float]:
    input_name = session.get_inputs()[0].name
    feeds = {input_name: data}
    for _ in range(warmup):
        session.run(None, feeds)

    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = session.run(None, feeds)[0]
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

    assert output is not None
    return output, statistics.median(samples), statistics.fmean(samples)


def validation_tolerance(target: str) -> tuple[float, float]:
    """Allow expected FP16 rounding on GPU/NPU while keeping CPU validation tight."""
    requested_devices = target.split(":", 1)[1].split(",") if ":" in target else [target]
    if requested_devices == ["CPU"]:
        return 1e-4, 1e-4
    return 1e-2, 5e-3


def print_header(ort: Any, devices: list[str], ov_info: dict[str, str]) -> None:
    print("=" * 72)
    print("ONNX Runtime + Intel OpenVINO Execution Provider: one-click demo")
    print("=" * 72)
    print(f"OS / Python       : {platform.platform()} / {platform.python_version()}")
    print(f"ONNX Runtime      : {ort.__version__}")
    print(f"ORT providers     : {ort.get_available_providers()}")
    if "version" in ov_info:
        print(f"OpenVINO Runtime  : {ov_info['version']}")
    elif "version_error" in ov_info:
        print(f"OpenVINO version  : unavailable ({ov_info['version_error']})")
    print(f"Device query      : {ov_info.get('source', 'unknown source')}")
    if "error" in ov_info:
        print(f"Device query error: {ov_info['error']}")
    print(f"Intel devices     : {devices or ['not enumerated']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a tiny ONNX model and run it with ONNX Runtime OpenVINO EP."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("OV_DEVICE", "CPU"),
        help="CPU (default), GPU, GPU.0, GPU.1, NPU, or an explicit OpenVINO meta-device string",
    )
    parser.add_argument("--runs", type=int, default=50, help="timed runs (default: 50)")
    parser.add_argument(
        "--warmups",
        "--warmup",
        dest="warmups",
        type=int,
        default=5,
        help="warm-up runs (default: 5)",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="generated model path")
    args = parser.parse_args()

    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be >= 1 and --warmups must be >= 0")

    try:
        onnx, ort = import_runtime()
        devices, ov_info = discover_openvino_devices(ort)
        print_header(ort, devices, ov_info)

        if "OpenVINOExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "OpenVINOExecutionProvider is absent. Remove conflicting ONNX Runtime wheels "
                "and install onnxruntime-openvino in this virtual environment."
            )

        target = normalize_target(args.device)
        print(f"Requested target  : {args.device.strip().upper()}")
        print(f"Resolved target   : {target}")
        ensure_target_is_available(target, devices)

        build_model(onnx, args.model)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1, 512)).astype(np.float32)

        reference_session = create_cpu_reference(ort, args.model)
        reference = reference_session.run(
            None, {reference_session.get_inputs()[0].name: data}
        )[0]

        session = create_session(ort, args.model, target, cache_path(target))
        ensure_openvino_session(session)
        assigned_node_count, assigned_node_names = verify_graph_assignment(session)
        output, median_ms, mean_ms = benchmark(session, data, args.warmups, args.runs)
        rtol, atol = validation_tolerance(target)
        np.testing.assert_allclose(output, reference, rtol=rtol, atol=atol)

        print(f"Session providers : {session.get_providers()}")
        print(
            f"Graph assignment  : OpenVINOExecutionProvider "
            f"({assigned_node_count}/5 nodes: {', '.join(assigned_node_names)})"
        )
        print(f"Validation limits : rtol={rtol:g}, atol={atol:g}")
        print(f"Median latency    : {median_ms:.3f} ms")
        print(f"Mean latency      : {mean_ms:.3f} ms ({args.runs} runs)")
        print(f"Max |CPU-target|  : {float(np.max(np.abs(reference - output))):.6g}")
        print("\nPASS: all five demo nodes were assigned to OpenVINO EP and output is valid.")
        print("Note: session creation compiles the model; timed runs follow warm-up.")
        return 0
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check the matching README troubleshooting table. For GPU/NPU, verify the "
            "driver and confirm that the demo lists the requested device.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())