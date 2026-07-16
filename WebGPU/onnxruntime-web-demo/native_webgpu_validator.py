#!/usr/bin/env python3
"""One-click ONNX Runtime native WebGPU plugin EP validation and benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import struct
import sys
import tempfile
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "execution_provider_demo.onnx"
REGISTRATION_NAME = "webgpu_demo_registration"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _hardware_summary(ep_device: object, index: int) -> str:
    hardware = ep_device.device
    metadata_map = dict(getattr(hardware, "metadata", {}) or {})
    vendor_id = int(getattr(hardware, "vendor_id", 0) or 0)
    device_id = int(getattr(hardware, "device_id", 0) or 0)
    hardware_type = getattr(getattr(hardware, "type", None), "name", "GPU")
    known_vendors = {
        0x1002: "AMD",
        0x106B: "Apple",
        0x10DE: "NVIDIA",
        0x17CB: "Qualcomm",
        0x5143: "Qualcomm",
        0x8086: "Intel",
    }
    reported_vendor = str(getattr(hardware, "vendor", "") or "")
    vendor = (
        reported_vendor
        if reported_vendor.lower() not in {"", "unknown"}
        else known_vendors.get(vendor_id, "unknown")
    )
    details = ", ".join(f"{key}={value}" for key, value in sorted(metadata_map.items()))
    if not details:
        details = "no extra metadata"
    return (
        f"[{index}] {hardware_type} vendor={vendor} "
        f"vendor_id=0x{vendor_id:04x} device_id=0x{device_id:04x} ({details})"
    )


def _make_inputs() -> dict[str, np.ndarray]:
    shape = (1, 4, 128, 128)
    size = int(np.prod(shape))
    positions = np.arange(size, dtype=np.float32)
    left = (np.sin(positions * 0.01) * 0.25).reshape(shape)
    right = (np.cos(positions * 0.013) * 0.25).reshape(shape)
    return {"left": left, "right": right}


def _compare_outputs(
    names: list[str], reference: list[np.ndarray], candidate: list[np.ndarray]
) -> bool:
    passed = True
    print("\n[Correctness] CPU reference versus native WebGPU plugin")
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            print(
                f"  FAIL {name}: expected {expected.dtype}{expected.shape}, "
                f"got {actual.dtype}{actual.shape}"
            )
            passed = False
            continue

        if np.issubdtype(expected.dtype, np.integer):
            max_abs = int(
                np.max(
                    np.abs(
                        expected.astype(np.int64, copy=False)
                        - actual.astype(np.int64, copy=False)
                    )
                )
            )
            ok = max_abs <= 1
        else:
            max_abs = float(
                np.max(
                    np.abs(
                        expected.astype(np.float64, copy=False)
                        - actual.astype(np.float64, copy=False)
                    )
                )
            )
            ok = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-3))

        print(
            f"  {'PASS' if ok else 'FAIL'} {name}: "
            f"dtype={actual.dtype}, shape={actual.shape}, max_abs_diff={max_abs:.6g}"
        )
        passed &= ok
    return passed


def _profile_summary(
    profile_path: Path,
) -> tuple[Counter[str], dict[str, set[tuple[str, str]]]]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    assignments: dict[str, set[tuple[str, str]]] = {}
    for event in events:
        event_args = event.get("args", {})
        provider = event_args.get("provider")
        if not provider:
            continue
        provider = str(provider)
        counts[provider] += 1
        op_name = str(event_args.get("op_name") or "unknown-op")
        node_name = str(event.get("name") or "unknown-node").removesuffix(
            "_kernel_time"
        )
        assignments.setdefault(provider, set()).add((op_name, node_name))
    return counts, assignments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_model_metadata(session: ort.InferenceSession) -> None:
    print("[Session] Model contract:")
    for item in session.get_inputs():
        print(f"  input  {item.name}: {item.type} {item.shape}")
    for item in session.get_outputs():
        print(f"  output {item.name}: {item.type} {item.shape}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ONNX Runtime's native WebGPU plugin EP on a local GPU."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="ONNX model to run (default: included execution_provider_demo.onnx).",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--layout", choices=("NCHW", "NHWC"), default="NCHW")
    parser.add_argument(
        "--power-preference",
        choices=("high-performance", "low-power"),
        default="high-performance",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("disabled", "wgpuOnly", "basic", "full"),
        default="basic",
    )
    parser.add_argument(
        "--keep-profile",
        action="store_true",
        help="Keep the generated ORT JSON profile instead of deleting it.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help=(
            "Allow unsupported nodes to run on CPU. The default is strict "
            "WebGPU-only execution so a passing smoke test proves full assignment."
        ),
    )
    fallback_group.add_argument(
        "--strict", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if args.warmup < 0 or args.iterations < 1:
        print("ERROR: --warmup must be >= 0 and --iterations must be >= 1.", file=sys.stderr)
        return 2

    print("=" * 72)
    print("ONNX Runtime native WebGPU plugin EP demo")
    print("=" * 72)
    print(
        f"Python:                  {sys.version.split()[0]} "
        f"({sys.platform}, {platform.machine()}, {struct.calcsize('P') * 8}-bit)"
    )
    print(f"onnxruntime:             {ort.__version__}")
    print(f"onnxruntime-ep-webgpu:   {_package_version('onnxruntime-ep-webgpu')}")
    print(f"Plugin library:          {webgpu_ep.get_library_path()}")
    print(f"Plugin EP name:          {webgpu_ep.get_ep_name()}")
    print(f"Model:                   {model_path}")
    print(f"Model SHA-256:           {_sha256(model_path)}")
    print(
        "CPU fallback:            "
        + ("allowed (explicit opt-in)" if args.allow_cpu_fallback else "disabled (default strict mode)")
    )

    registered = False
    session: ort.InferenceSession | None = None
    cpu_session: ort.InferenceSession | None = None
    profile_path: Path | None = None
    temporary_profile_dir: tempfile.TemporaryDirectory[str] | None = None

    try:
        ort.register_execution_provider_library(
            REGISTRATION_NAME, webgpu_ep.get_library_path()
        )
        registered = True

        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == webgpu_ep.get_ep_name()
        ]
        print(f"\n[Discovery] Found {len(devices)} WebGPU device(s):")
        for index, device in enumerate(devices):
            print(" ", _hardware_summary(device, index))

        if not devices:
            print(
                "ERROR: the plugin loaded, but Dawn found no compatible GPU. "
                "Check the OS GPU driver and D3D12/Vulkan/Metal support.",
                file=sys.stderr,
            )
            return 3
        if not 0 <= args.device_index < len(devices):
            print(
                f"ERROR: --device-index {args.device_index} is out of range "
                f"(valid: 0..{len(devices) - 1}).",
                file=sys.stderr,
            )
            return 3

        selected_device = devices[args.device_index]
        print(f"[Discovery] Selected: {_hardware_summary(selected_device, args.device_index)}")

        provider_options = {
            "preferredLayout": args.layout,
            "enableGraphCapture": "0",
            "powerPreference": args.power_preference,
            "validationMode": args.validation_mode,
        }

        temporary_profile_dir = tempfile.TemporaryDirectory(prefix="ort_webgpu_profile_")
        profile_prefix = Path(temporary_profile_dir.name) / "native_webgpu"

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(profile_prefix)
        if not args.allow_cpu_fallback:
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        session_options.add_provider_for_devices([selected_device], provider_options)

        print("\n[Session] Provider options:")
        for key, value in provider_options.items():
            print(f"  {key}={value}")

        load_start = time.perf_counter()
        session = ort.InferenceSession(str(model_path), sess_options=session_options)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        print(f"[Session] Active providers: {session.get_providers()}")
        print(f"[Session] Load/compile time: {load_ms:.3f} ms")
        _print_model_metadata(session)

        feeds = _make_inputs()
        expected_inputs = {
            "left": ("tensor(float)", [1, 4, 128, 128]),
            "right": ("tensor(float)", [1, 4, 128, 128]),
        }
        actual_inputs = {
            item.name: (item.type, item.shape) for item in session.get_inputs()
        }
        if actual_inputs != expected_inputs:
            raise RuntimeError(
                "Demo model inputs changed: "
                f"expected {expected_inputs}, got {actual_inputs}"
            )

        for _ in range(args.warmup):
            session.run(None, feeds)

        latencies: list[float] = []
        outputs: list[np.ndarray] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = session.run(None, feeds)
            latencies.append((time.perf_counter() - start) * 1000.0)

        output_names = [
            f"{item.name} ({item.type})" for item in session.get_outputs()
        ]
        print("\n[Benchmark] End-to-end session.run with NumPy CPU I/O; warm-up is excluded")
        print(f"  iterations: {len(latencies)}")
        print(f"  min:        {min(latencies):.3f} ms")
        print(f"  mean:       {statistics.fmean(latencies):.3f} ms")
        print(f"  median:     {statistics.median(latencies):.3f} ms")
        print(f"  max:        {max(latencies):.3f} ms")

        cpu_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        reference_outputs = cpu_session.run(None, feeds)
        parity_ok = _compare_outputs(output_names, reference_outputs, outputs)

        profile_path = Path(session.end_profiling())
        provider_counts, assignments = _profile_summary(profile_path)
        print("\n[Proof] Profiled kernel events by execution provider:")
        for provider, count in provider_counts.most_common():
            print(f"  {provider}: {count}")
            for op_name, node_name in sorted(assignments.get(provider, set())):
                print(f"    - {op_name}: {node_name}")
        webgpu_event_count = provider_counts.get(webgpu_ep.get_ep_name(), 0)
        webgpu_compute_nodes = {
            (op_name, node_name)
            for op_name, node_name in assignments.get(webgpu_ep.get_ep_name(), set())
            if op_name not in {"MemcpyFromHost", "MemcpyToHost"}
        }
        used_webgpu = bool(webgpu_compute_nodes)
        print(
            f"  {'PASS' if used_webgpu else 'FAIL'}: "
            f"{webgpu_event_count} event(s), including "
            f"{len(webgpu_compute_nodes)} unique compute node(s), ran on "
            f"{webgpu_ep.get_ep_name()}."
        )
        cpu_event_count = provider_counts.get("CPUExecutionProvider", 0)
        if cpu_event_count:
            print(
                f"  NOTE: {cpu_event_count} profiled event(s) ran on CPU. "
                "This is mixed-provider inference, not an all-GPU claim."
            )

        if args.keep_profile:
            kept_path = SCRIPT_DIR / profile_path.name
            kept_path.write_bytes(profile_path.read_bytes())
            print(f"[Profile] Kept: {kept_path}")

        assignment_ok = args.allow_cpu_fallback or cpu_event_count == 0
        passed = parity_ok and used_webgpu and assignment_ok
        print("\n" + "=" * 72)
        print("PASS: native WebGPU plugin inference is working." if passed else "FAIL: see diagnostics above.")
        print("=" * 72)
        return 0 if passed else 4

    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Check the matching English/Chinese README troubleshooting table, "
            "the GPU driver, wheel architecture, and selected device index.",
            file=sys.stderr,
        )
        return 5
    finally:
        # Plugin libraries must remain registered until every session that uses them
        # has been destroyed. Explicit destruction avoids shutdown-order problems.
        if session is not None:
            del session
        if cpu_session is not None:
            del cpu_session
        gc.collect()

        if profile_path is not None and profile_path.exists():
            try:
                profile_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"WARNING: profile cleanup failed: {error}", file=sys.stderr)
        if temporary_profile_dir is not None:
            try:
                temporary_profile_dir.cleanup()
            except OSError as error:
                print(
                    f"WARNING: temporary profile directory cleanup failed: {error}",
                    file=sys.stderr,
                )

        if registered:
            try:
                ort.unregister_execution_provider_library(REGISTRATION_NAME)
            except Exception as error:
                print(f"WARNING: plugin unregister failed during cleanup: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
