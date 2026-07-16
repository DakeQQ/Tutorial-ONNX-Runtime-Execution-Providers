#!/usr/bin/env python3
"""One-click Qualcomm QNN CPU/GPU/HTP execution-provider proof test.

The launcher creates an isolated environment, installs the pinned plugin-QNN
stack, generates deterministic static FP32 and QDQ models, and starts a worker
that proves graph assignment without allowing ONNX Runtime CPU fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import venv
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENV = SCRIPT_DIR / ".venv-qnn"
DEFAULT_ARTIFACTS = SCRIPT_DIR / ".qnn-smoke"
REQUIREMENTS = SCRIPT_DIR / "requirements.txt"
EXPECTED_VERSIONS = {
    "onnx": "1.21.0",
    "onnxruntime": "1.26.0",
    "onnxruntime-qnn": "2.4.0",
    "sympy": "1.14.0",
}
QNN_EP_NAME = "QNNExecutionProvider"
BACKEND_ALIASES = {"npu": "htp"}


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _validate_python_host() -> None:
    if platform.python_implementation() != "CPython":
        raise RuntimeError("The pinned QNN wheels require CPython.")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Use 64-bit CPython 3.11, 3.12, 3.13, or 3.14.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Use a 64-bit Python installation.")
    if platform.system() == "Windows":
        machine = platform.machine().lower()
        if machine not in {"arm64", "aarch64", "amd64", "x86_64"}:
            raise RuntimeError(f"Unsupported Windows process architecture: {machine}")


def _validate_local_execution_host(backend: str) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows" and machine in {"amd64", "x86_64"} and backend in {
        "gpu",
        "htp",
    }:
        raise RuntimeError(
            "Windows x64 QNN packages are for model preparation/AOT compilation, not "
            "local Adreno/HTP execution. Run this inference demo with native ARM64 "
            "Python on a Snapdragon Windows-on-Arm PC."
        )
    if system == "Darwin":
        raise RuntimeError("The local QNN inference demo supports Windows and Linux, not macOS.")


def _environment_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    expected = repr(EXPECTED_VERSIONS)
    check = (
        "import importlib.metadata as m, platform, sys; "
        f"expected={expected}; "
        "actual={name:m.version(name) for name in expected}; "
        "raise SystemExit(not (actual == expected and "
        "platform.python_implementation() == 'CPython' and "
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
    _validate_python_host()
    python = _venv_python(venv_dir)
    if python.is_file() and not refresh and _environment_ready(python):
        print(f"[1/3] Reusing pinned environment: {venv_dir}")
        return python

    if not python.is_file():
        print(f"[1/3] Creating isolated environment: {venv_dir}")
        try:
            venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
        except Exception as exc:
            raise RuntimeError(
                "Could not create the virtual environment. On Debian/Ubuntu install "
                "the matching python3-venv package."
            ) from exc
    else:
        print(f"[1/3] Refreshing isolated environment: {venv_dir}")

    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing requirements file: {REQUIREMENTS}")

    print("[2/3] Installing pinned ONNX Runtime + QNN plugin packages...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    install_command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-r",
        str(REQUIREMENTS),
    ]
    if refresh:
        install_command.insert(-2, "--force-reinstall")
    subprocess.run(install_command, check=True)
    if not _environment_ready(python):
        raise RuntimeError("The isolated environment does not match the pinned QNN stack.")
    return python


def _candidate_sdk_roots(explicit: Path | None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(explicit.expanduser().resolve())
    for variable in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "QNN_SDK_PATH"):
        value = os.environ.get(variable)
        if value:
            root = Path(value).expanduser().resolve()
            if root not in roots:
                roots.append(root)
    return roots


def _find_sdk_backend(root: Path, filename: str) -> Path | None:
    if root.is_file():
        return root if root.name.lower() == filename.lower() else None
    if not root.is_dir():
        return None

    preferred_abis = (
        "aarch64-windows-msvc",
        "arm64x-windows-msvc",
        "x86_64-windows-msvc",
        "aarch64-oe-linux-gcc11.2",
        "aarch64-linux-gcc11.2",
        "x86_64-linux-clang",
    )
    for abi in preferred_abis:
        candidate = root / "lib" / abi / filename
        if candidate.is_file():
            return candidate.resolve()
    for candidate in sorted((root / "lib").glob(f"**/{filename}")):
        if "android" not in str(candidate.parent).lower():
            return candidate.resolve()
    return None


def _resolve_backend_path(
    backend: str,
    qnn_ep: Any,
    explicit_backend: Path | None,
    qnn_sdk: Path | None,
) -> Path:
    if explicit_backend is not None:
        candidate = explicit_backend.expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError(f"Backend library does not exist: {candidate}")
        return candidate

    helper = {
        "cpu": qnn_ep.get_qnn_cpu_path,
        "gpu": qnn_ep.get_qnn_gpu_path,
        "htp": qnn_ep.get_qnn_htp_path,
    }[backend]
    packaged_candidate = Path(helper()).resolve()
    if packaged_candidate.is_file():
        return packaged_candidate

    filename = {
        "cpu": "QnnCpu.dll" if os.name == "nt" else "libQnnCpu.so",
        "gpu": "QnnGpu.dll" if os.name == "nt" else "libQnnGpu.so",
        "htp": "QnnHtp.dll" if os.name == "nt" else "libQnnHtp.so",
    }[backend]
    for root in _candidate_sdk_roots(qnn_sdk):
        candidate = _find_sdk_backend(root, filename)
        if candidate is not None:
            return candidate

    extra = ""
    if backend == "cpu":
        extra = (
            " The 2.4.0 release intentionally does not bundle the QNN CPU reference "
            "backend. Install QAIRT 2.48.40 with Qualcomm Package Manager and pass "
            "--qnn-sdk <SDK-root> or --backend-path <QnnCpu library>."
        )
    raise RuntimeError(
        f"Could not find {filename}. Packaged candidate: {packaged_candidate}.{extra}"
    )


def _describe_device(device: Any) -> str:
    hardware = getattr(device, "device", None)
    device_type = getattr(hardware, "type", "unknown")
    vendor = getattr(hardware, "vendor", "unknown")
    device_id = getattr(hardware, "device_id", getattr(hardware, "id", "unknown"))
    return (
        f"ep={getattr(device, 'ep_name', 'unknown')}, type={device_type}, "
        f"vendor={vendor}, id={device_id}"
    )


def _select_device(ort: Any, backend: str) -> tuple[Any, list[Any]]:
    all_devices = list(ort.get_ep_devices())
    qnn_devices = [device for device in all_devices if device.ep_name == QNN_EP_NAME]
    print("QNN EP devices:")
    for device in qnn_devices:
        print(f"  - {_describe_device(device)}")
    if not qnn_devices:
        raise RuntimeError(
            "The QNN plugin loaded but exposed no devices. Update the Qualcomm/OEM "
            "driver, use a supported Snapdragon device, and confirm native ARM64 Python."
        )

    expected_type = {
        "cpu": ort.OrtHardwareDeviceType.CPU,
        "gpu": ort.OrtHardwareDeviceType.GPU,
        "htp": ort.OrtHardwareDeviceType.NPU,
    }[backend]
    matches = [device for device in qnn_devices if device.device.type == expected_type]
    machine = platform.machine().lower()
    if (
        not matches
        and backend == "htp"
        and platform.system() == "Linux"
        and machine in {"amd64", "x86_64"}
    ):
        # The QNN Linux x64 HTP simulator is intentionally advertised through
        # a CPU-class OrtHardwareDevice; it compiles and executes QNN HTP graphs
        # without claiming local Snapdragon NPU hardware.
        matches = [
            device
            for device in qnn_devices
            if device.device.type == ort.OrtHardwareDeviceType.CPU
        ]
        if matches:
            print("QNN device mode   : Linux x64 HTP simulator (not local NPU hardware)")
    if not matches:
        raise RuntimeError(
            f"QNN exposed no {expected_type} device for backend {backend!r}. "
            f"Observed: {[ _describe_device(device) for device in qnn_devices ]}"
        )
    return matches[0], all_devices


def _session_options(ort: Any, profile_prefix: Path | None = None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 2
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    return options


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


def _verify_execution(
    assignment_counts: Counter[str],
    profile_counts: Counter[str],
) -> int:
    qnn_assigned = sum(
        count for name, count in assignment_counts.items() if "qnn" in name.lower()
    )
    qnn_profiled = sum(
        count for name, count in profile_counts.items() if "qnn" in name.lower()
    )
    cpu_profiled = sum(
        count
        for name, count in profile_counts.items()
        if name == "CPUExecutionProvider"
    )
    if qnn_assigned == 0 and qnn_profiled == 0:
        raise RuntimeError(
            "The session ran, but neither graph assignment nor profiling proved QNN "
            f"execution. Assignment={dict(assignment_counts)}, profile={dict(profile_counts)}"
        )
    if cpu_profiled:
        raise RuntimeError(
            "The strict target session unexpectedly profiled ONNX Runtime CPU nodes: "
            f"{dict(profile_counts)}"
        )
    return max(qnn_assigned, qnn_profiled)


def _worker(args: argparse.Namespace) -> int:
    backend = BACKEND_ALIASES.get(args.backend, args.backend)
    _validate_local_execution_host(backend)
    if backend == "cpu":
        # Plugin QNN 2.4 hides the ARM64 CPU reference device by default. This
        # must be set before importing/registering the plugin and creating OrtEnv.
        os.environ.setdefault("ORT_QNN_ENABLE_CPU_BACKEND", "1")

    import numpy as np
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    from smoke_model import deterministic_input, generate_smoke_models

    print("=" * 76)
    print("ONNX Runtime + Qualcomm QNN EP strict proof test")
    print("=" * 76)
    print(f"OS / process       : {platform.platform()} / {platform.machine()}")
    print(f"Python             : {platform.python_version()} ({sys.executable})")
    print(f"ONNX Runtime       : {ort.__version__}")
    print(f"QNN plugin         : {qnn_ep.__version__}")
    print(f"Requested backend  : {backend} ({'NPU/HTP' if backend == 'htp' else backend.upper()})")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    float_model, qdq_model = generate_smoke_models(args.artifacts_dir / "models")
    target_model = qdq_model if backend == "htp" else float_model
    print(f"Model              : {target_model.name} (static shapes)")

    backend_path = _resolve_backend_path(
        backend,
        qnn_ep,
        args.backend_path,
        args.qnn_sdk,
    )
    print(f"Backend library    : {backend_path}")

    target_session = None
    reference_session = None
    plugin_registered = False
    profile_ended = False
    pending_error: BaseException | None = None
    with tempfile.TemporaryDirectory(prefix="ort-qnn-profile-") as temp_dir:
        try:
            reference_session = ort.InferenceSession(
                str(target_model),
                sess_options=_session_options(ort),
                providers=["CPUExecutionProvider"],
            )
            input_data = deterministic_input(np)
            feeds = {reference_session.get_inputs()[0].name: input_data}
            reference, reference_median, _ = _run_timed(
                reference_session,
                feeds,
                args.warmups,
                args.runs,
            )

            ort.register_execution_provider_library(
                QNN_EP_NAME,
                qnn_ep.get_library_path(),
            )
            plugin_registered = True
            selected_device, _ = _select_device(ort, backend)

            profile_prefix = Path(temp_dir) / f"qnn-{backend}"
            options = _session_options(ort, profile_prefix)
            provider_options = {
                "backend_path": str(backend_path),
                # Keep graph I/O Q/DQ on QNN so strict HTP mode does not need
                # ONNX Runtime's CPU EP for the generated QDQ model.
                "offload_graph_io_quantization": "0",
            }
            if backend == "htp":
                provider_options.update(
                    {
                        "htp_performance_mode": args.performance_mode,
                        "htp_graph_finalization_optimization_mode": "3",
                    }
                )
            options.add_provider_for_devices([selected_device], provider_options)
            if not options.has_providers():
                raise RuntimeError("QNN device was not added to SessionOptions.")

            target_session = ort.InferenceSession(
                str(target_model),
                sess_options=options,
            )
            assignment_counts = _assignment_provider_counts(target_session)
            target, target_median, target_mean = _run_timed(
                target_session,
                feeds,
                args.warmups,
                args.runs,
            )
            profile_path = target_session.end_profiling()
            profile_ended = True
            profile_counts = _profile_provider_counts(profile_path)
            proof_count = _verify_execution(assignment_counts, profile_counts)

            tolerance = 8e-2 if backend == "htp" else 3e-3
            np.testing.assert_allclose(
                target,
                reference,
                rtol=tolerance,
                atol=tolerance,
            )
            max_error = float(np.max(np.abs(target - reference)))

            print(f"Session providers   : {target_session.get_providers()}")
            print(f"Graph assignment    : {dict(assignment_counts)}")
            print(f"Profiled providers  : {dict(profile_counts)}")
            print(f"CPU reference median: {reference_median:.3f} ms")
            print(f"QNN median / mean   : {target_median:.3f} / {target_mean:.3f} ms")
            print(f"Max |QNN-reference| : {max_error:.8g} (limit {tolerance:g})")
            print(
                f"\nPASS: QNN {backend.upper()} executed {proof_count} assigned/profiled "
                "node event(s) with ORT CPU fallback disabled."
            )
            print("Note: this tiny graph validates configuration; it is not a hardware benchmark.")
            return 0
        except BaseException as exc:
            pending_error = exc
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if target_session is not None and not profile_ended:
                try:
                    target_session.end_profiling()
                except Exception:
                    pass
            del target_session
            del reference_session
            gc.collect()
            if plugin_registered:
                try:
                    ort.unregister_execution_provider_library(QNN_EP_NAME)
                except Exception as cleanup_error:
                    if pending_error is None:
                        raise
                    print(
                        f"WARNING: QNN plugin cleanup failed: {cleanup_error}",
                        file=sys.stderr,
                    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap and prove local execution through Qualcomm QNN CPU, GPU, or HTP/NPU."
        )
    )
    parser.add_argument(
        "backend",
        nargs="?",
        choices=("cpu", "gpu", "htp", "npu"),
        default="htp",
        help="QNN backend to prove (default: htp; npu is an alias)",
    )
    parser.add_argument("--warmups", type=int, default=5, help="warm-up runs")
    parser.add_argument("--runs", type=int, default=30, help="measured runs")
    parser.add_argument(
        "--performance-mode",
        choices=(
            "burst",
            "balanced",
            "default",
            "high_performance",
            "high_power_saver",
            "low_balanced",
            "low_power_saver",
            "power_saver",
            "sustained_high_performance",
        ),
        default="burst",
        help="HTP power/performance policy (default: burst)",
    )
    parser.add_argument(
        "--qnn-sdk",
        type=Path,
        help="QAIRT SDK root; normally needed only for the QNN CPU backend",
    )
    parser.add_argument(
        "--backend-path",
        type=Path,
        help="explicit QnnCpu/QnnGpu/QnnHtp library path",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help=f"isolated environment directory (default: {DEFAULT_VENV.name})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"generated smoke model directory (default: {DEFAULT_ARTIFACTS.name})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="force reinstall the pinned environment",
    )
    parser.add_argument("--verbose", action="store_true", help="show failure traceback")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be >= 0 and --runs must be >= 1")
    args.venv = args.venv.expanduser().resolve()
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()

    try:
        if args.worker:
            print("[3/3] Running the strict QNN execution proof...")
            return _worker(args)

        backend = BACKEND_ALIASES.get(args.backend, args.backend)
        _validate_local_execution_host(backend)
        python = _ensure_environment(args.venv, args.refresh)
        command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
        command = [item for item in command if item != "--refresh"]
        command.append("--worker")
        result = subprocess.run(command, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        print(
            "Read the matching English/Chinese troubleshooting table. Do not treat "
            "QNNExecutionProvider appearing in a list as proof of acceleration.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
