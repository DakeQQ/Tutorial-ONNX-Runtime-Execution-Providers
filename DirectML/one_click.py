#!/usr/bin/env python3
"""One-click ONNX Runtime DirectML and Windows ML proof test.

The launcher creates a route-specific virtual environment, installs a pinned
stack, generates a deterministic ONNX model, and rejects CPU fallback using
both graph-assignment records and a current-run ONNX Runtime profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Iterable
import unittest
import venv


ROOT = Path(__file__).resolve().parent
VENV_BY_ROUTE = {
    "directml": ROOT / ".venv-directml",
    "windowsml": ROOT / ".venv-windowsml",
}
REQUIREMENTS_BY_ROUTE = {
    "directml": ROOT / "requirements-directml.txt",
    "windowsml": ROOT / "requirements-winml.txt",
}
EXPECTED_STACKS = {
    "directml": {
        "numpy": "1.26.4",
        "onnx": "1.22.0",
        "onnxruntime-directml": "1.24.4",
    },
    "windowsml": {
        "numpy": "2.4.6",
        "onnx": "1.22.0",
        "onnxruntime-windowsml": "1.24.6.202605042033",
        "wasdk-microsoft-windows-ai-machinelearning": "2.1.3",
        "wasdk-microsoft-windows-applicationmodel-dynamicdependency-bootstrap": "2.1.3",
    },
}
KNOWN_ORT_DISTRIBUTIONS = {
    "onnxruntime",
    "onnxruntime-directml",
    "onnxruntime-gpu",
    "onnxruntime-migraphx",
    "onnxruntime-openvino",
    "onnxruntime-qnn",
    "onnxruntime-rocm",
    "onnxruntime-training",
    "onnxruntime-vitisai",
    "onnxruntime-windowsml",
}
POLICY_NAMES = {
    "default": "DEFAULT",
    "prefer-cpu": "PREFER_CPU",
    "prefer-npu": "PREFER_NPU",
    "prefer-gpu": "PREFER_GPU",
    "max-performance": "MAX_PERFORMANCE",
    "max-efficiency": "MAX_EFFICIENCY",
    "min-power": "MIN_OVERALL_POWER",
}
DIRECTML_EP = "DmlExecutionProvider"
CPU_EP = "CPUExecutionProvider"


def canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distributions() -> dict[str, str]:
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions[canonical_distribution_name(name)] = distribution.version
    return distributions


def validate_host(route: str) -> None:
    if platform.system() != "Windows":
        raise RuntimeError(f"The {route} route requires native Windows; current OS is {platform.system()}.")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use 64-bit CPython 3.12 for the pinned tutorial stacks.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Use a 64-bit Python process.")
    if "windowsapps" in str(Path(sys.executable)).lower():
        raise RuntimeError("Use Python 3.12 from python.org or winget, not the Microsoft Store alias.")

    machine = platform.machine().lower()
    if route == "directml" and machine not in {"amd64", "x86_64"}:
        raise RuntimeError("The PyPI onnxruntime-directml wheel is available for Windows x64 only.")
    if route == "windowsml" and machine not in {"amd64", "x86_64", "arm64", "aarch64"}:
        raise RuntimeError(f"Windows ML requires an x64 or ARM64 process; found {machine}.")

    build = sys.getwindowsversion().build
    minimum = 18362 if route == "directml" else 26100
    if build < minimum:
        requirement = "Windows 10 1903" if route == "directml" else "Windows 11 24H2"
        raise RuntimeError(f"The {route} route requires {requirement} (build {minimum}) or newer; found {build}.")


def validate_arguments(args: argparse.Namespace) -> None:
    if args.runs < 1 or args.warmups < 0 or args.device_id < 0:
        raise RuntimeError("--runs must be positive; --warmups and --device-id must be non-negative.")
    if args.route == "directml" and (args.provider or args.allow_download):
        raise RuntimeError("--provider and --allow-download apply only to the windowsml route.")
    if args.route == "directml" and args.policy is not None:
        raise RuntimeError("--policy applies only to the windowsml route.")
    if args.route == "windowsml" and args.device_id != 0:
        raise RuntimeError(
            "--device-id applies only to standalone DirectML; Windows ML selects an OrtEpDevice by policy."
        )
    if args.route == "windowsml" and args.policy is None:
        args.policy = "max-performance"


def validate_worker_stack(route: str) -> None:
    distributions = installed_distributions()
    expected = EXPECTED_STACKS[route]
    mismatches = {
        name: (version, distributions.get(name))
        for name, version in expected.items()
        if distributions.get(name) != version
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {expected_version}, found {actual or 'missing'}"
            for name, (expected_version, actual) in mismatches.items()
        )
        raise RuntimeError(f"The isolated {route} environment is not pinned correctly ({details}).")

    expected_ort = "onnxruntime-directml" if route == "directml" else "onnxruntime-windowsml"
    installed_ort = set(distributions).intersection(KNOWN_ORT_DISTRIBUTIONS)
    if installed_ort != {expected_ort}:
        raise RuntimeError(
            f"Install exactly {expected_ort}; found ONNX Runtime distributions "
            f"{sorted(installed_ort) or ['none']}."
        )

    owners = {
        canonical_distribution_name(name)
        for name in importlib.metadata.packages_distributions().get("onnxruntime", [])
    }
    if owners != {expected_ort}:
        raise RuntimeError(
            f"The onnxruntime import must be owned only by {expected_ort}; found {sorted(owners) or ['none']}."
        )


def enumerate_dxgi_adapters() -> list[dict[str, Any]]:
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

    class DXGI_ADAPTER_DESC(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", ctypes.c_uint32),
            ("DeviceId", ctypes.c_uint32),
            ("SubSysId", ctypes.c_uint32),
            ("Revision", ctypes.c_uint32),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", LUID),
        ]

    iid_factory1 = GUID(
        0x770AAE78,
        0xF26F,
        0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    factory = ctypes.c_void_p()
    create_factory = ctypes.windll.dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long
    result = create_factory(ctypes.byref(iid_factory1), ctypes.byref(factory))
    if result < 0:
        raise OSError(f"CreateDXGIFactory1 failed with HRESULT 0x{result & 0xFFFFFFFF:08X}")

    def com_method(pointer: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
        vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])

    release_factory = com_method(factory, 2, ctypes.c_ulong)
    enum_adapters = com_method(
        factory,
        7,
        ctypes.c_long,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    not_found = ctypes.c_long(0x887A0002).value
    adapters: list[dict[str, Any]] = []
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            result = enum_adapters(factory, index, ctypes.byref(adapter))
            if result == not_found:
                break
            if result < 0:
                raise OSError(
                    f"IDXGIFactory::EnumAdapters failed at {index}: "
                    f"0x{result & 0xFFFFFFFF:08X}"
                )
            get_desc = com_method(adapter, 8, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC))
            release_adapter = com_method(adapter, 2, ctypes.c_ulong)
            try:
                description = DXGI_ADAPTER_DESC()
                result = get_desc(adapter, ctypes.byref(description))
                if result < 0:
                    raise OSError(
                        f"IDXGIAdapter::GetDesc failed with HRESULT "
                        f"0x{result & 0xFFFFFFFF:08X}"
                    )
                adapters.append(
                    {
                        "index": index,
                        "name": description.Description.rstrip("\x00"),
                        "vendor_id": int(description.VendorId),
                        "device_id": int(description.DeviceId),
                        "dedicated_video_memory": int(description.DedicatedVideoMemory),
                    }
                )
            finally:
                release_adapter(adapter)
            index += 1
    finally:
        release_factory(factory)
    return adapters


def print_directml_adapters(device_id: int) -> None:
    try:
        adapters = enumerate_dxgi_adapters()
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"Could not enumerate DirectML DXGI adapters: {exc}") from exc
    if device_id >= len(adapters):
        raise RuntimeError(
            f"DirectML device_id={device_id} does not exist; enumerated {len(adapters)} adapter(s)."
        )

    print("DXGI adapters:")
    for adapter in adapters:
        selected = " [selected]" if adapter["index"] == device_id else ""
        print(
            f"  - {adapter['index']}: {adapter['name']}, "
            f"vendor=0x{adapter['vendor_id']:04X}, device=0x{adapter['device_id']:04X}, "
            f"dedicated={adapter['dedicated_video_memory'] / (1024**2):.0f} MiB{selected}"
        )


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def environment_ready(route: str, python: Path) -> bool:
    if not python.is_file():
        return False
    expected = repr(EXPECTED_STACKS[route])
    known = repr(KNOWN_ORT_DISTRIBUTIONS)
    expected_ort = repr("onnxruntime-directml" if route == "directml" else "onnxruntime-windowsml")
    check = f"""
import importlib.metadata as metadata
import re

canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
distributions = {{}}
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        distributions[canonical(name)] = distribution.version
expected = {expected}
known = {known}
expected_ort = {expected_ort}
versions_match = all(distributions.get(name) == version for name, version in expected.items())
ort_match = set(distributions).intersection(known) == {{expected_ort}}
owners = {{canonical(name) for name in metadata.packages_distributions().get("onnxruntime", [])}}
raise SystemExit(0 if versions_match and ort_match and owners == {{expected_ort}} else 1)
"""
    try:
        result = subprocess.run(
            [str(python), "-c", check],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def ensure_environment(route: str, refresh: bool) -> Path:
    venv_dir = VENV_BY_ROUTE[route]
    python = venv_python(venv_dir)
    if not refresh and environment_ready(route, python):
        dependency_check = subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if dependency_check.returncode == 0:
            print(f"[1/3] Reusing pinned environment: {venv_dir}")
            return python

    requirements = REQUIREMENTS_BY_ROUTE[route]
    if not requirements.is_file():
        raise RuntimeError(f"Missing requirements file: {requirements}")

    action = "Rebuilding" if venv_dir.exists() else "Creating"
    print(f"[1/3] {action} isolated environment: {venv_dir}")
    try:
        venv.EnvBuilder(with_pip=True, clear=venv_dir.exists()).create(venv_dir)
    except Exception as exc:
        raise RuntimeError(f"Could not create {venv_dir}: {exc}") from exc

    print(f"[2/3] Installing the pinned {route} stack...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip==26.1.2"],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--requirement", str(requirements)],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    if not environment_ready(route, python):
        raise RuntimeError(f"The new {route} environment does not match the pinned stack.")
    return python


def launch_worker(args: argparse.Namespace, python: Path) -> int:
    command = [
        str(python),
        str(Path(__file__).resolve()),
        args.route,
        "--worker",
        "--device-id",
        str(args.device_id),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
    ]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.policy:
        command.extend(["--policy", args.policy])
    if args.allow_download:
        command.append("--allow-download")
    print(f"[3/3] Running strict {args.route} proof test...")
    return subprocess.run(command, check=False).returncode


def build_model(path: Path, onnx: Any, np: Any) -> None:
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(20260717)
    input_size = 256
    hidden_size = 512
    output_size = 128
    weight_1 = (rng.standard_normal((input_size, hidden_size)) * 0.025).astype(np.float32)
    bias_1 = (rng.standard_normal(hidden_size) * 0.01).astype(np.float32)
    weight_2 = (rng.standard_normal((hidden_size, output_size)) * 0.025).astype(np.float32)
    bias_2 = (rng.standard_normal(output_size) * 0.01).astype(np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight_1"], ["hidden_mm"], name="hidden_matmul"),
            helper.make_node("Add", ["hidden_mm", "bias_1"], ["hidden_add"], name="hidden_bias"),
            helper.make_node("Relu", ["hidden_add"], ["hidden"], name="hidden_relu"),
            helper.make_node("MatMul", ["hidden", "weight_2"], ["output_mm"], name="output_matmul"),
            helper.make_node("Add", ["output_mm", "bias_2"], ["output"], name="output_bias"),
        ],
        "directml_windowsml_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size])],
        [
            numpy_helper.from_array(weight_1, "weight_1"),
            numpy_helper.from_array(bias_1, "bias_1"),
            numpy_helper.from_array(weight_2, "weight_2"),
            numpy_helper.from_array(bias_2, "bias_2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="directml-windowsml-one-click",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def session_options(ort: Any, profile_prefix: Path) -> Any:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    return options


def profile_provider_counts(profile_path: Path) -> Counter[str]:
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse ONNX Runtime profile {profile_path}: {exc}") from exc
    if not isinstance(events, list):
        raise RuntimeError(f"ONNX Runtime profile is not an event list: {profile_path}")

    counts: Counter[str] = Counter()
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        arguments = event.get("args")
        provider = arguments.get("provider") if isinstance(arguments, dict) else None
        if isinstance(provider, str) and provider:
            counts[provider] += 1
    return counts


def assignment_provider_counts(session: Any) -> Counter[str]:
    assignments = session.get_provider_graph_assignment_info()
    if not assignments:
        raise RuntimeError("ONNX Runtime returned no graph-assignment records.")
    counts: Counter[str] = Counter()
    for assignment in assignments:
        nodes = list(assignment.get_nodes())
        counts[str(assignment.ep_name)] += max(1, len(nodes))
    return counts


def verify_assignments(
    route: str,
    counts: Counter[str],
    registered_windows_ml_eps: Iterable[str] = (),
) -> set[str]:
    if counts.get(CPU_EP, 0):
        raise RuntimeError(f"CPU fallback was recorded in graph assignment: {dict(counts)}")
    assigned = {name for name, count in counts.items() if count > 0 and name != CPU_EP}
    if route == "directml":
        if assigned != {DIRECTML_EP}:
            raise RuntimeError(f"Expected only {DIRECTML_EP} assignment; found {dict(counts)}")
        return assigned

    registered = set(registered_windows_ml_eps)
    if not assigned or not assigned.issubset(registered):
        raise RuntimeError(
            "Windows ML selected an unregistered or non-accelerator provider: "
            f"assignment={dict(counts)}, registered={sorted(registered)}"
        )
    return assigned


def enum_label(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).lower().replace("_", " ")


def prepare_windows_ml_providers(
    ort: Any,
    provider_name: str | None,
    allow_download: bool,
    winml_module: Any | None = None,
) -> tuple[list[str], list[str]]:
    if winml_module is None:
        try:
            import winui3.microsoft.windows.ai.machinelearning as winml
        except Exception as exc:
            raise RuntimeError(f"Could not import the Windows ML projection: {exc}") from exc
    else:
        winml = winml_module

    catalog = winml.ExecutionProviderCatalog.get_default()
    providers = list(catalog.find_all_providers())
    if not providers:
        raise RuntimeError("Windows ML returned an empty execution-provider catalog.")

    print("Windows ML catalog:")
    for provider in providers:
        state = getattr(provider, "ready_state", "unknown")
        print(f"  - {provider.name}: {enum_label(state)}")

    existing_ep_names = {device.ep_name for device in ort.get_ep_devices()}
    if provider_name:
        candidates = [provider for provider in providers if provider.name == provider_name]
        if not candidates:
            raise RuntimeError(
                f"Catalog provider {provider_name!r} was not found; available names are "
                f"{[provider.name for provider in providers]}."
            )
        state = enum_label(getattr(candidates[0], "ready_state", "unknown"))
        if not allow_download and "not present" in state and provider_name not in existing_ep_names:
            raise RuntimeError(
                f"Catalog provider {provider_name!r} is NotPresent; rerun with --allow-download "
                "only if provider acquisition is permitted."
            )
    else:
        candidates = []
        for provider in providers:
            state = enum_label(getattr(provider, "ready_state", "unknown"))
            if allow_download or "not present" not in state or provider.name in existing_ep_names:
                candidates.append(provider)

    certified_value = getattr(
        getattr(winml, "ExecutionProviderCertification", None),
        "CERTIFIED",
        None,
    )
    if certified_value is None:
        raise RuntimeError("The pinned Windows ML projection exposes no Certified provider identity.")

    prepared: list[str] = []
    dynamically_registered: list[str] = []
    try:
        for provider in candidates:
            if getattr(provider, "certification", None) != certified_value:
                print(f"Skipping non-certified provider: {provider.name}")
                continue
            try:
                existing_devices = {device.ep_name for device in ort.get_ep_devices()}
                if provider.name in existing_devices:
                    prepared.append(provider.name)
                    print(f"Catalog provider already available to ORT: {provider.name}")
                    continue

                state = enum_label(getattr(provider, "ready_state", "unknown"))
                if not allow_download and "not present" in state:
                    raise RuntimeError("provider is NotPresent; rerun with --allow-download to acquire it")

                result = provider.ensure_ready_async().get()
                status = getattr(result, "status", None)
                if status is not None and "success" not in enum_label(status):
                    diagnostic = getattr(result, "diagnostic_text", "")
                    raise RuntimeError(f"EnsureReady returned {enum_label(status)}: {diagnostic}")

                ready_devices = {device.ep_name for device in ort.get_ep_devices()}
                if provider.name in ready_devices:
                    prepared.append(provider.name)
                    print(f"Catalog provider already available to ORT: {provider.name}")
                    continue

                raw_library_path = str(provider.library_path)
                if not raw_library_path:
                    raise RuntimeError("catalog returned no library path and ORT exposes no matching device")
                library_path = Path(raw_library_path)
                if not library_path.is_file():
                    raise RuntimeError(f"catalog library path is not a file: {library_path}")

                ort.register_execution_provider_library(provider.name, str(library_path))
                dynamically_registered.append(provider.name)
                registered_devices = {device.ep_name for device in ort.get_ep_devices()}
                if provider.name not in registered_devices:
                    raise RuntimeError("library registered but exposed no matching OrtEpDevice")
                prepared.append(provider.name)
                print(f"Registered catalog provider: {provider.name} ({library_path})")
            except Exception as exc:
                if provider_name or provider.name == DIRECTML_EP:
                    raise RuntimeError(f"Could not prepare {provider.name}: {exc}") from exc
                if provider.name in dynamically_registered:
                    try:
                        ort.unregister_execution_provider_library(provider.name)
                    finally:
                        dynamically_registered.remove(provider.name)
                print(f"Skipping unavailable provider {provider.name}: {exc}", file=sys.stderr)

        if not prepared:
            suffix = "; rerun with --allow-download" if not allow_download else ""
            raise RuntimeError(f"No certified Windows ML provider was prepared{suffix}.")
        return prepared, dynamically_registered
    except Exception:
        for name in reversed(dynamically_registered):
            try:
                ort.unregister_execution_provider_library(name)
            except Exception:
                pass
        raise


def remove_winrt_runtime_cpp_library() -> None:
    """Apply the official Python sample workaround before importing Windows ML."""
    try:
        distribution = importlib.metadata.distribution("winrt-runtime")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("The Windows ML environment is missing winrt-runtime.") from exc
    dll_path = Path(distribution.locate_file("")) / "winrt" / "msvcp140.dll"
    if dll_path.is_file():
        dll_path.unlink()
        print(f"Removed WinRT's private C++ runtime copy: {dll_path}")


def benchmark(session: Any, feeds: dict[str, Any], warmups: int, runs: int) -> tuple[Any, float, float]:
    output = None
    for _ in range(warmups):
        output = session.run(None, feeds)[0]
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = session.run(None, feeds)[0]
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    if output is None:
        raise RuntimeError("Inference produced no output.")
    return output, statistics.median(samples), statistics.fmean(samples)


def run_smoke(
    args: argparse.Namespace,
    ort: Any,
    registered_windows_ml_eps: list[str] | None = None,
) -> int:
    import numpy as np
    import onnx

    print("=" * 76)
    print("ONNX Runtime DirectML / Windows ML strict proof test")
    print("=" * 76)
    print(f"Route              : {args.route}")
    print(f"OS / process       : {platform.platform()} / {platform.machine()}")
    print(f"Python             : {platform.python_version()} ({sys.executable})")
    print(f"ONNX Runtime       : {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    with tempfile.TemporaryDirectory(prefix=f"ort-{args.route}-proof-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        model_path = temp_dir / "directml_windowsml_smoke.onnx"
        build_model(model_path, onnx, np)
        data = np.random.default_rng(42).standard_normal((1, 256)).astype(np.float32)

        reference_session = ort.InferenceSession(str(model_path), providers=[CPU_EP])
        feeds = {reference_session.get_inputs()[0].name: data}
        reference = reference_session.run(None, feeds)[0]

        options = session_options(ort, temp_dir / "profile")
        if args.route == "directml":
            if DIRECTML_EP not in ort.get_available_providers():
                raise RuntimeError(
                    f"{DIRECTML_EP} is absent. Recreate the environment and update the graphics driver."
                )
            print_directml_adapters(args.device_id)
            providers: list[Any] | None = [(DIRECTML_EP, {"device_id": str(args.device_id)})]
            print(f"DirectML adapter ID : {args.device_id}")
        else:
            registered_windows_ml_eps = registered_windows_ml_eps or []
            policy_name = POLICY_NAMES[args.policy]
            policy = getattr(ort.OrtExecutionProviderDevicePolicy, policy_name)
            options.set_provider_selection_policy(policy)
            providers = None
            print(f"Selection policy    : {policy_name}")
            print(f"Registered catalog  : {registered_windows_ml_eps}")

        target_session = None
        profile_ended = False
        try:
            if providers is None:
                target_session = ort.InferenceSession(str(model_path), sess_options=options)
            else:
                target_session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=providers,
                )
            target_session.disable_fallback()
            assignments = assignment_provider_counts(target_session)
            assigned_providers = verify_assignments(
                args.route,
                assignments,
                registered_windows_ml_eps or [],
            )
            output, median_ms, mean_ms = benchmark(
                target_session,
                feeds,
                args.warmups,
                args.runs,
            )
            np.testing.assert_allclose(output, reference, rtol=1e-3, atol=1e-4)

            profile_path = Path(target_session.end_profiling())
            profile_ended = True
            profile_counts = profile_provider_counts(profile_path)
            profiled_target_events = sum(profile_counts.get(name, 0) for name in assigned_providers)
            if profile_counts.get(CPU_EP, 0):
                raise RuntimeError(f"CPU fallback appeared in the current-run profile: {dict(profile_counts)}")
            if profiled_target_events == 0:
                raise RuntimeError(
                    "Graph assignment named an accelerator, but the current-run profile contained no "
                    f"matching node event: assignment={dict(assignments)}, profile={dict(profile_counts)}"
                )

            print(f"Session providers   : {target_session.get_providers()}")
            print(f"Graph assignment    : {dict(assignments)}")
            print(f"Profiled providers  : {dict(profile_counts)}")
            print(f"Validation limits   : rtol=0.001, atol=0.0001")
            print(f"Median / mean       : {median_ms:.3f} / {mean_ms:.3f} ms ({args.runs} runs)")
            print(f"Max |target-CPU|    : {float(np.max(np.abs(output - reference))):.8g}")
            print(
                f"\nPASS: {', '.join(sorted(assigned_providers))} executed "
                f"{profiled_target_events} profiled node event(s) with ORT CPU fallback disabled."
            )
            print("Note: this tiny graph validates configuration; it is not a hardware benchmark.")
            return 0
        finally:
            if target_session is not None and not profile_ended:
                try:
                    target_session.end_profiling()
                except Exception:
                    pass


def run_worker(args: argparse.Namespace) -> int:
    validate_host(args.route)
    validate_worker_stack(args.route)
    if args.route == "directml":
        import onnxruntime as ort

        return run_smoke(args, ort)

    remove_winrt_runtime_cpp_library()
    try:
        import winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap as bootstrap
    except Exception as exc:
        raise RuntimeError(f"Could not import the Windows App SDK bootstrap projection: {exc}") from exc

    with bootstrap.initialize(options=bootstrap.InitializeOptions.ON_NO_MATCH_SHOW_UI):
        import onnxruntime as ort

        prepared, dynamically_registered = prepare_windows_ml_providers(
            ort,
            args.provider,
            args.allow_download,
        )
        try:
            return run_smoke(args, ort, prepared)
        finally:
            gc.collect()
            for name in reversed(dynamically_registered):
                try:
                    ort.unregister_execution_provider_library(name)
                except Exception:
                    pass


class HelperTests(unittest.TestCase):
    @staticmethod
    def fake_winml(provider: Any) -> Any:
        catalog = SimpleNamespace(find_all_providers=lambda: [provider])
        return SimpleNamespace(
            ExecutionProviderCatalog=SimpleNamespace(get_default=lambda: catalog),
            ExecutionProviderCertification=SimpleNamespace(CERTIFIED="certified"),
        )

    @staticmethod
    def ready_provider(name: str, library_path: str = "") -> Any:
        ready_result = SimpleNamespace(status="success")
        return SimpleNamespace(
            name=name,
            ready_state="ready",
            certification="certified",
            library_path=library_path,
            ensure_ready_async=lambda: SimpleNamespace(get=lambda: ready_result),
        )

    def test_distribution_name_normalization(self) -> None:
        self.assertEqual(
            canonical_distribution_name("wasdk_Microsoft.Windows-AI.MachineLearning"),
            "wasdk-microsoft-windows-ai-machinelearning",
        )

    def test_profile_parser_counts_node_providers_only(self) -> None:
        events = [
            {"cat": "Node", "args": {"provider": DIRECTML_EP}},
            {"cat": "Node", "args": {"provider": DIRECTML_EP}},
            {"cat": "Session", "args": {"provider": CPU_EP}},
            {"cat": "Node", "args": {}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            self.assertEqual(profile_provider_counts(path), Counter({DIRECTML_EP: 2}))

    def test_directml_assignment_must_be_exclusive(self) -> None:
        assigned = verify_assignments("directml", Counter({DIRECTML_EP: 5}))
        self.assertEqual(assigned, {DIRECTML_EP})
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            verify_assignments("directml", Counter({DIRECTML_EP: 4, CPU_EP: 1}))

    def test_windows_ml_assignment_must_match_registration(self) -> None:
        assigned = verify_assignments(
            "windowsml",
            Counter({"MIGraphXExecutionProvider": 1}),
            ["MIGraphXExecutionProvider", DIRECTML_EP],
        )
        self.assertEqual(assigned, {"MIGraphXExecutionProvider"})
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            verify_assignments(
                "windowsml",
                Counter({"UnexpectedExecutionProvider": 1}),
                [DIRECTML_EP],
            )

    def test_policy_mapping_covers_cli_choices(self) -> None:
        self.assertEqual(POLICY_NAMES["max-performance"], "MAX_PERFORMANCE")
        self.assertIn("min-power", POLICY_NAMES)

    def test_route_specific_arguments_fail_closed(self) -> None:
        directml_args = SimpleNamespace(
            route="directml",
            runs=1,
            warmups=0,
            device_id=0,
            provider=None,
            allow_download=True,
            policy=None,
        )
        with self.assertRaisesRegex(RuntimeError, "windowsml route"):
            validate_arguments(directml_args)
        directml_args.allow_download = False
        directml_args.policy = "prefer-gpu"
        with self.assertRaisesRegex(RuntimeError, "--policy"):
            validate_arguments(directml_args)

        windows_ml_args = SimpleNamespace(
            route="windowsml",
            runs=1,
            warmups=0,
            device_id=1,
            provider=None,
            allow_download=False,
            policy=None,
        )
        with self.assertRaisesRegex(RuntimeError, "standalone DirectML"):
            validate_arguments(windows_ml_args)

    def test_catalog_does_not_download_not_present_provider_without_consent(self) -> None:
        ensure_calls: list[str] = []
        provider = SimpleNamespace(
            name="VendorExecutionProvider",
            ready_state="not present",
            certification="certified",
            library_path="",
            ensure_ready_async=lambda: ensure_calls.append("called"),
        )
        ort = SimpleNamespace(get_ep_devices=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "No certified Windows ML provider"):
            prepare_windows_ml_providers(ort, None, False, self.fake_winml(provider))
        self.assertEqual(ensure_calls, [])

    def test_catalog_reuses_provider_already_exposed_by_ort(self) -> None:
        provider = self.ready_provider(DIRECTML_EP)
        provider.ready_state = "not present"
        ensure_calls: list[str] = []
        provider.ensure_ready_async = lambda: ensure_calls.append("called")
        register_calls: list[tuple[str, str]] = []
        ort = SimpleNamespace(
            get_ep_devices=lambda: [SimpleNamespace(ep_name=DIRECTML_EP)],
            register_execution_provider_library=lambda name, path: register_calls.append((name, path)),
        )
        prepared, dynamically_registered = prepare_windows_ml_providers(
            ort,
            None,
            False,
            self.fake_winml(provider),
        )
        self.assertEqual(prepared, [DIRECTML_EP])
        self.assertEqual(dynamically_registered, [])
        self.assertEqual(ensure_calls, [])
        self.assertEqual(register_calls, [])

    def test_catalog_cleans_registration_that_exposes_no_device(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            library_path = Path(temp_dir) / "vendor_ep.dll"
            library_path.touch()
            provider = self.ready_provider("VendorExecutionProvider", str(library_path))
            registered: list[str] = []
            unregistered: list[str] = []
            ort = SimpleNamespace(
                get_ep_devices=lambda: [],
                register_execution_provider_library=lambda name, path: registered.append(name),
                unregister_execution_provider_library=lambda name: unregistered.append(name),
            )
            with self.assertRaisesRegex(RuntimeError, "No certified Windows ML provider"):
                prepare_windows_ml_providers(ort, None, False, self.fake_winml(provider))
            self.assertEqual(registered, ["VendorExecutionProvider"])
            self.assertEqual(unregistered, ["VendorExecutionProvider"])


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HelperTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pinned environment and strictly prove DirectML or Windows ML execution."
    )
    parser.add_argument(
        "route",
        nargs="?",
        choices=("directml", "windowsml"),
        default="directml",
        help="directml (default) or windowsml",
    )
    parser.add_argument("--device-id", type=int, default=0, help="DirectML DXGI adapter index")
    parser.add_argument(
        "--policy",
        choices=tuple(POLICY_NAMES),
        default=None,
        help="Windows ML automatic EP policy (default: max-performance)",
    )
    parser.add_argument("--provider", help="Prepare only this exact Windows ML catalog provider")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Windows ML to acquire catalog providers that are not installed",
    )
    parser.add_argument("--warmups", type=int, default=3, help="warm-up runs (default: 3)")
    parser.add_argument("--runs", type=int, default=20, help="timed runs (default: 20)")
    parser.add_argument("--refresh", action="store_true", help="rebuild the route virtual environment")
    parser.add_argument("--self-test", action="store_true", help="run platform-neutral helper tests")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_tests()
    try:
        validate_arguments(args)
        if args.worker:
            return run_worker(args)
        validate_host(args.route)
        python = ensure_environment(args.route, args.refresh)
        return launch_worker(args, python)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Read the matching README troubleshooting table, repair the prerequisite, and rerun.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())