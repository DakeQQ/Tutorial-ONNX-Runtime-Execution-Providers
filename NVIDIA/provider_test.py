#!/usr/bin/env python3
"""Small, deterministic ONNX Runtime execution-provider smoke test.

The test generates one static FP32 ONNX model, computes a NumPy reference, runs the
requested provider, checks numerical parity, and parses an ORT profile to prove
that at least one node actually ran on the requested provider.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

PROVIDER_NAMES = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    # The 0.3.x wheel helper returns this recommended registration name. Plugin
    # registration names are application-defined and become the OrtEpDevice and
    # profile provider name for that registration.
    "nv_tensorrt_rtx": "nv_tensorrt_rtx",
}

INSTALL_HINTS = {
    "cuda": "python -m pip install -r NVIDIA/requirements-cuda.txt",
    "tensorrt": (
        "Python 3.11-3.13: python -m pip install "
        "-r NVIDIA/requirements-tensorrt.txt"
    ),
    "nv_tensorrt_rtx": (
        "python -m pip install -r NVIDIA/requirements-tensorrt-rtx.txt"
    ),
}


def _default_cache_root() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    base = os.environ.get("XDG_CACHE_HOME")
    return Path(base) if base else Path.home() / ".cache"


def _load_dependencies(provider: str) -> tuple[Any, Any, Any, Any]:
    # Importing TensorRT first makes pip-installed TensorRT shared libraries
    # visible before ONNX Runtime probes its TensorRT provider.
    tensorrt_module = None
    if provider == "tensorrt":
        try:
            tensorrt_module = importlib.import_module("tensorrt")
        except ImportError:
            # A system TensorRT installation does not need Python bindings.
            pass

    try:
        np = importlib.import_module("numpy")
        onnx = importlib.import_module("onnx")
        ort = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}.\nInstall with:\n  {INSTALL_HINTS[provider]}"
        ) from exc

    if provider in {"cuda", "tensorrt"} and hasattr(ort, "preload_dlls"):
        # Search NVIDIA wheels first and then normal OS loader paths. This keeps
        # the test valid for both pip-managed and system-managed CUDA/cuDNN.
        ort.preload_dlls(directory="")

    return np, onnx, ort, tensorrt_module


def _make_model(
    onnx: Any, np: Any, model_path: Path
) -> tuple[dict[str, Any], Any]:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto

    rng = np.random.default_rng(20260715)
    x = rng.normal(0.0, 1.0, (64, 256)).astype(np.float32)
    w1 = rng.normal(0.0, 0.03, (256, 512)).astype(np.float32)
    b1 = rng.normal(0.0, 0.01, (512,)).astype(np.float32)
    w2 = rng.normal(0.0, 0.03, (512, 128)).astype(np.float32)
    b2 = rng.normal(0.0, 0.01, (128,)).astype(np.float32)

    nodes = [
        helper.make_node(
            "MatMul", ["input", "weight_1"], ["hidden_mm"], name="matmul_1"
        ),
        helper.make_node("Add", ["hidden_mm", "bias_1"], ["hidden_add"], name="bias_1"),
        helper.make_node("Relu", ["hidden_add"], ["hidden"], name="relu"),
        helper.make_node(
            "MatMul", ["hidden", "weight_2"], ["output_mm"], name="matmul_2"
        ),
        helper.make_node("Add", ["output_mm", "bias_2"], ["output"], name="bias_2"),
    ]
    graph = helper.make_graph(
        nodes,
        "ort_provider_smoke_test",
        [helper.make_tensor_value_info("input", tensor_proto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("output", tensor_proto.FLOAT, [64, 128])],
        [
            numpy_helper.from_array(w1, name="weight_1"),
            numpy_helper.from_array(b1, name="bias_1"),
            numpy_helper.from_array(w2, name="weight_2"),
            numpy_helper.from_array(b2, name="bias_2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ort-provider-tutorial",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    # ONNX 1.22 defaults to a newer IR than some older ORT releases accept.
    # IR 10 is sufficient for this model and supported by the tested runtimes.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, model_path)

    # Keep the correctness oracle independent from every ONNX Runtime EP.
    hidden = np.maximum(x @ w1 + b1, np.float32(0.0))
    reference = (hidden @ w2 + b2).astype(np.float32, copy=False)
    return {"input": x}, reference


def _session_options(ort: Any, profile_prefix: Path | None = None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 2
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


def _run_timed(
    session: Any, feeds: dict[str, Any], warmups: int, runs: int
) -> tuple[Any, float]:
    output = None
    for _ in range(warmups):
        output = session.run(None, feeds)[0]

    samples_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        output = session.run(None, feeds)[0]
        samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    assert output is not None
    return output, statistics.median(samples_ms)


def _profile_provider_counts(profile_path: str | os.PathLike[str]) -> Counter[str]:
    path = Path(profile_path)
    if not path.is_file():
        raise RuntimeError(f"ONNX Runtime did not create the expected profile: {path}")

    with path.open("r", encoding="utf-8") as stream:
        events = json.load(stream)

    counts: Counter[str] = Counter()
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[str(provider)] += 1
    return counts


def _create_regular_session(
    ort: Any,
    provider: str,
    model_path: Path,
    profile_prefix: Path,
    device_id: int,
    cache_dir: Path,
    fp16: bool,
    workspace_mb: int,
) -> tuple[Any, None]:
    target_name = PROVIDER_NAMES[provider]
    available = ort.get_available_providers()
    if target_name not in available:
        details = ""
        if provider in {"cuda", "tensorrt"} and hasattr(ort, "print_debug_info"):
            details = " Run onnxruntime.print_debug_info() for loader diagnostics."
        raise RuntimeError(
            f"{target_name} is not available. Installed providers: {available}.{details}\n"
            f"Install the tested stack with:\n  {INSTALL_HINTS[provider]}"
        )

    options = _session_options(ort, profile_prefix)
    # If the target graph cannot remain on NVIDIA providers, creation fails.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if provider == "cuda":
        providers: list[Any] = [
            (
                target_name,
                {"device_id": device_id, "do_copy_in_default_stream": True},
            ),
        ]
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_engine_cache_prefix": "ort_tutorial",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache_dir),
            "trt_force_timing_cache": False,
            "trt_max_workspace_size": workspace_mb * 1024 * 1024,
            "trt_fp16_enable": fp16,
            "trt_dla_enable": False,
            "trt_sparsity_enable": False,
            "trt_dump_ep_context_model": False,
        }
        providers = [
            (target_name, trt_options),
            ("CUDAExecutionProvider", {"device_id": device_id}),
        ]

    return (
        ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
            enable_fallback=False,
        ),
        None,
    )


def _create_rtx_session(
    ort: Any,
    model_path: Path,
    profile_prefix: Path,
    device_id: int,
    cache_dir: Path,
) -> tuple[Any, tuple[Any, str]]:
    try:
        trt_ep = importlib.import_module("onnxruntime_ep_nv_tensorrt_rtx")
    except ImportError as exc:
        raise RuntimeError(
            "The TensorRT RTX EP plugin is not installed.\nInstall with:\n  "
            + INSTALL_HINTS["nv_tensorrt_rtx"]
        ) from exc

    registration_name = trt_ep.get_ep_name()
    ort.register_execution_provider_library(
        registration_name, trt_ep.get_library_path()
    )
    try:
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == registration_name
        ]
        if not devices:
            raise RuntimeError(
                "The plugin loaded, but no supported TensorRT RTX device was found. "
                "This EP requires a supported RTX GPU and a compatible NVIDIA driver."
            )

        devices_by_id: dict[int, Any] = {}
        for fallback_id, device in enumerate(devices):
            raw_id = getattr(device, "ep_options", {}).get(
                "device_id", fallback_id
            )
            try:
                exposed_id = int(raw_id)
            except (TypeError, ValueError):
                exposed_id = fallback_id
            devices_by_id[exposed_id] = device

        if device_id not in devices_by_id:
            raise RuntimeError(
                f"Device ID {device_id} is invalid; TensorRT RTX exposed "
                f"device IDs {sorted(devices_by_id)}."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        options = _session_options(ort, profile_prefix)
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(
            [devices_by_id[device_id]],
            {
                # Disable capture in a smoke test so changing temporary host
                # buffers cannot violate CUDA Graph address requirements.
                "enable_cuda_graph": "0",
                "nv_runtime_cache_path": str(cache_dir),
            },
        )
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            enable_fallback=False,
        )
        return session, (trt_ep, registration_name)
    except Exception:
        ort.unregister_execution_provider_library(registration_name)
        raise


def _build_parser(default_provider: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that an ONNX Runtime provider executes a real ONNX graph.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_NAMES),
        default=default_provider,
        help=f"provider to test (default: {default_provider})",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="zero-based GPU/device index"
    )
    parser.add_argument(
        "--warmups", type=int, default=3, help="warm-up inference count"
    )
    parser.add_argument("--runs", type=int, default=20, help="measured inference count")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="allow TensorRT to use FP16 internally (off by default for parity)",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="TensorRT workspace limit in MiB (default: 1024)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="provider cache directory (default: user cache directory)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print a traceback on failure"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.device_id < 0 or args.warmups < 0 or args.runs < 1 or args.workspace_mb < 1:
        raise ValueError(
            "device-id and warmups must be non-negative; runs/workspace-mb must be positive"
        )
    if args.fp16 and args.provider != "tensorrt":
        raise ValueError("--fp16 is supported only with --provider tensorrt")

    np, onnx, ort, tensorrt_module = _load_dependencies(args.provider)
    target_name = PROVIDER_NAMES[args.provider]
    cache_root = args.cache_dir or (
        _default_cache_root() / "ort-provider-tutorial" / args.provider
    )

    print(f"ONNX Runtime: {ort.get_version_string()}")
    print(f"Requested provider: {target_name}")
    print(f"Available built-in providers: {ort.get_available_providers()}")
    if tensorrt_module is not None:
        print(f"TensorRT Python package: {tensorrt_module.__version__}")

    plugin_state: tuple[Any, str] | None = None
    target_session = None
    profile_ended = False
    pending_exception: BaseException | None = None
    with tempfile.TemporaryDirectory(prefix="ort-provider-test-") as temp_dir:
        try:
            temp_path = Path(temp_dir)
            model_path = temp_path / "provider_smoke_test.onnx"
            feeds, reference = _make_model(onnx, np, model_path)

            profile_prefix = temp_path / f"profile-{args.provider}"
            if args.provider == "nv_tensorrt_rtx":
                target_session, plugin_state = _create_rtx_session(
                    ort,
                    model_path,
                    profile_prefix,
                    args.device_id,
                    cache_root,
                )
                target_name = plugin_state[1]
                print(f"TensorRT RTX plugin package: {plugin_state[0].__version__}")
                print(f"Registered plugin provider: {target_name}")
            else:
                target_session, _ = _create_regular_session(
                    ort,
                    args.provider,
                    model_path,
                    profile_prefix,
                    args.device_id,
                    cache_root,
                    args.fp16,
                    args.workspace_mb,
                )

            print(f"Session providers: {target_session.get_providers()}")
            result, target_ms = _run_timed(
                target_session, feeds, args.warmups, args.runs
            )
            profile_path = target_session.end_profiling()
            profile_ended = True
            provider_counts = _profile_provider_counts(profile_path)

            absolute_error = float(np.max(np.abs(reference - result)))
            tolerance = 2e-2 if args.fp16 else 2e-3
            np.testing.assert_allclose(
                result,
                reference,
                rtol=tolerance,
                atol=tolerance,
            )

            if provider_counts[target_name] == 0:
                raise RuntimeError(
                    f"The session ran, but profiling found no nodes on {target_name}. "
                    f"Observed node providers: {dict(provider_counts)}"
                )

            allowed_providers = {target_name}
            if args.provider == "tensorrt":
                allowed_providers.add("CUDAExecutionProvider")
            unexpected_providers = {
                name: count
                for name, count in provider_counts.items()
                if name not in allowed_providers
            }
            if unexpected_providers:
                raise RuntimeError(
                    "The NVIDIA-only target session executed graph work on an "
                    f"unexpected provider: {unexpected_providers}"
                )

            print(f"Profiled node events: {dict(provider_counts)}")
            print(f"Maximum absolute error vs NumPy: {absolute_error:.8g}")
            print(f"{target_name} median host-to-host latency: {target_ms:.3f} ms")
            print(
                f"PASS: {target_name} executed {provider_counts[target_name]} profiled node event(s)."
            )
        except BaseException as exc:
            pending_exception = exc
            # Completed helper frames can otherwise retain the plugin session
            # through their traceback locals while the native library unloads.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            if target_session is not None and not profile_ended:
                try:
                    target_session.end_profiling()
                except Exception as cleanup_error:  # noqa: BLE001 - best-effort cleanup
                    print(
                        f"WARNING: Could not finalize ORT profiling: {cleanup_error}",
                        file=sys.stderr,
                    )

            # Every session using a plugin must be destroyed before unloading
            # that plugin library. This also releases files in temp_dir first.
            del target_session
            gc.collect()

            if plugin_state is not None:
                _, registration_name = plugin_state
                try:
                    ort.unregister_execution_provider_library(registration_name)
                except Exception as cleanup_error:  # noqa: BLE001 - cleanup boundary
                    if pending_exception is None:
                        raise
                    print(
                        f"WARNING: Could not unregister {registration_name}: {cleanup_error}",
                        file=sys.stderr,
                    )

    return 0


def main(default_provider: str = "cuda") -> int:
    parser = _build_parser(default_provider)
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - command-line boundary
        print(f"FAIL: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        print(
            f"\nTested installation command:\n  {INSTALL_HINTS[args.provider]}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
