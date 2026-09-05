#!/usr/bin/env python3
"""Build and benchmark a small transformer-like ONNX graph on NVIDIA EPs.

The benchmark is intentionally synthetic: it models a static-shape decoder
prefill pass with token embedding, causal self-attention, SwiGLU MLP blocks,
residual connections, layer normalization, and a tied last-token LM head. It is
not a text generator and does not implement a KV cache.

CUDA and classic TensorRT run from an onnxruntime-gpu environment. The standalone
TensorRT RTX plugin runs from a separate plain-onnxruntime environment because
the two ONNX Runtime distributions cannot safely share one environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROVIDERS = ("cuda", "tensorrt", "nv_tensorrt_rtx")
PROVIDER_NAMES = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "nv_tensorrt_rtx": "nv_tensorrt_rtx",
}
RESULT_MARKER = "LLM_BENCHMARK_RESULT="


@dataclass(frozen=True)
class ModelConfig:
    batch_size: int
    sequence_length: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    num_layers: int
    vocab_size: int
    seed: int

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 1 for key, value in values.items() if key != "seed"):
            raise ValueError("All model dimensions and layer counts must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden-size must be divisible by num-heads")
        if self.vocab_size < 2:
            raise ValueError("vocab-size must be at least 2")

    @property
    def parameter_count(self) -> int:
        embedding = self.vocab_size * self.hidden_size
        per_layer = (
            3 * self.hidden_size * self.hidden_size
            + 3 * self.hidden_size
            + self.hidden_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * (2 * self.intermediate_size)
            + 2 * self.intermediate_size
            + self.intermediate_size * self.hidden_size
            + self.hidden_size
            + 4 * self.hidden_size
        )
        final_norm_and_bias = 2 * self.hidden_size + self.vocab_size
        return embedding + self.num_layers * per_layer + final_norm_and_bias

    @property
    def estimated_macs(self) -> int:
        batch = self.batch_size
        sequence = self.sequence_length
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        per_layer = (
            batch * sequence * hidden * (3 * hidden)
            + batch * sequence * hidden * hidden
            + 2 * batch * sequence * sequence * hidden
            + batch * sequence * hidden * (2 * intermediate)
            + batch * sequence * intermediate * hidden
        )
        lm_head = batch * hidden * self.vocab_size
        return self.num_layers * per_layer + lm_head


def _default_cache_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "ort-provider-tutorial" / "llm-benchmark"
    return Path.home() / ".cache" / "ort-provider-tutorial" / "llm-benchmark"


def _find_conda_python(environment_name: str) -> Path | None:
    conda_executable = os.environ.get("CONDA_EXE")
    if conda_executable:
        candidate = (
            Path(conda_executable).resolve().parent.parent
            / "envs"
            / environment_name
            / ("python.exe" if sys.platform == "win32" else "bin/python")
        )
        if candidate.is_file():
            return candidate

    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if parent.name.lower() == "envs":
            candidate = parent / environment_name
            candidate /= "python.exe" if sys.platform == "win32" else "bin/python"
            if candidate.is_file():
                return candidate
            break
    return None


def _config_fingerprint(config: ModelConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _build_model(model_path: Path, config: ModelConfig) -> None:
    np = importlib.import_module("numpy")
    onnx = importlib.import_module("onnx")
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto
    rng = np.random.default_rng(config.seed)

    nodes: list[Any] = []
    initializers: list[Any] = []

    def add_initializer(name: str, value: Any) -> str:
        array = np.asarray(value)
        initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def add_weight(name: str, shape: tuple[int, ...], fan_in: int) -> str:
        value = rng.standard_normal(shape).astype(np.float32)
        value *= np.float32(1.0 / math.sqrt(fan_in))
        return add_initializer(name, value)

    def add_linear(
        source: str,
        prefix: str,
        input_size: int,
        output_size: int,
    ) -> str:
        weight = add_weight(
            f"{prefix}.weight", (input_size, output_size), input_size
        )
        bias = add_initializer(
            f"{prefix}.bias", np.zeros((output_size,), dtype=np.float32)
        )
        matmul = f"{prefix}.matmul"
        output = f"{prefix}.output"
        nodes.append(helper.make_node("MatMul", [source, weight], [matmul]))
        nodes.append(helper.make_node("Add", [matmul, bias], [output]))
        return output

    token_embedding = add_weight(
        "token_embedding",
        (config.vocab_size, config.hidden_size),
        config.hidden_size,
    )
    reshape_bshd = add_initializer(
        "reshape_bshd",
        np.asarray(
            [
                config.batch_size,
                config.sequence_length,
                config.num_heads,
                config.hidden_size // config.num_heads,
            ],
            dtype=np.int64,
        ),
    )
    reshape_bsh = add_initializer(
        "reshape_bsh",
        np.asarray(
            [config.batch_size, config.sequence_length, config.hidden_size],
            dtype=np.int64,
        ),
    )
    mlp_split = add_initializer(
        "mlp_split",
        np.asarray(
            [config.intermediate_size, config.intermediate_size], dtype=np.int64
        ),
    )
    attention_scale = add_initializer(
        "attention_scale",
        np.asarray(
            1.0 / math.sqrt(config.hidden_size // config.num_heads),
            dtype=np.float32,
        ),
    )
    causal_mask_value = np.triu(
        np.full(
            (config.sequence_length, config.sequence_length),
            -10000.0,
            dtype=np.float32,
        ),
        k=1,
    ).reshape(1, 1, config.sequence_length, config.sequence_length)
    causal_mask = add_initializer("causal_mask", causal_mask_value)

    hidden = "embedding_output"
    nodes.append(
        helper.make_node(
            "Gather", [token_embedding, "input_ids"], [hidden], axis=0
        )
    )

    for layer_index in range(config.num_layers):
        prefix = f"layer_{layer_index}"
        norm_1_scale = add_initializer(
            f"{prefix}.norm_1.scale",
            np.ones((config.hidden_size,), dtype=np.float32),
        )
        norm_1_bias = add_initializer(
            f"{prefix}.norm_1.bias",
            np.zeros((config.hidden_size,), dtype=np.float32),
        )
        normalized = f"{prefix}.norm_1.output"
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [hidden, norm_1_scale, norm_1_bias],
                [normalized],
                axis=-1,
                epsilon=1e-5,
            )
        )

        projected: dict[str, str] = {}
        for projection in ("query", "key", "value"):
            linear_output = add_linear(
                normalized,
                f"{prefix}.attention.{projection}",
                config.hidden_size,
                config.hidden_size,
            )
            reshaped = f"{prefix}.attention.{projection}.reshaped"
            transposed = f"{prefix}.attention.{projection}.transposed"
            nodes.append(
                helper.make_node(
                    "Reshape", [linear_output, reshape_bshd], [reshaped]
                )
            )
            permutation = [0, 2, 3, 1] if projection == "key" else [0, 2, 1, 3]
            nodes.append(
                helper.make_node(
                    "Transpose", [reshaped], [transposed], perm=permutation
                )
            )
            projected[projection] = transposed

        attention_scores = f"{prefix}.attention.scores"
        scaled_scores = f"{prefix}.attention.scaled_scores"
        masked_scores = f"{prefix}.attention.masked_scores"
        probabilities = f"{prefix}.attention.probabilities"
        context_heads = f"{prefix}.attention.context_heads"
        context_transposed = f"{prefix}.attention.context_transposed"
        context = f"{prefix}.attention.context"
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [projected["query"], projected["key"]],
                    [attention_scores],
                ),
                helper.make_node(
                    "Mul", [attention_scores, attention_scale], [scaled_scores]
                ),
                helper.make_node(
                    "Add", [scaled_scores, causal_mask], [masked_scores]
                ),
                helper.make_node(
                    "Softmax", [masked_scores], [probabilities], axis=-1
                ),
                helper.make_node(
                    "MatMul",
                    [probabilities, projected["value"]],
                    [context_heads],
                ),
                helper.make_node(
                    "Transpose",
                    [context_heads],
                    [context_transposed],
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node(
                    "Reshape", [context_transposed, reshape_bsh], [context]
                ),
            ]
        )
        attention_output = add_linear(
            context,
            f"{prefix}.attention.output",
            config.hidden_size,
            config.hidden_size,
        )
        attention_residual = f"{prefix}.attention.residual"
        nodes.append(
            helper.make_node(
                "Add", [hidden, attention_output], [attention_residual]
            )
        )

        norm_2_scale = add_initializer(
            f"{prefix}.norm_2.scale",
            np.ones((config.hidden_size,), dtype=np.float32),
        )
        norm_2_bias = add_initializer(
            f"{prefix}.norm_2.bias",
            np.zeros((config.hidden_size,), dtype=np.float32),
        )
        mlp_input = f"{prefix}.norm_2.output"
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [attention_residual, norm_2_scale, norm_2_bias],
                [mlp_input],
                axis=-1,
                epsilon=1e-5,
            )
        )
        mlp_projection = add_linear(
            mlp_input,
            f"{prefix}.mlp.input",
            config.hidden_size,
            2 * config.intermediate_size,
        )
        gate = f"{prefix}.mlp.gate"
        up = f"{prefix}.mlp.up"
        sigmoid = f"{prefix}.mlp.sigmoid"
        silu = f"{prefix}.mlp.silu"
        activated = f"{prefix}.mlp.activated"
        nodes.extend(
            [
                helper.make_node(
                    "Split",
                    [mlp_projection, mlp_split],
                    [gate, up],
                    axis=-1,
                ),
                helper.make_node("Sigmoid", [gate], [sigmoid]),
                helper.make_node("Mul", [gate, sigmoid], [silu]),
                helper.make_node("Mul", [silu, up], [activated]),
            ]
        )
        mlp_output = add_linear(
            activated,
            f"{prefix}.mlp.output",
            config.intermediate_size,
            config.hidden_size,
        )
        hidden = f"{prefix}.output"
        nodes.append(
            helper.make_node("Add", [attention_residual, mlp_output], [hidden])
        )

    final_norm_scale = add_initializer(
        "final_norm.scale", np.ones((config.hidden_size,), dtype=np.float32)
    )
    final_norm_bias = add_initializer(
        "final_norm.bias", np.zeros((config.hidden_size,), dtype=np.float32)
    )
    final_hidden = "final_norm.output"
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            [hidden, final_norm_scale, final_norm_bias],
            [final_hidden],
            axis=-1,
            epsilon=1e-5,
        )
    )
    last_token_index = add_initializer(
        "last_token_index",
        np.asarray(config.sequence_length - 1, dtype=np.int64),
    )
    last_token = "last_token"
    lm_head_weight = "lm_head.weight"
    logits_matmul = "logits.matmul"
    logits_bias = add_initializer(
        "lm_head.bias", np.zeros((config.vocab_size,), dtype=np.float32)
    )
    nodes.extend(
        [
            helper.make_node(
                "Gather", [final_hidden, last_token_index], [last_token], axis=1
            ),
            helper.make_node(
                "Transpose", [token_embedding], [lm_head_weight], perm=[1, 0]
            ),
            helper.make_node(
                "MatMul", [last_token, lm_head_weight], [logits_matmul]
            ),
            helper.make_node("Add", [logits_matmul, logits_bias], ["logits"]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "pseudo_small_llm",
        [
            helper.make_tensor_value_info(
                "input_ids",
                tensor_proto.INT64,
                [config.batch_size, config.sequence_length],
            )
        ],
        [
            helper.make_tensor_value_info(
                "logits",
                tensor_proto.FLOAT,
                [config.batch_size, config.vocab_size],
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="ort-provider-pseudo-llm-benchmark",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    metadata = model.metadata_props.add()
    metadata.key = "pseudo_llm_config"
    metadata.value = json.dumps(asdict(config), sort_keys=True)
    onnx.checker.check_model(model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, model_path)


def _session_options(ort: Any, profile_prefix: Path | None = None) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return options


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


def _option_bool(value: bool) -> str:
    return "1" if value else "0"


def _tensorrt_bool(value: bool) -> str:
    return "True" if value else "False"


def _cuda_provider_options(args: argparse.Namespace) -> dict[str, str]:
    gpu_mem_limit = (
        args.gpu_mem_limit_mb * 1024 * 1024
        if args.gpu_mem_limit_mb
        else 2**64 - 1
    )
    return {
        "device_id": str(args.device_id),
        "gpu_mem_limit": str(gpu_mem_limit),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
        "cudnn_conv_use_max_workspace": "1",
        "cudnn_conv1d_pad_to_nc1d": "0",
        "enable_cudnn": "1",
        "enable_cuda_graph": _option_bool(args.cuda_enable_graph),
        "tunable_op_enable": _option_bool(args.cuda_tunable_ops),
        "tunable_op_tuning_enable": _option_bool(args.cuda_tunable_ops),
        "tunable_op_max_tuning_duration_ms": str(
            args.cuda_tuning_duration_ms
        ),
        "prefer_nhwc": _option_bool(args.cuda_prefer_nhwc),
        "use_ep_level_unified_stream": _option_bool(
            args.cuda_unified_stream
        ),
        "use_tf32": _option_bool(args.cuda_tf32),
        "fuse_conv_bias": "0",
        "sdpa_kernel": str(args.cuda_sdpa_kernel),
    }


def _tensorrt_provider_options(
    args: argparse.Namespace, cache_dir: Path
) -> dict[str, str]:
    return {
        "device_id": str(args.device_id),
        "trt_max_partition_iterations": "1000",
        "trt_min_subgraph_size": "1",
        "trt_max_workspace_size": str(args.workspace_mb * 1024 * 1024),
        "trt_fp16_enable": _tensorrt_bool(args.trt_precision == "fp16"),
        "trt_bf16_enable": _tensorrt_bool(args.trt_precision == "bf16"),
        "trt_int8_enable": "False",
        "trt_int8_use_native_calibration_table": "False",
        "trt_dla_enable": "False",
        "trt_dla_core": "0",
        "trt_dump_subgraphs": "False",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(cache_dir),
        "trt_engine_cache_prefix": "pseudo_llm",
        "trt_force_sequential_engine_build": "False",
        "trt_context_memory_sharing_enable": _tensorrt_bool(
            args.trt_context_memory_sharing
        ),
        "trt_layer_norm_fp32_fallback": "False",
        "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": str(cache_dir),
        "trt_force_timing_cache": "False",
        "trt_detailed_build_log": "False",
        "trt_build_heuristics_enable": _tensorrt_bool(
            args.trt_build_heuristics
        ),
        "trt_sparsity_enable": _tensorrt_bool(args.trt_sparsity),
        "trt_builder_optimization_level": str(
            args.trt_builder_optimization_level
        ),
        "trt_auxiliary_streams": str(args.trt_auxiliary_streams),
        "trt_cuda_graph_enable": _tensorrt_bool(
            args.trt_enable_cuda_graph
        ),
        "trt_dump_ep_context_model": "False",
        "trt_weight_stripped_engine_enable": "False",
        "trt_engine_hw_compatible": _tensorrt_bool(
            args.trt_engine_hw_compatible
        ),
    }


def _tensorrt_rtx_provider_options(
    args: argparse.Namespace, cache_dir: Path
) -> dict[str, str]:
    options = {
        "nv_max_workspace_size": str(args.workspace_mb * 1024 * 1024),
        "nv_max_shared_mem_size": str(args.rtx_max_shared_mem_bytes),
        "nv_dump_subgraphs": "0",
        "nv_detailed_build_log": "0",
        "enable_cuda_graph": _option_bool(args.rtx_enable_cuda_graph),
        "nv_multi_profile_enable": "0",
        "nv_use_external_data_initializer": _option_bool(
            args.rtx_use_external_data_initializer
        ),
        "nv_runtime_cache_path": str(cache_dir),
        "nv_weight_streaming_budget": args.rtx_weight_streaming_budget,
        "nv_enable_profiling": "0",
        "nv_use_sync_gpu_allocator": _option_bool(
            args.rtx_use_sync_gpu_allocator
        ),
        "nv_multi_rotary_cache_concat_offset": "0",
        "nv_weight_stripped_engine_enable_experimental": "0",
    }
    if args.rtx_auxiliary_streams >= 0:
        options["nv_length_aux_stream_array"] = str(
            args.rtx_auxiliary_streams
        )
    return options


def _provider_options(
    provider: str, args: argparse.Namespace, cache_dir: Path
) -> dict[str, str]:
    if provider == "cuda":
        return _cuda_provider_options(args)
    if provider == "tensorrt":
        return _tensorrt_provider_options(args, cache_dir)
    return _tensorrt_rtx_provider_options(args, cache_dir)


def _performance_fingerprint(provider: str, args: argparse.Namespace) -> str:
    options = _provider_options(provider, args, Path("<cache>"))
    normalized = {
        key: ("<cache>" if key.endswith("_path") else value)
        for key, value in options.items()
    }
    payload = json.dumps(normalized, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _create_target_session(
    ort: Any,
    provider: str,
    model_path: Path,
    profile_prefix: Path | None,
    plugin_device: Any | None,
    provider_options: dict[str, str],
    cuda_fallback_options: dict[str, str],
) -> Any:
    options = _session_options(ort, profile_prefix)
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    if provider == "cuda":
        providers: list[Any] = [
            ("CUDAExecutionProvider", provider_options)
        ]
        return ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
            enable_fallback=False,
        )

    if provider == "tensorrt":
        providers = [
            ("TensorrtExecutionProvider", provider_options),
            ("CUDAExecutionProvider", cuda_fallback_options),
        ]
        return ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers,
            enable_fallback=False,
        )

    if plugin_device is None:
        raise RuntimeError("TensorRT RTX plugin device was not initialized")
    options.add_provider_for_devices(
        [plugin_device],
        provider_options,
    )
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        enable_fallback=False,
    )


def _load_worker_runtime(
    provider: str, device_id: int
) -> tuple[Any, Any | None, tuple[Any, str] | None, str | None]:
    tensorrt_module = None
    if provider == "tensorrt":
        tensorrt_module = importlib.import_module("tensorrt")

    ort = importlib.import_module("onnxruntime")
    if provider in {"cuda", "tensorrt"} and hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")

    if provider != "nv_tensorrt_rtx":
        return ort, None, None, getattr(tensorrt_module, "__version__", None)

    plugin = importlib.import_module("onnxruntime_ep_nv_tensorrt_rtx")
    registration_name = plugin.get_ep_name()
    ort.register_execution_provider_library(
        registration_name, plugin.get_library_path()
    )
    try:
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == registration_name
        ]
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
                f"TensorRT RTX device {device_id} is unavailable; "
                f"available IDs: {sorted(devices_by_id)}"
            )
        return (
            ort,
            devices_by_id[device_id],
            (plugin, registration_name),
            plugin.__version__,
        )
    except Exception:
        ort.unregister_execution_provider_library(registration_name)
        raise


def _run_worker(args: argparse.Namespace, config: ModelConfig) -> int:
    np = importlib.import_module("numpy")
    model_path = args.model.resolve()
    model_digest = _file_digest(model_path)
    performance_fingerprint = _performance_fingerprint(
        args.worker_provider, args
    )
    cache_dir = (
        args.cache_dir.resolve()
        / args.worker_provider
        / model_digest
        / performance_fingerprint
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider_options = _provider_options(
        args.worker_provider, args, cache_dir
    )
    cuda_fallback_options = _cuda_provider_options(args)

    ort, plugin_device, plugin_state, provider_version = _load_worker_runtime(
        args.worker_provider, args.device_id
    )
    target_name = (
        plugin_state[1]
        if plugin_state is not None
        else PROVIDER_NAMES[args.worker_provider]
    )
    rng = np.random.default_rng(config.seed + 1)
    feeds = {
        "input_ids": rng.integers(
            0,
            config.vocab_size,
            size=(config.batch_size, config.sequence_length),
            dtype=np.int64,
        )
    }

    cpu_session = None
    proof_session = None
    timing_session = None
    pending_exception: BaseException | None = None
    result: dict[str, Any] | None = None
    try:
        cpu_session = ort.InferenceSession(
            model_path,
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        reference = cpu_session.run(None, feeds)[0]
        del cpu_session
        cpu_session = None
        gc.collect()

        with tempfile.TemporaryDirectory(
            prefix="ort-llm-profile-", ignore_cleanup_errors=True
        ) as temp_dir:
            profile_prefix = Path(temp_dir) / f"profile-{args.worker_provider}"
            setup_start = time.perf_counter_ns()
            proof_session = _create_target_session(
                ort,
                args.worker_provider,
                model_path,
                profile_prefix,
                plugin_device,
                provider_options,
                cuda_fallback_options,
            )
            proof_setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0
            proof_output = proof_session.run(None, feeds)[0]
            profile_path = proof_session.end_profiling()
            provider_counts = _profile_provider_counts(profile_path)
            np.testing.assert_allclose(
                proof_output, reference, rtol=args.tolerance, atol=args.tolerance
            )

        if provider_counts[target_name] == 0:
            raise RuntimeError(
                f"No profiled node ran on {target_name}; observed "
                f"providers: {dict(provider_counts)}"
            )
        allowed_providers = {target_name}
        if args.worker_provider == "tensorrt":
            allowed_providers.add("CUDAExecutionProvider")
        unexpected = {
            name: count
            for name, count in provider_counts.items()
            if name not in allowed_providers
        }
        if unexpected:
            raise RuntimeError(f"Unexpected profiled providers: {unexpected}")

        del proof_session
        proof_session = None
        gc.collect()

        setup_start = time.perf_counter_ns()
        timing_session = _create_target_session(
            ort,
            args.worker_provider,
            model_path,
            None,
            plugin_device,
            provider_options,
            cuda_fallback_options,
        )
        timed_setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000.0

        first_start = time.perf_counter_ns()
        output = timing_session.run(None, feeds)[0]
        first_run_ms = (time.perf_counter_ns() - first_start) / 1_000_000.0
        for _ in range(args.warmups):
            output = timing_session.run(None, feeds)[0]

        samples_ms: list[float] = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            output = timing_session.run(None, feeds)[0]
            samples_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

        np.testing.assert_allclose(
            output, reference, rtol=args.tolerance, atol=args.tolerance
        )
        maximum_error = float(np.max(np.abs(reference - output)))
        median_ms = float(statistics.median(samples_ms))
        result = {
            "provider": args.worker_provider,
            "provider_name": target_name,
            "ort_version": ort.__version__,
            "provider_version": provider_version,
            "provider_options": provider_options,
            "cache_directory": str(cache_dir),
            "profiled_node_events": dict(provider_counts),
            "proof_session_setup_ms": proof_setup_ms,
            "timed_session_setup_ms": timed_setup_ms,
            "first_run_ms": first_run_ms,
            "mean_ms": float(statistics.fmean(samples_ms)),
            "median_ms": median_ms,
            "p90_ms": float(np.percentile(samples_ms, 90)),
            "minimum_ms": float(min(samples_ms)),
            "maximum_ms": float(max(samples_ms)),
            "prefill_tokens_per_second": (
                config.batch_size * config.sequence_length * 1000.0 / median_ms
            ),
            "maximum_absolute_error": maximum_error,
            "runs": args.runs,
            "warmups": args.warmups,
        }
    except BaseException as exc:
        pending_exception = exc
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        del cpu_session, proof_session, timing_session
        gc.collect()
        if plugin_state is not None:
            _, registration_name = plugin_state
            try:
                ort.unregister_execution_provider_library(registration_name)
            except Exception:
                if pending_exception is None:
                    raise

    if result is None:
        raise RuntimeError("Benchmark worker did not produce a result")
    print(f"Provider: {result['provider_name']}")
    print(
        "Performance options: "
        + json.dumps(result["provider_options"], sort_keys=True)
    )
    print(f"Profiled node events: {result['profiled_node_events']}")
    print(f"Median latency: {result['median_ms']:.3f} ms")
    print(
        "Prefill throughput: "
        f"{result['prefill_tokens_per_second']:.1f} token/s"
    )
    print(f"Maximum absolute error: {result['maximum_absolute_error']:.8g}")
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))
    return 0


def _worker_command(
    python_executable: Path,
    script_path: Path,
    provider: str,
    model_path: Path,
    config: ModelConfig,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(python_executable),
        str(script_path),
        "--worker-provider",
        provider,
        "--model",
        str(model_path),
        "--cache-dir",
        str(args.cache_dir),
        "--device-id",
        str(args.device_id),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
        "--workspace-mb",
        str(args.workspace_mb),
        "--tolerance",
        str(args.tolerance),
    ]
    scalar_options = (
        "gpu_mem_limit_mb",
        "cuda_tuning_duration_ms",
        "cuda_sdpa_kernel",
        "trt_precision",
        "trt_builder_optimization_level",
        "trt_auxiliary_streams",
        "rtx_auxiliary_streams",
        "rtx_weight_streaming_budget",
        "rtx_max_shared_mem_bytes",
    )
    boolean_options = (
        "cuda_enable_graph",
        "cuda_tunable_ops",
        "cuda_unified_stream",
        "cuda_tf32",
        "cuda_prefer_nhwc",
        "trt_build_heuristics",
        "trt_context_memory_sharing",
        "trt_enable_cuda_graph",
        "trt_engine_hw_compatible",
        "trt_sparsity",
        "rtx_enable_cuda_graph",
        "rtx_use_sync_gpu_allocator",
        "rtx_use_external_data_initializer",
    )
    for name in scalar_options:
        command.extend(
            ["--" + name.replace("_", "-"), str(getattr(args, name))]
        )
    for name in boolean_options:
        option_name = name.replace("_", "-")
        command.append(
            f"--{option_name}"
            if getattr(args, name)
            else f"--no-{option_name}"
        )
    for name, value in asdict(config).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def _print_results(results: list[dict[str, Any]]) -> None:
    cuda_result = next(
        (result for result in results if result["provider"] == "cuda"), None
    )
    headers = (
        "Provider",
        "Nodes",
        "Build/proof ms",
        "Reload ms",
        "First ms",
        "Median ms",
        "P90 ms",
        "Token/s",
        "vs CUDA",
    )
    rows: list[tuple[str, ...]] = []
    for result in results:
        nodes = ", ".join(
            f"{name}:{count}"
            for name, count in result["profiled_node_events"].items()
        )
        speedup = "-"
        if cuda_result is not None:
            speedup = f"{cuda_result['median_ms'] / result['median_ms']:.2f}x"
        rows.append(
            (
                result["provider"],
                nodes,
                f"{result['proof_session_setup_ms']:.1f}",
                f"{result['timed_session_setup_ms']:.1f}",
                f"{result['first_run_ms']:.3f}",
                f"{result['median_ms']:.3f}",
                f"{result['p90_ms']:.3f}",
                f"{result['prefill_tokens_per_second']:.0f}",
                speedup,
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    print("\n" + format_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def _run_controller(args: argparse.Namespace, config: ModelConfig) -> int:
    selected_providers = [
        provider.strip() for provider in args.providers.split(",") if provider.strip()
    ]
    invalid = [provider for provider in selected_providers if provider not in PROVIDERS]
    if invalid:
        raise ValueError(f"Unknown providers: {invalid}; choose from {PROVIDERS}")
    if not selected_providers:
        raise ValueError("At least one provider must be selected")

    model_path = args.model
    if model_path is None:
        model_path = (
            args.cache_dir
            / "models"
            / f"pseudo_small_llm_{_config_fingerprint(config)}.onnx"
        )
    model_path = model_path.resolve()
    if args.rebuild_model or not model_path.is_file():
        print(f"Building model: {model_path}")
        _build_model(model_path, config)
    else:
        print(f"Reusing model: {model_path}")

    model_digest = _file_digest(model_path)
    if args.clear_cache:
        for provider in selected_providers:
            provider_cache = args.cache_dir.resolve() / provider / model_digest
            if provider_cache.is_dir():
                shutil.rmtree(provider_cache)

    gpu_python = args.gpu_python.resolve()
    rtx_python = args.rtx_python.resolve()
    for executable in {gpu_python, rtx_python}:
        if not executable.is_file():
            raise FileNotFoundError(f"Python interpreter not found: {executable}")

    print(
        f"Model: {config.parameter_count / 1_000_000:.2f}M parameters, "
        f"{config.estimated_macs / 1_000_000_000:.2f} GMAC/run, "
        f"{model_path.stat().st_size / 1024**2:.1f} MiB"
    )
    print(f"GPU Python: {gpu_python}")
    print(f"TensorRT RTX Python: {rtx_python}")

    script_path = Path(__file__).resolve()
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for provider in selected_providers:
        python_executable = (
            rtx_python if provider == "nv_tensorrt_rtx" else gpu_python
        )
        command = _worker_command(
            python_executable,
            script_path,
            provider,
            model_path,
            config,
            args,
        )
        print(f"\n=== {provider} ===")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            check=False,
        )
        stdout_lines = completed.stdout.splitlines()
        marker_lines = [
            line for line in stdout_lines if line.startswith(RESULT_MARKER)
        ]
        visible_lines = [
            line for line in stdout_lines if not line.startswith(RESULT_MARKER)
        ]
        if visible_lines:
            print("\n".join(visible_lines))
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode or not marker_lines:
            failures.append(provider)
            continue
        results.append(json.loads(marker_lines[-1][len(RESULT_MARKER) :]))

    if results:
        _print_results(results)
    report = {
        "model": str(model_path),
        "model_sha256_prefix": model_digest,
        "config": asdict(config),
        "parameter_count": config.parameter_count,
        "estimated_macs": config.estimated_macs,
        "results": results,
        "failures": failures,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_output.resolve()}")
    if failures:
        print(f"\nFAILED providers: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    gpu_python = _find_conda_python("python_313") or Path(sys.executable)
    rtx_python = _find_conda_python("python_313_trt_rtx") or Path(sys.executable)
    parser = argparse.ArgumentParser(
        description="Benchmark a transformer-like ONNX graph on NVIDIA EPs."
    )
    parser.add_argument(
        "--providers",
        default=",".join(PROVIDERS),
        help="comma-separated providers: cuda,tensorrt,nv_tensorrt_rtx",
    )
    parser.add_argument("--gpu-python", type=Path, default=gpu_python)
    parser.add_argument("--rtx-python", type=Path, default=rtx_python)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=_default_cache_root())
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--rebuild-model", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=0)
    parser.add_argument(
        "--cuda-enable-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CUDA EP graph capture; requires stable device-bound I/O",
    )
    parser.add_argument(
        "--cuda-tunable-ops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="profile TunableOp kernels; opt-in because tuning can regress a model",
    )
    parser.add_argument("--cuda-tuning-duration-ms", type=int, default=0)
    parser.add_argument(
        "--cuda-unified-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cuda-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cuda-prefer-nhwc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cuda-sdpa-kernel", type=int, default=0)
    parser.add_argument(
        "--trt-precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument(
        "--trt-builder-optimization-level", type=int, default=5
    )
    parser.add_argument("--trt-auxiliary-streams", type=int, default=-1)
    parser.add_argument(
        "--trt-build-heuristics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trt-context-memory-sharing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trt-enable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="classic TensorRT graph capture; requires device-bound I/O",
    )
    parser.add_argument(
        "--trt-engine-hw-compatible",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trt-sparsity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rtx-enable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rtx-auxiliary-streams", type=int, default=-1)
    parser.add_argument("--rtx-weight-streaming-budget", default="0")
    parser.add_argument(
        "--rtx-use-sync-gpu-allocator",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rtx-use-external-data-initializer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rtx-max-shared-mem-bytes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--worker-provider", choices=PROVIDERS, default=None, help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.device_id < 0 or args.warmups < 0 or args.runs < 1:
        parser.error("device-id/warmups must be non-negative and runs must be positive")
    if args.workspace_mb < 1 or args.tolerance <= 0:
        parser.error("workspace-mb and tolerance must be positive")
    if args.gpu_mem_limit_mb < 0 or args.cuda_tuning_duration_ms < 0:
        parser.error("GPU memory limit and CUDA tuning duration cannot be negative")
    if not 0 <= args.trt_builder_optimization_level <= 5:
        parser.error("trt-builder-optimization-level must be between 0 and 5")
    if args.trt_auxiliary_streams < -1 or args.rtx_auxiliary_streams < -1:
        parser.error("auxiliary stream counts must be -1 (automatic) or non-negative")
    if args.rtx_max_shared_mem_bytes < 0:
        parser.error("rtx-max-shared-mem-bytes cannot be negative")
    config = ModelConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    config.validate()
    try:
        if args.worker_provider is not None:
            if args.model is None or not args.model.is_file():
                raise FileNotFoundError(f"Worker model not found: {args.model}")
            return _run_worker(args, config)
        return _run_controller(args, config)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - command-line boundary
        print(f"FAIL: {exc}", file=sys.stderr)
        if args.worker_provider is not None:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())