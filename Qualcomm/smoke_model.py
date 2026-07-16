#!/usr/bin/env python3
"""Generate deterministic static FP32 and QDQ models for the QNN demos.

Runtime-only dependencies are imported inside ``generate_smoke_models`` so the
one-click launchers can inspect the host and create a virtual environment before
requiring ONNX or ONNX Runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BATCH_SIZE = 32
INPUT_SIZE = 64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 32


def deterministic_input(np: Any) -> Any:
    """Return the exact input used by both the desktop and Android demos."""
    values = np.arange(BATCH_SIZE * INPUT_SIZE, dtype=np.float32)
    values = ((values % 257.0) - 128.0) / 128.0
    return values.reshape(BATCH_SIZE, INPUT_SIZE)


def generate_smoke_models(output_dir: Path) -> tuple[Path, Path]:
    """Create a static FP32 model and its QNN-compatible uint8 QDQ variant."""
    import logging

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    float_path = output_dir / "qnn_smoke_fp32.onnx"
    preprocessed_path = output_dir / "qnn_smoke_preprocessed.onnx"
    qdq_path = output_dir / "qnn_smoke_qdq.onnx"

    rng = np.random.default_rng(20260716)
    weight_1 = rng.normal(0.0, 0.08, (INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)
    bias_1 = rng.normal(0.0, 0.02, (HIDDEN_SIZE,)).astype(np.float32)
    weight_2 = rng.normal(0.0, 0.08, (HIDDEN_SIZE, OUTPUT_SIZE)).astype(np.float32)
    bias_2 = rng.normal(0.0, 0.02, (OUTPUT_SIZE,)).astype(np.float32)

    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["input", "weight_1"], ["hidden_mm"], name="matmul_1"
            ),
            helper.make_node(
                "Add", ["hidden_mm", "bias_1"], ["hidden_add"], name="bias_1"
            ),
            helper.make_node("Relu", ["hidden_add"], ["hidden"], name="relu"),
            helper.make_node(
                "MatMul", ["hidden", "weight_2"], ["output_mm"], name="matmul_2"
            ),
            helper.make_node(
                "Add", ["output_mm", "bias_2"], ["output"], name="bias_2"
            ),
        ],
        "ort_qnn_provider_smoke",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [BATCH_SIZE, INPUT_SIZE]
            )
        ],
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [BATCH_SIZE, OUTPUT_SIZE]
            )
        ],
        [
            numpy_helper.from_array(weight_1, "weight_1"),
            numpy_helper.from_array(bias_1, "bias_1"),
            numpy_helper.from_array(weight_2, "weight_2"),
            numpy_helper.from_array(bias_2, "bias_2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="ort-qnn-provider-tutorial",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, float_path)

    class SmokeCalibrationReader(CalibrationDataReader):
        def __init__(self) -> None:
            calibration_rng = np.random.default_rng(20260717)
            self._samples = [
                {
                    "input": calibration_rng.uniform(
                        -1.25,
                        1.25,
                        (BATCH_SIZE, INPUT_SIZE),
                    ).astype(np.float32)
                }
                for _ in range(16)
            ]
            self.rewind()

        def get_next(self) -> dict[str, Any] | None:
            return next(self._iterator, None)

        def rewind(self) -> None:
            self._iterator = iter(self._samples)

    model_changed = qnn_preprocess_model(float_path, preprocessed_path)
    model_to_quantize = preprocessed_path if model_changed else float_path
    calibration_reader = SmokeCalibrationReader()
    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        calibration_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )

    # ORT's generic quantizer emits an informational preprocessing warning when
    # the QNN preprocessor correctly determines that this tiny graph needs no
    # rewrite. Keep one-click output focused on actionable diagnostics.
    previous_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        quantize(model_to_quantize, qdq_path, qnn_config)
    finally:
        logging.disable(previous_level)

    qdq_model = onnx.load(qdq_path)
    onnx.checker.check_model(qdq_model)
    if any(
        dimension.dim_param
        for value in (*qdq_model.graph.input, *qdq_model.graph.output)
        for dimension in value.type.tensor_type.shape.dim
    ):
        raise RuntimeError("Generated QDQ smoke model unexpectedly has dynamic dimensions.")

    if preprocessed_path.exists() and not model_changed:
        preprocessed_path.unlink()
    return float_path, qdq_path
