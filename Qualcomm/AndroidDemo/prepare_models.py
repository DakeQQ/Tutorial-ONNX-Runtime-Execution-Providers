#!/usr/bin/env python3
"""Generate the static FP32/QDQ assets consumed by the Android demo."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
QUALCOMM_DIR = PROJECT_DIR.parent
ASSETS_DIR = PROJECT_DIR / "app" / "src" / "main" / "assets"
sys.path.insert(0, str(QUALCOMM_DIR))

from smoke_model import (  # noqa: E402
    BATCH_SIZE,
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    generate_smoke_models,
)


def main() -> int:
    fp32_path, qdq_path = generate_smoke_models(ASSETS_DIR)
    metadata = {
        "batch_size": BATCH_SIZE,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "fp32_model": fp32_path.name,
        "qdq_model": qdq_path.name,
    }
    (ASSETS_DIR / "qnn_smoke_models.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {fp32_path} ({fp32_path.stat().st_size} bytes)")
    print(f"Generated {qdq_path} ({qdq_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
