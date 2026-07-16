#!/usr/bin/env python3
"""One-command AMD ONNX Runtime smoke test with fail-closed EP verification.

The script selects an AMD-capable execution provider, creates or locates a small
ONNX model, runs inference, and inspects the ONNX Runtime profile to prove that
at least one graph node executed on the requested accelerator.

Examples:
    Windows AMD GPU: python amd_ort_hardware_test.py --target dml --bootstrap
    Ubuntu AMD GPU:  python amd_ort_hardware_test.py --target migraphx --bootstrap
    Windows ML GPU:  python amd_ort_hardware_test.py --target migraphx --windows-ml
    Ryzen AI NPU:    python amd_ort_hardware_test.py --target npu
    Script self-test: python amd_ort_hardware_test.py --target cpu
    Built-in tests:   python amd_ort_hardware_test.py --unit-tests
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, NoReturn

AMD_PROVIDERS = (
    "VitisAIExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
)
ORT_DISTRIBUTIONS = {
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-migraphx",
    "onnxruntime-openvino",
    "onnxruntime-rocm",
    "onnxruntime-training",
    "onnxruntime-vitisai",
    "onnxruntime-windowsml",
}
PROTECTED_ORT_DISTRIBUTIONS = {
    "onnxruntime-vitisai",
    "onnxruntime-windowsml",
}
DIRECTML_ORT_VERSION = "1.24.4"
WINDOWS_NUMPY_VERSION = "1.26.4"
DIRECTML_CP312_WHEEL = "onnxruntime_directml-1.24.4-cp312-cp312-win_amd64.whl"
DIRECTML_CP312_SHA256 = "f2ecb68b7b7b259d2ef3112ae760149f9b5a1e7c0fbb73d539da6250a648a614"
DIRECTML_DLL_SHA256 = "b73972115320e906a49602f2027a3266622881b0d325ba685e0f165a9482a8d7"
WINDOWS_ML_WASDK_VERSION = "2.1.3"
WINDOWS_ML_ORT_DISTRIBUTION_VERSION = "1.24.6.202605042033"
WINDOWS_ML_MIGRAPHX_MSIX_VERSION = (1, 8, 57, 0)
MIGRAPHX_ORT_BY_ROCM = {
    "7.2.1": "1.23.2",
    "7.2.4": "1.23.2",
}
MIGRAPHX_WHEEL_SHA256 = {
    ("7.2.1", "cp310"): "07f485fbeb8fbd6a89fa42d24832b4e206057fca62654b0eb39eb1edf9d6e70a",
    ("7.2.1", "cp312"): "663bff4dc3f72582d69f12ad073eb5695dfb526d574376cc8e5b161c7d2f0f08",
    ("7.2.4", "cp310"): "4886faab646a7ef12f33fb53f085208182fab8dac249ba199dc5d23f8bd128ec",
    ("7.2.4", "cp312"): "ee8edeb2ba6a8d99b3043b23e812423e6f10333b508e003fc77b0feda197449f",
}
MIGRAPHX_PROVIDER_SHA256_BY_ROCM = {
    "7.2.1": "8079986332cdf12234635ed4f2b5abd1b49519f6592d6dfcd8afaf5000887b7b",
    "7.2.4": "f3fb0b10996b2a2f94afc59edf6fab421bfa12842f09518339d1e0d8f3bd86c7",
}
AMD_PCI_VENDOR_ID = 0x1002
SMOKE_MODEL_SHA256 = "d8b1fcc3bfdd175afab01c11d502f0b1b28b3516434961e6de65a1f315434b7d"
SMOKE_MODEL_BASE64 = (
    "CAoSEGFtZF9vcnRfb25lY2xpY2s6tw8KJQoBWAoBVwoBQhIBQyIEQ29udioRCgRwYWRzQAFAAUABQAGgAQcKDAoBQxIBUiIEUmVsdQoZCgFSEgFZIhFHbG9iYWxBdmVyYWdlUG9vbBINYW1kX29ydF9zbW9rZSrQDQgQCAMIAwgDEAFCAVdKwA3yTc88QsngvHwrE72LvlM8lrm4vS8TsjtxxoQ5mvMGPacoDby6kLW9/W1avAZcNz0y8ug9ClxsPb+bbr3MRCC9M1eGPcO/N7vLoOe7z+MRPiMGUT0PcG89bWGdPeU8rzzZIK68WotMu/aBC72XioY90mCnPExePb2z5YU8L7rfvK4QtLz7+Zi8mJqYvU9asryQ6Js9Y4YlvGUHbz1khoo80o+rPXHVrTzjRfk84SgmvGfuPDyrUkM7b8cjvTF+tzx/JN68hxa/PHRdNr3oEIq8RWS9vHgQCj0KoWy9qXM1PfjkvzsZ2ow65rYBPU2iST0HPEg9oP+mvMrcX7x27Y89AZypvVt9ZT1vb3A9Z6s3PdKAXD3ArXi9cLstPaMnJD20o6U8LQZwvUAvYT3flJU9OkhZvZL0YLxIPYI9mbiJvQ1/1rzFCro7bUJ2veNmgr1DqgC+pcDxPPvXp732igC+A3Mtvf2IFz6HJN49q1cuPVh4E7wvbEm99RH9vD/q+b0fuyI8u+9XPSTmHL2SA4g7maaSvWdhGzz4Ubq9iIJgPacLqr1/jU88YuucvbN4Aru+u0c9kuatPAcbYj25QEA9KfUSvcHhKT1Q/N08tdqWvaP/RD0IJGk8BTvuPI+6Ej1wNl68n8AyPOgi/btjS605dn8gvXn+pDzrAlG9SNeXPFxdRrw+JkU9UhwpPDcCG76vnN48dyKvPIFQOr3uH7I9Kt8CPdSbET2AafS8IEDCPHBIqjs4Waq7d5HlvFO6mTyyXqq9T/cpPCCSWj2rMnY8wRebu4eWrb1Kf4W9MziSPbIBjrrFbsk9LKSrPV/8BL35FYi811+aPJnrvD3Q62g95MIFPRis1TrB27C9L4M9vaJiY71PPlM73yyoPb0Q17s3Y1c8NPcuu0zsmLyCmfK8CjXXPN0TkTyY8WS9DcpzvTBLhD0z7Fa9TwqSvEobID2vAFg8fsuQO5IVcD3YDVa8D9wyPdOvKD3OR4w8y8dBvfSVqDx1sCE98mHavIt1KT3vSDe9fWOSvQ+UvroHBha9pzW2vTzrBb6/XIq8SYWBvDkZA73HoCE8Uil/vJ+j/by5H7g9k4BJvIB4JbvEjDc9yc+DvESJtbxf5Iq5u+IlvRNAlLuTy2c9zqIjPGthWTufLlW9C+x7vaN4sroTJ5m8zM2Ouv86tbyv7mI909AGvLPb9DwNtAm+JNqgvHetCT3VzEE9qw8YPRsy5j2MfRK9hfyGvFUFrzxaTpY891qUvEF+iT2SzCs9x0CNPSOPzr1/QIW8BeHYPP6tHL1vklc9uxD4O0aDJr3wNhI8ChMKPLsqzz3lyAa9DXiLvJ4yMT3HGWs9Cgb0vSnWErr92mO78/e4veb4Q7qbZAm8XlDGvCmCL7239Hs90HTYOwIzY7wuPDm7j8nUvLPBKbyKL5C9zxo+PBD31LyNCGm88jdtPFNoHjtPNAI+/xhyvSd/uz2HqMY8k7MnvXeuGj1VB3g851qsOqvg+jzXz2C8vbEdu/SfKD1KW0o9zmPAOm6JrjzuQyU7mSSePAeQqTyFy0E9ToGbu8NWLr2Xosy8L1hAPA0qHD3TiyC8dWCwvSuf8D17YQi8bFgNPdjeZLv105M9V9YIPYZfBr1rbxa99yaFuz/Rhj0X2tQ8ArCZPWoIUTuan888sBVpPW/ClDwRsx++/0y9PCQixrzLuRk7ZTVAPb+EJz0ST0s9Q66/vdOfLTxGoqM9JiiYPX4bSj0sahq+jR47vbf4MryzJmM8Mq+JvSwJn700pYs9e5Utvf+puD2Tf/+9nc3OvL27nTuDTzU8fqE4PCAT1b2IAbs9mZavPU8loT1HcNY8+/YoPVtgFL4rvL+8lCCivMBYmLw/IBA788bDvLgn/DyZ2hO9n3w0vUNScT1CIcw83A+avKJsezuVOAc9WR+uPV96kj3fOhG98IelPHqoST2nAaa8RcsdPf1Mlz2vU/U8p+ZaPBlEQbzF3nA9LrzGPA9sSD1Lc8u79ye0vDOm/byD5Pq8FXpDPR/S9zyg5Bm9G6EuvUvFqz2Tiju9TBGyvRsqaz2/Bbo9X2eoPaZVsjtutCW98zeqvTP8KT0Cwdk6JaNZuJcg6Lwb7B+7r0C2PYbbMTskdIu9sUqZvevH5rwXQRQ9RLsxPcKkRj27tb48OXkLvVuNRD0n3169KPcNvj4zDzwfMJY8caEavlYKxDzrTWy9C/HzPVOOSr0zfHc8ni6PvTCPkLyL7wY9+6MxPddTGr2tYWo9C38KvYZnST3jELQ9brKrPc3lErvsfD683RfjvGLEPbwqSQgQEAFCAUJKQJT9Xzxmlgs8U3U1O+RFWby8Gqq7xFBTPAqQRjuARMa8nWGmPGJNGDyGBog7qUaSu9nFQjxgP2W8C6A3O9RGOrxaGwoBWBIWChQIARIQCgIIAQoCCAMKAghACgIIQGIbCgFZEhYKFAgBEhAKAggBCgIIEAoCCAEKAggBQgQKABAR"
)


def info(message: str) -> None:
    print(f"[INFO/信息] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN/警告] {message}", file=sys.stderr, flush=True)


def fail(message: str, exit_code: int = 2) -> NoReturn:
    print(f"[FAIL/失败] {message}", file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


def canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_ort_distributions() -> set[str]:
    installed: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            canonical = canonical_package_name(name)
            if canonical in ORT_DISTRIBUTIONS:
                installed.add(canonical)
    return installed


def require_distribution_version(distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        fail(f"Required Python distribution is missing: {distribution}=={expected}")
    if actual != expected:
        fail(
            f"Unaudited Python distribution version: {distribution}=={actual}; "
            f"this guide requires {expected}. Recreate the documented environment."
        )


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    info("Running/执行: " + " ".join(command))
    try:
        return subprocess.run(command, check=check, text=True)
    except FileNotFoundError as exc:
        fail(f"Command not found: {command[0]} ({exc})")
    except subprocess.CalledProcessError as exc:
        fail(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}. "
            "Review the messages immediately above. / 命令执行失败；请检查上方输出。"
        )


def in_isolated_python_environment() -> bool:
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_prefix and conda_name.lower() != "base":
        try:
            return Path(conda_prefix).resolve() == Path(sys.prefix).resolve()
        except OSError:
            return False
    return False


def version_tuple(version: str) -> tuple[int, ...] | None:
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        return None
    return tuple(int(part) for part in version.split("."))


def canonical_rocm_release(version: str) -> str | None:
    if version in MIGRAPHX_ORT_BY_ROCM:
        return version
    parsed = version_tuple(version)
    if parsed and len(parsed) == 3 and parsed[-1] == 0:
        short = ".".join(str(part) for part in parsed[:2])
        if short in MIGRAPHX_ORT_BY_ROCM:
            return short
    return None


def migraphx_wheel_details(release: str, python_version: tuple[int, int]) -> tuple[str, str, str]:
    ort_version = MIGRAPHX_ORT_BY_ROCM.get(release)
    python_tag = f"cp{python_version[0]}{python_version[1]}"
    expected_sha256 = MIGRAPHX_WHEEL_SHA256.get((release, python_tag))
    if ort_version is None or expected_sha256 is None:
        raise ValueError(f"No audited AMD MIGraphX wheel for ROCm {release}, Python {python_version}.")
    wheel_name = (
        f"onnxruntime_migraphx-{ort_version}-{python_tag}-{python_tag}-"
        "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    wheel_url = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{release}/{wheel_name}"
    return wheel_name, wheel_url, expected_sha256


def pip_install_for_target(target: str, rocm_version: str | None) -> None:
    system = platform.system()
    if not in_isolated_python_environment():
        fail(
            "--bootstrap is allowed only inside an activated venv or non-base Conda environment. "
            "Create and activate the environment shown in the guide first. / --bootstrap 只能在已激活的"
            "虚拟环境或非 base Conda 环境中运行。"
        )

    installed = installed_ort_distributions()
    protected = installed & PROTECTED_ORT_DISTRIBUTIONS
    if os.environ.get("RYZEN_AI_INSTALLATION_PATH") or protected:
        fail(
            "Refusing to modify a Ryzen AI or Windows ML managed environment with --bootstrap. "
            f"Detected: {sorted(protected) or ['RYZEN_AI_INSTALLATION_PATH']}. Create a separate "
            "virtual environment. / 拒绝修改 Ryzen AI 或 Windows ML 厂商环境；请新建独立环境。"
        )

    if target in {"npu", "vitisai"}:
        fail(
            "Vitis AI EP is supplied by the Ryzen AI/Vitis AI installer; do not replace it "
            "with a public PyPI wheel. Activate the vendor environment first. / "
            "Vitis AI EP 由 Ryzen AI/Vitis AI 安装器提供；请先激活厂商环境。"
        )

    expected_runtime_wheel: tuple[str, str, str] | None = None
    if target == "cpu":
        desired = "onnxruntime"
        dependency_specs = ["numpy"]
        runtime_specs = [desired]
        runtime_download_specs = runtime_specs
    elif system == "Windows":
        if target == "migraphx":
            fail(
                "The regular pip bootstrap uses DirectML on Windows. Acquire MIGraphX through "
                "Windows ML instead. / Windows 的常规 pip 引导使用 DirectML；MIGraphX 请通过 Windows ML 获取。"
            )
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            fail(
                "The PyPI DirectML wheel is x64-only. Use an x64 Python process or Windows ML "
                "on ARM64. / PyPI DirectML wheel 仅支持 x64。"
            )
        if sys.version_info[:2] != (3, 12):
            fail(
                "The audited DirectML bootstrap recipe requires Python 3.12.x; "
                f"current Python is {platform.python_version()}. Use the guide's Python 3.12 venv."
            )
        desired = "onnxruntime-directml"
        dependency_specs = [f"numpy=={WINDOWS_NUMPY_VERSION}"]
        runtime_specs = [f"{desired}=={DIRECTML_ORT_VERSION}"]
        runtime_download_specs = runtime_specs
        expected_runtime_wheel = (DIRECTML_CP312_WHEEL, DIRECTML_CP312_SHA256, "Microsoft DirectML")
    elif system == "Linux":
        if target == "dml":
            fail("DirectML is Windows-only. / DirectML 仅支持 Windows。")
        kernel_release = platform.release().lower()
        if "microsoft" in kernel_release or "wsl" in kernel_release:
            fail(
                "AMD's current WSL guidance marks MIGraphX unsupported; bootstrap will not "
                "modify this environment. Use native Ubuntu, DirectML, or Windows ML."
            )
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            fail("AMD's published MIGraphX wheel is Linux x86-64 only. / MIGraphX wheel 仅支持 Linux x86-64。")
        if sys.version_info[:2] not in {(3, 10), (3, 12)}:
            fail(
                "AMD's current MIGraphX wheel is published for CPython 3.10 and 3.12. "
                f"Current interpreter: {platform.python_version()}. / 当前 MIGraphX wheel "
                "仅提供 CPython 3.10/3.12。"
            )
        if not Path("/opt/rocm/bin/migraphx-driver").is_file():
            fail(
                "MIGraphX is not installed at /opt/rocm/bin/migraphx-driver. Run "
                "'sudo apt install migraphx' first. / 未检测到 MIGraphX，请先安装 migraphx。"
            )
        detected_rocm = detect_rocm_version()
        if not detected_rocm:
            fail(
                "Cannot verify the installed ROCm release from /opt/rocm. Repair the /opt/rocm "
                "installation/symlink before bootstrapping; --rocm-version is an assertion, not "
                "a substitute for detection. / 无法验证 /opt/rocm 的已安装版本；请先修复安装或软链接。"
            )
        if rocm_version:
            requested_tuple = version_tuple(rocm_version)
            detected_tuple = version_tuple(detected_rocm)
            if requested_tuple and detected_tuple:
                requested_tuple = requested_tuple + (0,) * (3 - len(requested_tuple))
                detected_tuple = detected_tuple + (0,) * (3 - len(detected_tuple))
            if requested_tuple != detected_tuple:
                fail(
                    f"--rocm-version={rocm_version} does not match installed ROCm {detected_rocm}. "
                    "Mixing release sets is unsafe. / 指定 ROCm 版本与已安装版本不一致。"
                )
        rocm_version = detected_rocm
        release = canonical_rocm_release(rocm_version)
        if release is None:
            fail(
                f"No audited MIGraphX wheel mapping is defined for ROCm {rocm_version}. "
                f"Audited releases: {', '.join(MIGRAPHX_ORT_BY_ROCM)}."
            )
        desired = "onnxruntime-migraphx"
        ort_version = MIGRAPHX_ORT_BY_ROCM[release]
        wheel_name, wheel_url, wheel_sha256 = migraphx_wheel_details(
            release,
            (sys.version_info.major, sys.version_info.minor),
        )
        dependency_specs = ["numpy==1.26.4"]
        runtime_specs = [f"{desired}=={ort_version}"]
        runtime_download_specs = [wheel_url]
        expected_runtime_wheel = (wheel_name, wheel_sha256, "AMD MIGraphX")
    else:
        fail("Only Windows and Linux are supported by this demo. / 此演示仅支持 Windows 和 Linux。")

    conflicts = sorted(installed - {desired})
    if conflicts:
        fail(
            "Refusing to uninstall an existing ONNX Runtime distribution during bootstrap: "
            f"{conflicts}. Recreate the disposable venv from the guide. This avoids leaving a "
            "working environment damaged if a download or install fails. / bootstrap 不会卸载现有 "
            "ONNX Runtime；请按指南重建专用虚拟环境。"
        )
    if desired in installed:
        fail(
            f"{desired} is already installed but did not provide a usable target EP. Bootstrap "
            "will not rewrite an existing runtime in place; recreate the dedicated venv and "
            "check the driver/runtime prerequisites. / 已安装的运行时不可用；请重建专用环境。"
        )

    download_requirements = [*dependency_specs, *runtime_download_specs]
    install_requirements = [*dependency_specs, *runtime_specs]
    with tempfile.TemporaryDirectory(prefix="amd-ort-wheelhouse-") as wheelhouse_raw:
        wheelhouse = Path(wheelhouse_raw)
        info("Downloading all wheels before changing the environment. / 先完整下载所有 wheel。")
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--dest",
                str(wheelhouse),
                *download_requirements,
            ]
        )
        if expected_runtime_wheel is not None:
            wheel_name, expected_sha256, artifact_name = expected_runtime_wheel
            runtime_wheel = wheelhouse / wheel_name
            if not runtime_wheel.is_file():
                fail(f"The expected {artifact_name} wheel was not downloaded: {runtime_wheel}")
            actual_sha256 = sha256_file(runtime_wheel)
            if actual_sha256 != expected_sha256:
                fail(
                    f"{artifact_name} wheel integrity check failed; refusing installation. "
                    f"Expected {expected_sha256}, got {actual_sha256}."
                )
            info(f"Verified {artifact_name} wheel SHA-256/已验证 wheel: {actual_sha256}")
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                *install_requirements,
            ]
        )

    os.environ["AMD_ORT_BOOTSTRAPPED"] = "1"
    info("Packages installed; restarting this script. / 依赖安装完成，正在重启脚本。")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def detect_rocm_version() -> str | None:
    candidates = [
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    ]
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", text)
        if match:
            return match.group(1)

    hipconfig = Path("/opt/rocm/bin/hipconfig")
    if hipconfig.exists():
        try:
            result = subprocess.run(
                [str(hipconfig), "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
            match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", result.stdout + result.stderr)
            if match:
                return match.group(1)
        except OSError:
            pass
    return None


def import_runtime() -> tuple[Any, Any]:
    try:
        np = importlib.import_module("numpy")
        ort = importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Cannot import NumPy/ONNX Runtime: {exc} / 无法导入 NumPy/ONNX Runtime。"
        ) from exc
    return np, ort


def choose_provider(target: str, available: Iterable[str]) -> str:
    available_set = set(available)
    if target == "npu":
        candidates = ["VitisAIExecutionProvider"]
    elif target == "gpu":
        candidates = ["MIGraphXExecutionProvider", "DmlExecutionProvider"]
    elif target == "migraphx":
        candidates = ["MIGraphXExecutionProvider"]
    elif target == "dml":
        candidates = ["DmlExecutionProvider"]
    elif target == "vitisai":
        candidates = ["VitisAIExecutionProvider"]
    elif target == "cpu":
        candidates = ["CPUExecutionProvider"]
    else:
        candidates = [*AMD_PROVIDERS]

    for provider in candidates:
        if provider in available_set:
            return provider
    raise RuntimeError(
        f"Requested target '{target}' is unavailable. Available EPs: {sorted(available_set)}. / "
        f"请求的目标“{target}”不可用；当前 EP：{sorted(available_set)}。"
    )


def initialize_windows_ml_migraphx(ort: Any, device_id: int) -> list[Any]:
    verify_runtime_artifact("MIGraphXExecutionProvider", ort, windows_ml=True)
    try:
        winml = importlib.import_module("winui3.microsoft.windows.ai.machinelearning")
    except Exception as exc:
        fail(
            "Windows ML Python packages are missing or the matching Windows App Runtime is not "
            f"installed: {exc}. Follow Part B, section 10. / Windows ML 依赖缺失或 Runtime 不匹配。"
        )

    try:
        catalog = winml.ExecutionProviderCatalog.get_default()
        catalog_providers = list(catalog.find_all_providers())
    except Exception as exc:
        fail(f"Windows ML could not enumerate execution providers: {exc}")
    provider = next(
        (item for item in catalog_providers if item.name == "MIGraphXExecutionProvider"),
        None,
    )
    if provider is None:
        names = [item.name for item in catalog_providers]
        fail(
            "Windows ML reports no compatible AMD MIGraphX provider. Check Windows 11 24H2, "
            f"the exact AMD driver requirement, Windows Update, and the live EP table. Catalog: {names}"
        )
    certified = getattr(getattr(winml, "ExecutionProviderCertification", None), "CERTIFIED", None)
    if certified is None or getattr(provider, "certification", None) != certified:
        fail("The Windows ML MIGraphX catalog entry is not marked Certified; refusing download.")

    info("Ensuring the Windows ML MIGraphX plugin is ready; a first download can take minutes.")
    try:
        result = provider.ensure_ready_async().get()
    except Exception as exc:
        fail(f"Windows ML could not acquire MIGraphX: {exc}")
    success = result.status == winml.ExecutionProviderReadyResultState.SUCCESS
    if not success:
        diagnostic = getattr(result, "diagnostic_text", "")
        extended = getattr(result, "extended_error", "")
        fail(f"Windows ML MIGraphX acquisition failed: {diagnostic} ({extended})")

    package_id = getattr(provider, "package_id", None)
    package_version = getattr(package_id, "version", None)
    try:
        actual_package_version = tuple(
            int(getattr(package_version, field))
            for field in ("major", "minor", "build", "revision")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        fail(f"Windows ML MIGraphX exposed no auditable MSIX package version: {exc}")
    if actual_package_version != WINDOWS_ML_MIGRAPHX_MSIX_VERSION:
        expected = ".".join(map(str, WINDOWS_ML_MIGRAPHX_MSIX_VERSION))
        actual = ".".join(map(str, actual_package_version))
        fail(
            f"Windows ML installed MIGraphX MSIX {actual}, but this 2026-07-16 audit requires "
            f"the currently supported {expected}. Recheck Microsoft's live EP table and update "
            "this guide before accepting another catalog release."
        )

    library_path = Path(provider.library_path)
    if not library_path.is_file():
        fail(f"Windows ML MIGraphX library path is not a regular file: {library_path}")
    info(
        "Verified Windows ML MIGraphX package/已验证 Windows ML 包: "
        + ".".join(map(str, actual_package_version))
    )

    try:
        ort.register_execution_provider_library(provider.name, str(library_path))
    except Exception as exc:
        fail(f"Could not register the Windows ML MIGraphX plugin with ONNX Runtime: {exc}")

    devices = [device for device in ort.get_ep_devices() if device.ep_name == provider.name]
    if not devices:
        fail("MIGraphX registered, but ONNX Runtime exposed no MIGraphX OrtEpDevice.")

    amd_devices: list[Any] = []
    gpu_type = getattr(getattr(ort, "OrtHardwareDeviceType", None), "GPU", None)
    if gpu_type is None:
        fail("This onnxruntime-windowsml build exposes no OrtHardwareDeviceType.GPU identity.")
    for index, device in enumerate(devices):
        hardware = getattr(device, "device", None)
        if hardware is None:
            hardware = getattr(device, "hardware_device", None)
        if hardware is None:
            fail("A MIGraphX OrtEpDevice exposed no hardware identity; refusing an AMD-device claim.")
        vendor_id = int(getattr(hardware, "vendor_id", -1))
        hardware_type = getattr(hardware, "type", None)
        info(
            "Windows ML EP device/Windows ML 设备: "
            f"index={index}, ep={device.ep_name}, hardware={hardware}, "
            f"type={hardware_type}, vendor_id=0x{vendor_id & 0xFFFFFFFF:04X}, "
            f"metadata={getattr(device, 'ep_metadata', {})}, "
            f"options={getattr(device, 'ep_options', {})}"
        )
        if vendor_id == AMD_PCI_VENDOR_ID and hardware_type == gpu_type:
            amd_devices.append(device)

    if not amd_devices:
        fail("MIGraphX registered, but none of its OrtEpDevices identifies as an AMD (0x1002) GPU.")
    if device_id >= len(amd_devices):
        fail(
            f"Windows ML AMD MIGraphX device index {device_id} does not exist; "
            f"valid range is 0..{len(amd_devices) - 1}."
        )
    return [amd_devices[device_id]]


def locate_ryzen_ai_quicktest_model() -> Path | None:
    roots: list[Path] = []
    install_path = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
    if install_path:
        roots.append(Path(install_path))
    roots.extend([Path(sys.prefix), Path(sys.executable).resolve().parent.parent])

    relative_candidates = (
        Path("quicktest/test_model.onnx"),
        Path("quicktest/model.onnx"),
    )
    for root in roots:
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def create_gpu_smoke_model(model_path: Path) -> None:
    model_bytes = base64.b64decode(SMOKE_MODEL_BASE64, validate=True)
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != SMOKE_MODEL_SHA256:
        fail(f"Embedded smoke-model integrity check failed: {digest}")
    model_path.write_bytes(model_bytes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_artifact(provider: str, ort: Any, *, windows_ml: bool) -> None:
    if windows_ml:
        require_distribution_version("onnxruntime-windowsml", WINDOWS_ML_ORT_DISTRIBUTION_VERSION)
        require_distribution_version(
            "wasdk-Microsoft.Windows.AI.MachineLearning",
            WINDOWS_ML_WASDK_VERSION,
        )
        require_distribution_version(
            "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap",
            WINDOWS_ML_WASDK_VERSION,
        )
        return

    package_root = Path(ort.__file__).resolve().parent
    if provider == "DmlExecutionProvider":
        require_distribution_version("onnxruntime-directml", DIRECTML_ORT_VERSION)
        artifact = package_root / "capi" / "DirectML.dll"
        expected_sha256 = DIRECTML_DLL_SHA256
        artifact_name = "Microsoft DirectML.dll"
    elif provider == "MIGraphXExecutionProvider" and platform.system() == "Linux":
        release = canonical_rocm_release(detect_rocm_version() or "")
        if release is None:
            fail("Cannot match the installed ROCm release to an audited MIGraphX provider binary.")
        require_distribution_version("onnxruntime-migraphx", MIGRAPHX_ORT_BY_ROCM[release])
        artifact = package_root / "capi" / "libonnxruntime_providers_migraphx.so"
        expected_sha256 = MIGRAPHX_PROVIDER_SHA256_BY_ROCM[release]
        artifact_name = f"AMD ROCm {release} MIGraphX provider"
    else:
        return

    if not artifact.is_file() or artifact.is_symlink():
        fail(f"Required audited runtime artifact is missing or not a regular file: {artifact}")
    actual_sha256 = sha256_file(artifact)
    if actual_sha256 != expected_sha256:
        fail(
            f"{artifact_name} does not match the audited release artifact. "
            f"Expected SHA-256 {expected_sha256}, got {actual_sha256}. Recreate the environment "
            "from this guide; do not use a same-named wheel from another package source."
        )
    info(f"Verified installed runtime artifact/已验证运行时: {artifact_name}, sha256={actual_sha256}")


def parse_shape_overrides(values: list[str]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for value in values:
        if "=" not in value:
            fail(f"Invalid --shape '{value}'; expected NAME=1,3,224,224.")
        name, raw_shape = value.split("=", 1)
        if name in result:
            fail(f"Duplicate --shape override for input '{name}'.")
        try:
            shape = tuple(int(item) for item in raw_shape.split(","))
        except ValueError:
            fail(f"Invalid integer in --shape '{value}'.")
        if not name or not shape or any(dim <= 0 for dim in shape):
            fail(f"Invalid --shape '{value}'; dimensions must be positive.")
        result[name] = shape
    return result


def numpy_dtype_for_ort_type(type_name: str, np: Any) -> Any:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(int16)": np.int16,
        "tensor(uint16)": np.uint16,
        "tensor(int32)": np.int32,
        "tensor(uint32)": np.uint32,
        "tensor(int64)": np.int64,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }
    dtype = mapping.get(type_name)
    if dtype is None:
        fail(f"Unsupported demo input type: {type_name}. Supply a model-specific runner.")
    return dtype


def make_inputs(session: Any, shape_overrides: dict[str, tuple[int, ...]], np: Any) -> dict[str, Any]:
    rng = np.random.default_rng(20260715)
    feeds: dict[str, Any] = {}
    input_metas = session.get_inputs()
    input_names = {meta.name for meta in input_metas}
    unknown_overrides = sorted(set(shape_overrides) - input_names)
    if unknown_overrides:
        fail(
            f"--shape names do not match model inputs: {unknown_overrides}. "
            f"Valid inputs: {sorted(input_names)}"
        )

    for meta in input_metas:
        shape = shape_overrides.get(meta.name)
        if shape is None:
            dynamic_dims = [dim for dim in meta.shape if not isinstance(dim, int) or dim <= 0]
            if dynamic_dims:
                fail(
                    f"Input '{meta.name}' has dynamic shape {meta.shape}. Supply an explicit "
                    f"--shape {meta.name}=... override. / 动态输入必须显式指定 --shape。"
                )
            shape = tuple(int(dim) for dim in meta.shape)
        else:
            if len(shape) != len(meta.shape):
                fail(
                    f"--shape for '{meta.name}' has rank {len(shape)}, but the model expects "
                    f"rank {len(meta.shape)} ({meta.shape})."
                )
            for axis, (actual, expected) in enumerate(zip(shape, meta.shape)):
                if isinstance(expected, int) and expected > 0 and actual != expected:
                    fail(
                        f"--shape for '{meta.name}' changes fixed axis {axis} from {expected} "
                        f"to {actual}."
                    )
        dtype = numpy_dtype_for_ort_type(meta.type, np)
        if np.issubdtype(dtype, np.floating):
            value = rng.uniform(-1.0, 1.0, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            value = np.zeros(shape, dtype=dtype)
        else:
            # Zero is the safest generic value for IDs, masks, lengths, and shape-like inputs.
            # Semantically correlated inputs still require a model-specific runner.
            value = np.zeros(shape, dtype=dtype)
        feeds[meta.name] = np.ascontiguousarray(value)
        info(f"Input/输入 {meta.name}: shape={shape}, dtype={value.dtype}")
    return feeds


def detect_npu_type() -> str:
    system = platform.system()
    if system == "Windows":
        command = ["pnputil", "/enum-devices", "/bus", "PCI", "/deviceids"]
    elif system == "Linux":
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            return "UNSUPPORTED"
        command = ["lspci", "-nn"]
    else:
        return "UNSUPPORTED"
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="ignore",
        )
        text = result.stdout.upper()
    except OSError:
        return "UNKNOWN"
    if system == "Windows" and "VEN_1022&DEV_1502" in text:
        return "PHX/HPT"
    if (system == "Windows" and "VEN_1022&DEV_17F0" in text) or (
        system == "Linux" and "1022:17F0" in text
    ):
        return "STX/KRK"
    return "UNKNOWN"


def enumerate_dxgi_adapters() -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []

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
                raise OSError(f"IDXGIFactory::EnumAdapters failed at {index}: 0x{result & 0xFFFFFFFF:08X}")
            get_desc = com_method(adapter, 8, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC))
            release_adapter = com_method(adapter, 2, ctypes.c_ulong)
            try:
                desc = DXGI_ADAPTER_DESC()
                result = get_desc(adapter, ctypes.byref(desc))
                if result < 0:
                    raise OSError(f"IDXGIAdapter::GetDesc failed: 0x{result & 0xFFFFFFFF:08X}")
                adapters.append(
                    {
                        "index": index,
                        "name": desc.Description.rstrip("\x00"),
                        "vendor_id": int(desc.VendorId),
                        "device_id": int(desc.DeviceId),
                        "dedicated_video_memory": int(desc.DedicatedVideoMemory),
                    }
                )
            finally:
                release_adapter(adapter)
            index += 1
    finally:
        release_factory(factory)
    return adapters


def verify_windows_directml_hardware(device_id: int) -> None:
    if sys.getwindowsversion().build < 18362:
        fail("DirectML requires Windows 10 version 1903 (build 18362) or newer.")
    try:
        adapters = enumerate_dxgi_adapters()
    except (AttributeError, OSError) as exc:
        fail(f"Could not enumerate DXGI adapters, so AMD adapter identity cannot be proven: {exc}")
    if device_id >= len(adapters):
        fail(f"DirectML device_id={device_id} does not exist. DXGI adapters: {adapters}")
    for adapter in adapters:
        info(
            "DXGI adapter/显示适配器 "
            f"{adapter['index']}: {adapter['name']}, vendor=0x{adapter['vendor_id']:04X}, "
            f"device=0x{adapter['device_id']:04X}, "
            f"dedicated={adapter['dedicated_video_memory'] / (1024 ** 2):.0f} MiB"
        )
    selected = adapters[device_id]
    if selected["vendor_id"] != AMD_PCI_VENDOR_ID:
        fail(
            f"DirectML device_id={device_id} is '{selected['name']}' with PCI vendor "
            f"0x{selected['vendor_id']:04X}, not AMD (0x{AMD_PCI_VENDOR_ID:04X}). "
            "Select the AMD DXGI index with --device-id."
        )


def verify_linux_migraphx_hardware() -> None:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        fail(
            "AMD's current WSL guidance explicitly marks MIGraphX as unsupported. "
            "Use native Ubuntu for MIGraphX, or use DirectML/Windows ML on Windows. "
            "/ AMD 当前 WSL 指南明确标注不支持 MIGraphX；请使用原生 Ubuntu、"
            "Windows DirectML 或 Windows ML。"
        )

    kfd = Path("/dev/kfd")
    render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
    if not kfd.exists() or not render_nodes:
        fail("Missing /dev/kfd or /dev/dri/renderD*. The AMD kernel driver is not ready.")
    if not os.access(kfd, os.R_OK | os.W_OK):
        fail(
            "The current user cannot access /dev/kfd. Add the user to render,video and "
            "log out or reboot."
        )
    if not any(os.access(path, os.R_OK | os.W_OK) for path in render_nodes):
        fail(
            f"The current user cannot access any render node ({render_nodes}). Add the user to "
            "render,video and log out or reboot."
        )


def verify_linux_ryzen_ai_platform() -> None:
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        fail("Ryzen AI for Linux 1.7.1 supports x86-64 STX/KRK PCs, not Arm Adaptive SoCs.")
    if sys.version_info[:2] != (3, 12):
        fail(
            "Ryzen AI for Linux 1.7.1 requires Python 3.12.x; detected "
            f"{platform.python_version()}. Activate the installer-created Linux venv."
        )

    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    except OSError as exc:
        fail(f"Cannot read /etc/os-release to verify the Ryzen AI Linux platform: {exc}")
    if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "24.04":
        fail(
            "Ryzen AI 1.7.1 NPU support is documented only for Ubuntu 24.04 LTS; "
            f"detected {os_release.get('PRETTY_NAME', os_release)}."
        )

    kernel_match = re.match(r"^(\d+)\.(\d+)", platform.release())
    if not kernel_match or tuple(map(int, kernel_match.groups())) < (6, 10):
        fail(
            f"Ryzen AI 1.7.1 requires Linux kernel >= 6.10; detected {platform.release()}. "
            "Install Ubuntu's supported HWE/OEM kernel, reboot, and retry."
        )
    if not Path("/opt/xilinx/xrt/setup.sh").is_file():
        fail("XRT is missing at /opt/xilinx/xrt/setup.sh. Install the Ryzen AI 1.7.1 NPU/XRT bundle.")


def provider_configuration(
    provider: str,
    device_id: int,
    model_path: Path,
    cache_root: Path,
    ort: Any,
    plugin_devices: list[Any] | None = None,
) -> tuple[Any, list[Any] | None, Path | None]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(cache_root / "amd_ort_profile")
    report_path: Path | None = None

    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers: list[Any] = [
            (provider, {"device_id": str(device_id)}),
            "CPUExecutionProvider",
        ]
    elif provider == "MIGraphXExecutionProvider" and plugin_devices:
        if not hasattr(options, "add_provider_for_devices"):
            fail("This onnxruntime-windowsml build lacks SessionOptions.add_provider_for_devices().")
        options.add_provider_for_devices(plugin_devices, {})
        providers = None
    elif provider == "MIGraphXExecutionProvider":
        providers = [
            (provider, {"device_id": str(device_id)}),
            "CPUExecutionProvider",
        ]
    elif provider == "VitisAIExecutionProvider":
        digest = sha256_file(model_path)[:16]
        vai_cache = cache_root / f"vitisai-{digest}"
        vai_cache.mkdir(parents=True, exist_ok=True)
        vai_options: dict[str, str] = {
            "cache_dir": str(vai_cache),
            "cache_key": "amd-ort-oneclick",
            "enable_cache_file_io_in_mem": "0",
        }
        npu_type = detect_npu_type()
        info(f"Detected NPU family/检测到 NPU 系列: {npu_type}")
        if npu_type == "PHX/HPT":
            install_root = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
            if not install_root:
                fail("RYZEN_AI_INSTALLATION_PATH is required for PHX/HPT xclbin lookup.")
            xclbin = (
                Path(install_root)
                / "voe-4.0-win_amd64"
                / "xclbins"
                / "phoenix"
                / "4x4.xclbin"
            )
            if not xclbin.is_file():
                fail(f"Required PHX/HPT xclbin not found: {xclbin}")
            vai_options.update(
                {
                    "target": "X1",
                    "xclbin": str(xclbin),
                    "xlnx_enable_py3_round": "0",
                }
            )
        elif npu_type == "STX/KRK":
            vai_options["target"] = "X2"
        else:
            fail(
                "Could not identify a supported Ryzen AI NPU (PHX/HPT/STX/KRK). On Linux, "
                "this verifier supports x86-64 STX/KRK PCs only, not Adaptive SoCs."
            )

        report_path = (cache_root / "amd_ort_assignment.json").resolve()
        # AMD documents this variable as a report file name generated beneath the
        # configured cache directory. A recursive lookup remains inside this run's
        # new cache root, so nested provider layouts cannot introduce stale evidence.
        os.environ["XLNX_ONNX_EP_REPORT_FILE"] = report_path.name
        providers = [(provider, vai_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return options, providers, report_path


def configure_strict_offload(options: Any, provider: str, strict_all: bool) -> None:
    if strict_all and provider != "CPUExecutionProvider":
        # Make unsupported CPU placement fail during session creation. The
        # profile/report checks remain independent current-run evidence.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")


def provider_counts_from_profile(profile_path: Path) -> tuple[dict[str, int], bool]:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"Could not parse ORT profile {profile_path}: {exc}")
        return {}, False
    if not isinstance(payload, list):
        warn(f"ORT profile is not a JSON event list: {profile_path}")
        return {}, False
    counts: dict[str, int] = {}
    for event in payload:
        if (
            not isinstance(event, dict)
            or event.get("cat") != "Node"
            or "dur" not in event
            or not str(event.get("name", "")).endswith("_kernel_time")
        ):
            continue
        args = event.get("args") if isinstance(event, dict) else None
        provider = args.get("provider") if isinstance(args, dict) else None
        op_name = args.get("op_name") if isinstance(args, dict) else None
        if isinstance(provider, str) and provider and isinstance(op_name, str) and op_name:
            counts[provider] = counts.get(provider, 0) + 1
    return counts, True


def assignment_counts_from_report(report_path: Path | None) -> dict[str, int]:
    if report_path is None:
        return {}
    candidates = [report_path]
    try:
        candidates.extend(
            candidate
            for candidate in sorted(report_path.parent.rglob(report_path.name))
            if candidate != report_path
        )
    except OSError:
        pass

    valid_reports: list[tuple[Path, dict[str, int]]] = []
    invalid_reports: list[tuple[Path, str]] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            invalid_reports.append((candidate, "not a regular file"))
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid_reports.append((candidate, f"unreadable or invalid JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            invalid_reports.append((candidate, "top-level value is not an object"))
            continue
        device_stats = payload.get("deviceStat")
        if not isinstance(device_stats, list):
            invalid_reports.append((candidate, "deviceStat is not a list"))
            continue
        counts: dict[str, int] = {}
        report_error: str | None = None
        for item in device_stats:
            if not isinstance(item, dict):
                report_error = "deviceStat contains a non-object entry"
                break
            raw_name = item.get("name")
            name = raw_name.upper() if isinstance(raw_name, str) else ""
            if not name:
                report_error = "deviceStat contains an empty or non-string name"
                break
            if name in counts:
                report_error = f"duplicate deviceStat entry '{name}'"
                break
            node_count = item.get("nodeNum")
            if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 0:
                report_error = f"invalid nodeNum for '{name}': {node_count!r}"
                break
            counts[name] = node_count
        required = {"ALL", "CPU", "NPU"}
        if report_error is None and not required.issubset(counts):
            report_error = f"missing required deviceStat entries: {sorted(required - counts.keys())}"
        if report_error is None and counts["ALL"] != counts["CPU"] + counts["NPU"]:
            report_error = (
                "inconsistent node totals: "
                f"ALL={counts['ALL']}, CPU={counts['CPU']}, NPU={counts['NPU']}"
            )
        if report_error is not None:
            invalid_reports.append((candidate, report_error))
            continue
        valid_reports.append((candidate, counts))

    if invalid_reports:
        fail(
            "A fresh Vitis AI assignment report is malformed, so node placement cannot be proven: "
            + "; ".join(f"{path} ({reason})" for path, reason in invalid_reports)
        )

    if not valid_reports:
        return {}
    distinct_counts = {tuple(sorted(counts.items())) for _, counts in valid_reports}
    if len(distinct_counts) > 1:
        fail(
            "Fresh Vitis AI report files disagree, so node placement cannot be proven: "
            + ", ".join(str(path) for path, _ in valid_reports)
        )
    selected_path, selected_counts = valid_reports[0]
    info(f"Vitis AI assignment report/Vitis AI 分配报告: {selected_path}")
    return selected_counts


def validate_outputs(outputs: list[Any], np: Any) -> None:
    if not outputs:
        fail("Inference returned no outputs. / 推理没有返回输出。")
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        if array.size == 0:
            fail(f"Output {index} is empty. / 输出 {index} 为空。")
        if array.dtype.hasobject:
            fail(
                f"Output {index} is an object/sequence/map value that the generic verifier cannot "
                "validate safely. Use a model-specific runner. / 通用验证器无法安全检查对象类型输出。"
            )
        if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
            if not np.all(np.isfinite(array)):
                fail(f"Output {index} contains NaN or infinity. / 输出 {index} 包含 NaN 或无穷值。")


def output_summary(outputs: list[Any], np: Any) -> None:
    for index, output in enumerate(outputs):
        array = np.asarray(output)
        digest = hashlib.sha256(array.tobytes()).hexdigest()[:16]
        info(f"Output/输出 {index}: shape={array.shape}, dtype={array.dtype}, sha256={digest}")


def verify_outputs_against_cpu(
    model_path: Path,
    feeds: dict[str, Any],
    accelerated_outputs: list[Any],
    ort: Any,
    np: Any,
    rtol: float,
    atol: float,
) -> None:
    try:
        cpu_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        cpu_outputs = cpu_session.run(None, feeds)
    except Exception as exc:
        fail(f"CPU reference run failed: {exc}")
    validate_outputs(cpu_outputs, np)
    if len(cpu_outputs) != len(accelerated_outputs):
        fail("Accelerator and CPU reference returned different output counts.")
    for index, (actual_raw, expected_raw) in enumerate(zip(accelerated_outputs, cpu_outputs)):
        actual = np.asarray(actual_raw)
        expected = np.asarray(expected_raw)
        if actual.shape != expected.shape:
            fail(f"Output {index} shape differs from CPU: {actual.shape} vs {expected.shape}")
        if actual.dtype != expected.dtype:
            fail(f"Output {index} dtype differs from CPU: {actual.dtype} vs {expected.dtype}")
        if np.issubdtype(actual.dtype, np.inexact):
            if not np.all(np.isfinite(actual)):
                fail(f"Accelerated output {index} contains NaN or infinity.")
            if not np.allclose(actual, expected, rtol=rtol, atol=atol):
                max_diff = float(np.max(np.abs(actual.astype(np.complex128) - expected.astype(np.complex128))))
                fail(
                    f"Accelerated output {index} disagrees with CPU reference; "
                    f"max_abs_diff={max_diff}, rtol={rtol}, atol={atol}"
                )
            max_diff = float(np.max(np.abs(actual.astype(np.complex128) - expected.astype(np.complex128))))
            info(f"CPU reference parity/CPU 参考一致性 output={index}, max_abs_diff={max_diff:.3g}")
        elif not np.array_equal(actual, expected):
            fail(f"Accelerated integer/Boolean output {index} disagrees with the CPU reference.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and prove AMD GPU/NPU inference through ONNX Runtime.",
    )
    parser.add_argument(
        "--unit-tests",
        action="store_true",
        help="Run the verifier's built-in unit tests without requiring AMD hardware, then exit.",
    )
    parser.add_argument(
        "--target",
        choices=("auto", "gpu", "npu", "migraphx", "dml", "vitisai", "cpu"),
        default="auto",
        help="Requested device class or exact EP (default: auto).",
    )
    parser.add_argument("--model", type=Path, help="Optional ONNX model. A smoke model is used by default.")
    parser.add_argument("--device-id", type=int, default=0, help="GPU adapter/device index (default: 0).")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs (default: 2).")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs (default: 10).")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME=D0,D1,...",
        help="Override a dynamic input shape; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Install the appropriate GPU Python wheel when missing (drivers are never installed).",
    )
    parser.add_argument(
        "--windows-ml",
        action="store_true",
        help="Acquire/register the Windows ML AMD MIGraphX plugin in this process (Windows GPU only).",
    )
    parser.add_argument(
        "--rocm-version",
        help="Assert the detected ROCm wheel repository version, for example 7.2.4.",
    )
    parser.add_argument(
        "--strict-all",
        action="store_true",
        help="Fail if profiling or a Vitis assignment report detects any CPU fallback.",
    )
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="Compare a user-supplied model's outputs with a separate CPU EP run.",
    )
    parser.add_argument("--rtol", type=float, default=1e-4, help="CPU comparison relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-5, help="CPU comparison absolute tolerance.")
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> int:
    if args.device_id < 0 or args.warmup < 0 or args.runs < 1:
        fail("--device-id/--warmup must be non-negative and --runs must be positive.")
    if args.windows_ml and args.bootstrap:
        fail(
            "--windows-ml and --bootstrap cannot be combined. Install the pinned Windows ML "
            "packages and matching Windows App Runtime from section 10 first."
        )
    if args.windows_ml and (platform.system() != "Windows" or args.target not in {"auto", "gpu", "migraphx"}):
        fail("--windows-ml currently verifies the Windows AMD MIGraphX GPU plugin only.")
    if args.windows_ml and platform.machine().lower() not in {"amd64", "x86_64"}:
        fail("This audited Windows ML AMD MIGraphX recipe requires an x64 Python process.")
    if args.windows_ml and sys.getwindowsversion().build < 26100:
        fail("Dynamic Windows ML hardware EPs require Windows 11 24H2 (build 26100) or newer.")
    if args.rtol < 0 or args.atol < 0 or not math.isfinite(args.rtol) or not math.isfinite(args.atol):
        fail("--rtol and --atol must be finite, non-negative numbers.")

    distributions = installed_ort_distributions()
    if len(distributions) > 1:
        fail(
            f"Multiple ONNX Runtime distributions are installed: {sorted(distributions)}. "
            "Create a clean environment with exactly one runtime package."
        )

    try:
        np, ort = import_runtime()
    except RuntimeError as exc:
        if args.bootstrap and os.environ.get("AMD_ORT_BOOTSTRAPPED") != "1":
            pip_install_for_target(args.target, args.rocm_version)
        fail(str(exc))

    plugin_devices: list[Any] | None = None
    if args.windows_ml:
        provider = "MIGraphXExecutionProvider"
        plugin_devices = initialize_windows_ml_migraphx(ort, args.device_id)
        available = sorted({device.ep_name for device in ort.get_ep_devices()})
    else:
        available = ort.get_available_providers()
        try:
            provider = choose_provider(args.target, available)
        except RuntimeError as exc:
            if args.bootstrap and args.target != "npu" and os.environ.get("AMD_ORT_BOOTSTRAPPED") != "1":
                pip_install_for_target(args.target, args.rocm_version)
            fail(str(exc))

    info(f"OS={platform.platform()}")
    info(f"Python={platform.python_version()}, ONNX Runtime={ort.__version__}")
    info(f"Available EPs/可用 EP: {available}")
    info(f"Selected EP/已选择 EP: {provider}")
    if not args.windows_ml:
        verify_runtime_artifact(provider, ort, windows_ml=False)
    if provider == "DmlExecutionProvider":
        verify_windows_directml_hardware(args.device_id)
    elif provider == "MIGraphXExecutionProvider" and platform.system() == "Linux":
        verify_linux_migraphx_hardware()
    elif provider == "VitisAIExecutionProvider" and platform.system() == "Linux":
        verify_linux_ryzen_ai_platform()

    shape_overrides = parse_shape_overrides(args.shape)
    with tempfile.TemporaryDirectory(prefix="amd-ort-oneclick-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        generated_smoke_model = False
        if args.model:
            model_path = args.model.expanduser().resolve()
            if not model_path.is_file():
                fail(f"Model not found: {model_path}")
        elif provider == "VitisAIExecutionProvider":
            model_path = locate_ryzen_ai_quicktest_model()  # type: ignore[assignment]
            if model_path is None:
                fail(
                    "Cannot find the vendor quicktest model. Pass --model or run from an activated "
                    "Ryzen AI environment. / 找不到厂商 quicktest 模型；请传入 --model 或激活 Ryzen AI 环境。"
                )
            info(f"Using Ryzen AI quicktest model/使用 Ryzen AI quicktest 模型: {model_path}")
        else:
            model_path = temp_dir / "amd_gpu_smoke.onnx"
            create_gpu_smoke_model(model_path)
            generated_smoke_model = True
            info(f"Created smoke model/已创建测试模型: {model_path}")

        cache_base = Path(os.environ.get("AMD_ORT_DEMO_CACHE", Path.home() / ".cache" / "amd-ort-oneclick"))
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}"
        cache_root = cache_base / "runs" / run_id
        cache_root.mkdir(parents=True, exist_ok=False)
        session_options, providers, report_path = provider_configuration(
            provider,
            args.device_id,
            model_path,
            cache_root,
            ort,
            plugin_devices,
        )
        configure_strict_offload(session_options, provider, args.strict_all)

        if providers is None:
            info("Provider selection/EP 选择: explicit Windows ML OrtEpDevice in SessionOptions")
        else:
            info(f"Provider order/EP 优先级: {providers}")
        try:
            if providers is None:
                session = ort.InferenceSession(str(model_path), sess_options=session_options)
            else:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                    providers=providers,
                )
        except Exception as exc:
            fail(f"Session creation failed: {exc} / 会话创建失败。", exit_code=3)

        if provider != "CPUExecutionProvider":
            session.disable_fallback()

        info(f"Session EPs/会话 EP: {session.get_providers()}")
        info(f"Provider options/EP 选项: {session.get_provider_options()}")
        feeds = make_inputs(session, shape_overrides, np)

        outputs: list[Any] = []
        try:
            for _ in range(args.warmup):
                outputs = session.run(None, feeds)
                validate_outputs(outputs, np)
            elapsed = 0.0
            for _ in range(args.runs):
                start = time.perf_counter()
                outputs = session.run(None, feeds)
                elapsed += time.perf_counter() - start
                validate_outputs(outputs, np)
        except Exception as exc:
            fail(f"Inference failed: {exc} / 推理失败。", exit_code=4)

        output_summary(outputs, np)
        if generated_smoke_model and provider != "CPUExecutionProvider":
            verify_outputs_against_cpu(model_path, feeds, outputs, ort, np, args.rtol, args.atol)
        elif args.compare_cpu and provider != "CPUExecutionProvider":
            verify_outputs_against_cpu(model_path, feeds, outputs, ort, np, args.rtol, args.atol)
        elif args.compare_cpu:
            warn("--compare-cpu is redundant for a CPU-only run and was skipped.")
        mean_ms = elapsed * 1000.0 / args.runs
        info(f"Mean synchronous latency/平均同步延迟: {mean_ms:.3f} ms ({args.runs} runs)")

        profile_path = Path(session.end_profiling())
        profile_counts, _profile_valid = provider_counts_from_profile(profile_path)
        info(f"Profiled node events by EP/按 EP 统计的节点事件: {profile_counts}")

        target_events = profile_counts.get(provider, 0)
        assignment_counts: dict[str, int] = {}
        if provider == "VitisAIExecutionProvider":
            assignment_counts = assignment_counts_from_report(report_path)
            if assignment_counts:
                info(f"Assigned graph nodes by device/按设备分配的图节点: {assignment_counts}")

        assigned_npu_nodes = assignment_counts.get("NPU", 0)
        if target_events <= 0 and assigned_npu_nodes <= 0:
            fail(
                f"'{provider}' was registered, but no executed node was attributed to it. "
                "This is CPU fallback, not verified acceleration. / EP 虽已注册，但没有节点归属于该 EP；"
                "这属于 CPU 回退，不能证明硬件加速。",
                exit_code=5,
            )

        cpu_events = profile_counts.get("CPUExecutionProvider", 0)
        assigned_cpu_nodes = assignment_counts.get("CPU", 0)
        if args.strict_all and provider != "CPUExecutionProvider":
            if cpu_events or assigned_cpu_nodes:
                fail(
                    "--strict-all requested, but CPU fallback was detected: "
                    f"profile_events={cpu_events}, assigned_graph_nodes={assigned_cpu_nodes}. / "
                    "--strict-all 已启用，但检测到 CPU 回退。",
                    exit_code=6,
                )
        if provider == "CPUExecutionProvider":
            print(
                f"[PASS/通过] CPU self-test completed with {target_events} profiled node event(s). "
                f"CPU 自检完成，共记录 {target_events} 个节点事件。",
                flush=True,
            )
        elif target_events > 0:
            print(
                f"[PASS/通过] Runtime profile verified {target_events} executed node event(s) on "
                f"{provider}. 已通过运行时 profile 验证 {target_events} 个加速节点事件。",
                flush=True,
            )
        else:
            print(
                f"[PASS/通过] Inference succeeded and the fresh Vitis AI report assigns "
                f"{assigned_npu_nodes} graph node(s) to the NPU. 推理成功，且本次新生成的报告将 "
                f"{assigned_npu_nodes} 个图节点分配到 NPU。",
                flush=True,
            )
        if (cpu_events or assigned_cpu_nodes) and provider != "CPUExecutionProvider":
            warn(
                "CPU fallback is present: "
                f"profile_events={cpu_events}, assigned_graph_nodes={assigned_cpu_nodes}. "
                "Use --strict-all to reject partial offload. / 存在部分 CPU 回退。"
            )
        info(f"ORT profile/性能档案: {profile_path}")
        info(f"Run artifacts/本次运行产物: {cache_root}")
    return 0


def run_unit_tests() -> int:
    """Run deterministic verifier tests without requiring AMD hardware."""
    import unittest
    from types import SimpleNamespace
    from unittest import mock

    module = sys.modules[__name__]

    class VersionAndSelectionTests(unittest.TestCase):
        def test_package_names_are_canonicalized(self) -> None:
            self.assertEqual(canonical_package_name("ONNXRuntime_MIGraphX"), "onnxruntime-migraphx")

        def test_rocm_release_mapping_is_explicit(self) -> None:
            self.assertEqual(canonical_rocm_release("7.2.4"), "7.2.4")
            self.assertEqual(canonical_rocm_release("7.2.1"), "7.2.1")
            for value in ("7.2.0", "7.2.3", "8.0", "latest"):
                with self.subTest(value=value):
                    self.assertIsNone(canonical_rocm_release(value))

        def test_migraphx_wheel_is_pinned_to_amd_repository(self) -> None:
            name, url, digest = migraphx_wheel_details("7.2.4", (3, 12))
            self.assertEqual(
                name,
                "onnxruntime_migraphx-1.23.2-cp312-cp312-"
                "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            )
            self.assertEqual(url, f"https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/{name}")
            self.assertEqual(digest, "ee8edeb2ba6a8d99b3043b23e812423e6f10333b508e003fc77b0feda197449f")
            with self.assertRaises(ValueError):
                migraphx_wheel_details("7.2.4", (3, 11))

        def test_directml_wheel_hash_is_pinned(self) -> None:
            self.assertEqual(DIRECTML_CP312_WHEEL, "onnxruntime_directml-1.24.4-cp312-cp312-win_amd64.whl")
            self.assertEqual(
                DIRECTML_CP312_SHA256,
                "f2ecb68b7b7b259d2ef3112ae760149f9b5a1e7c0fbb73d539da6250a648a614",
            )
            self.assertEqual(
                DIRECTML_DLL_SHA256,
                "b73972115320e906a49602f2027a3266622881b0d325ba685e0f165a9482a8d7",
            )

        def test_all_artifact_fingerprints_are_sha256(self) -> None:
            fingerprints = [
                DIRECTML_CP312_SHA256,
                DIRECTML_DLL_SHA256,
                *MIGRAPHX_WHEEL_SHA256.values(),
                *MIGRAPHX_PROVIDER_SHA256_BY_ROCM.values(),
            ]
            self.assertEqual(len(fingerprints), 8)
            for fingerprint in fingerprints:
                with self.subTest(fingerprint=fingerprint):
                    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", fingerprint))

        def test_auto_provider_priority_is_npu_then_gpu(self) -> None:
            available = ["CPUExecutionProvider", "DmlExecutionProvider", "VitisAIExecutionProvider"]
            self.assertEqual(choose_provider("auto", available), "VitisAIExecutionProvider")
            self.assertEqual(choose_provider("gpu", available), "DmlExecutionProvider")

        def test_unavailable_provider_is_rejected(self) -> None:
            with self.assertRaises(RuntimeError):
                choose_provider("migraphx", ["CPUExecutionProvider"])

    class InputAndModelTests(unittest.TestCase):
        def test_shape_overrides(self) -> None:
            self.assertEqual(
                parse_shape_overrides(["images=1,3,224,224", "mask=1,224,224"]),
                {"images": (1, 3, 224, 224), "mask": (1, 224, 224)},
            )

        def test_invalid_shape_is_rejected(self) -> None:
            for value in ("images", "images=1,0,224", "images=1,nope,224"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    parse_shape_overrides([value])

        def test_embedded_model_digest(self) -> None:
            model_bytes = base64.b64decode(SMOKE_MODEL_BASE64, validate=True)
            self.assertEqual(hashlib.sha256(model_bytes).hexdigest(), SMOKE_MODEL_SHA256)

    class HardwareDetectionTests(unittest.TestCase):
        @staticmethod
        def completed(stdout: str) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

        def test_phx_detection_does_not_depend_on_pci_revision(self) -> None:
            with (
                mock.patch.object(platform, "system", return_value="Windows"),
                mock.patch.object(
                    subprocess,
                    "run",
                    return_value=self.completed("PCI\\VEN_1022&DEV_1502&REV_42"),
                ),
            ):
                self.assertEqual(detect_npu_type(), "PHX/HPT")

        def test_linux_stx_detection(self) -> None:
            with (
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.object(platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    subprocess,
                    "run",
                    return_value=self.completed(
                        "c5:00.1 Signal processing controller [1180]: AMD [1022:17f0]"
                    ),
                ),
            ):
                self.assertEqual(detect_npu_type(), "STX/KRK")

        def test_wsl_migraphx_is_rejected_as_unsupported(self) -> None:
            with (
                mock.patch.object(platform, "release", return_value="5.15.153.1-microsoft-standard-WSL2"),
                self.assertRaises(SystemExit),
            ):
                verify_linux_migraphx_hardware()

        def test_linux_ryzen_ai_requires_python_312(self) -> None:
            with (
                mock.patch.object(platform, "machine", return_value="x86_64"),
                mock.patch.object(sys, "version_info", (3, 11, 9)),
                self.assertRaises(SystemExit),
            ):
                verify_linux_ryzen_ai_platform()

    class EvidenceParsingTests(unittest.TestCase):
        @staticmethod
        def assignment_report(npu: int, cpu: int = 0) -> dict[str, object]:
            return {
                "deviceStat": [
                    {"name": "all", "nodeNum": npu + cpu},
                    {"name": "CPU", "nodeNum": cpu},
                    {"name": "NPU", "nodeNum": npu},
                ]
            }

        def test_profile_provider_counts(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                profile = Path(raw_dir) / "profile.json"
                profile.write_text(
                    json.dumps(
                        [
                            {
                                "cat": "Node",
                                "name": "conv_kernel_time",
                                "dur": 10,
                                "args": {
                                    "provider": "MIGraphXExecutionProvider",
                                    "op_name": "Conv",
                                },
                            },
                            {
                                "cat": "Node",
                                "name": "relu_kernel_time",
                                "dur": 5,
                                "args": {
                                    "provider": "MIGraphXExecutionProvider",
                                    "op_name": "Relu",
                                },
                            },
                            {
                                "cat": "Node",
                                "name": "shape_kernel_time",
                                "dur": 2,
                                "args": {
                                    "provider": "CPUExecutionProvider",
                                    "op_name": "Shape",
                                },
                            },
                            {
                                "cat": "Session",
                                "dur": 20,
                                "args": {"provider": "MIGraphXExecutionProvider"},
                            },
                            {
                                "cat": "Node",
                                "name": "conv_fence_before",
                                "dur": 1,
                                "args": {
                                    "provider": "MIGraphXExecutionProvider",
                                    "op_name": "Conv",
                                },
                            },
                            {"cat": "Node", "name": "unknown_kernel_time", "dur": 1, "args": {}},
                        ]
                    ),
                    encoding="utf-8",
                )
                counts, valid = provider_counts_from_profile(profile)
            self.assertTrue(valid)
            self.assertEqual(counts, {"MIGraphXExecutionProvider": 2, "CPUExecutionProvider": 1})

        def test_nested_fresh_assignment_report_is_found(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                expected = root / "amd_ort_assignment.json"
                nested = root / "vitisai-model" / "amd-ort-oneclick"
                nested.mkdir(parents=True)
                (nested / expected.name).write_text(
                    json.dumps(self.assignment_report(3)),
                    encoding="utf-8",
                )
                counts = assignment_counts_from_report(expected)
            self.assertEqual(counts["CPU"], 0)
            self.assertEqual(counts["NPU"], 3)

        def test_disagreeing_assignment_reports_fail_closed(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                expected = root / "amd_ort_assignment.json"
                expected.write_text(json.dumps(self.assignment_report(3)), encoding="utf-8")
                nested = root / "nested"
                nested.mkdir()
                (nested / expected.name).write_text(
                    json.dumps(self.assignment_report(2)),
                    encoding="utf-8",
                )
                with self.assertRaises(SystemExit):
                    assignment_counts_from_report(expected)

        def test_malformed_assignment_reports_fail_closed(self) -> None:
            payloads = (
                "{not-json",
                json.dumps({"deviceStat": [{"name": "NPU", "nodeNum": 3}]}),
                json.dumps(
                    {
                        "deviceStat": [
                            {"name": "all", "nodeNum": 4},
                            {"name": "CPU", "nodeNum": 0},
                            {"name": "NPU", "nodeNum": 3},
                        ]
                    }
                ),
                json.dumps(
                    {
                        "deviceStat": [
                            {"name": "all", "nodeNum": -1},
                            {"name": "CPU", "nodeNum": 0},
                            {"name": "NPU", "nodeNum": -1},
                        ]
                    }
                ),
            )
            for payload in payloads:
                with self.subTest(payload=payload), tempfile.TemporaryDirectory() as raw_dir:
                    report = Path(raw_dir) / "amd_ort_assignment.json"
                    report.write_text(payload, encoding="utf-8")
                    with self.assertRaises(SystemExit):
                        assignment_counts_from_report(report)

        def test_non_list_profile_is_invalid(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                profile = Path(raw_dir) / "profile.json"
                profile.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
                counts, valid = provider_counts_from_profile(profile)
            self.assertFalse(valid)
            self.assertEqual(counts, {})

    class SafetyTests(unittest.TestCase):
        @staticmethod
        def fake_download(command: list[str]) -> subprocess.CompletedProcess[str]:
            if "download" in command:
                destination = Path(command[command.index("--dest") + 1])
                (destination / DIRECTML_CP312_WHEEL).write_bytes(b"staged wheel")
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        def test_directml_bootstrap_stages_and_hash_checks_before_install(self) -> None:
            with (
                mock.patch.object(module, "in_isolated_python_environment", return_value=True),
                mock.patch.object(module, "installed_ort_distributions", return_value=set()),
                mock.patch.object(platform, "system", return_value="Windows"),
                mock.patch.object(platform, "machine", return_value="AMD64"),
                mock.patch.object(sys, "version_info", (3, 12, 9)),
                mock.patch.object(module, "sha256_file", return_value=DIRECTML_CP312_SHA256),
                mock.patch.object(module, "run_command", side_effect=self.fake_download) as run_mock,
                mock.patch.object(os, "execv") as execv,
            ):
                pip_install_for_target("dml", None)
            self.assertEqual(run_mock.call_count, 2)
            download_command = run_mock.call_args_list[0].args[0]
            install_command = run_mock.call_args_list[1].args[0]
            self.assertIn("onnxruntime-directml==1.24.4", download_command)
            self.assertIn("--no-index", install_command)
            execv.assert_called_once()

        def test_directml_bootstrap_rejects_hash_mismatch_before_install(self) -> None:
            with (
                mock.patch.object(module, "in_isolated_python_environment", return_value=True),
                mock.patch.object(module, "installed_ort_distributions", return_value=set()),
                mock.patch.object(platform, "system", return_value="Windows"),
                mock.patch.object(platform, "machine", return_value="AMD64"),
                mock.patch.object(sys, "version_info", (3, 12, 9)),
                mock.patch.object(module, "sha256_file", return_value="0" * 64),
                mock.patch.object(module, "run_command", side_effect=self.fake_download) as run_mock,
                self.assertRaises(SystemExit),
            ):
                pip_install_for_target("dml", None)
            run_mock.assert_called_once()

        def test_installed_directml_binary_is_hash_checked(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                package = Path(raw_dir) / "onnxruntime"
                artifact = package / "capi" / "DirectML.dll"
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"runtime")
                ort = SimpleNamespace(__file__=str(package / "__init__.py"))
                with (
                    mock.patch.object(module, "require_distribution_version") as require_version,
                    mock.patch.object(module, "sha256_file", return_value=DIRECTML_DLL_SHA256),
                ):
                    verify_runtime_artifact("DmlExecutionProvider", ort, windows_ml=False)
            require_version.assert_called_once_with("onnxruntime-directml", DIRECTML_ORT_VERSION)

        def test_installed_runtime_hash_mismatch_fails_closed(self) -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                package = Path(raw_dir) / "onnxruntime"
                artifact = package / "capi" / "DirectML.dll"
                artifact.parent.mkdir(parents=True)
                artifact.write_bytes(b"runtime")
                ort = SimpleNamespace(__file__=str(package / "__init__.py"))
                with (
                    mock.patch.object(module, "require_distribution_version"),
                    mock.patch.object(module, "sha256_file", return_value="0" * 64),
                    self.assertRaises(SystemExit),
                ):
                    verify_runtime_artifact("DmlExecutionProvider", ort, windows_ml=False)

        def test_windows_ml_runtime_versions_are_pinned(self) -> None:
            with mock.patch.object(module, "require_distribution_version") as require_version:
                verify_runtime_artifact(
                    "MIGraphXExecutionProvider",
                    SimpleNamespace(__file__=__file__),
                    windows_ml=True,
                )
            self.assertEqual(
                require_version.call_args_list,
                [
                    mock.call("onnxruntime-windowsml", "1.24.6.202605042033"),
                    mock.call("wasdk-Microsoft.Windows.AI.MachineLearning", "2.1.3"),
                    mock.call(
                        "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap",
                        "2.1.3",
                    ),
                ],
            )

        @staticmethod
        def windows_ml_fixture(
            version: tuple[int, int, int, int],
            certification: str = "certified",
        ) -> tuple[Any, Any, Any, Any]:
            library_dir = tempfile.TemporaryDirectory()
            library = Path(library_dir.name) / "migraphx.dll"
            library.write_bytes(b"plugin")
            package_version = SimpleNamespace(
                major=version[0],
                minor=version[1],
                build=version[2],
                revision=version[3],
            )
            ready_result = SimpleNamespace(status="success")
            provider = SimpleNamespace(
                name="MIGraphXExecutionProvider",
                certification=certification,
                package_id=SimpleNamespace(version=package_version),
                library_path=str(library),
                ensure_ready_async=lambda: SimpleNamespace(get=lambda: ready_result),
            )
            winml = SimpleNamespace(
                ExecutionProviderCertification=SimpleNamespace(CERTIFIED="certified"),
                ExecutionProviderReadyResultState=SimpleNamespace(SUCCESS="success"),
                ExecutionProviderCatalog=SimpleNamespace(
                    get_default=lambda: SimpleNamespace(find_all_providers=lambda: [provider])
                ),
            )
            device = SimpleNamespace(
                ep_name=provider.name,
                device=SimpleNamespace(vendor_id=AMD_PCI_VENDOR_ID, type="gpu"),
                ep_metadata={},
                ep_options={},
            )
            ort = SimpleNamespace(
                OrtHardwareDeviceType=SimpleNamespace(GPU="gpu"),
                register_execution_provider_library=mock.Mock(),
                get_ep_devices=lambda: [device],
            )
            return library_dir, provider, winml, ort

        def test_windows_ml_migraphx_msix_version_is_verified_before_registration(self) -> None:
            library_dir, provider, winml, ort = self.windows_ml_fixture((1, 8, 57, 0))
            try:
                with (
                    mock.patch.object(module, "verify_runtime_artifact"),
                    mock.patch.object(importlib, "import_module", return_value=winml),
                ):
                    selected = initialize_windows_ml_migraphx(ort, 0)
                self.assertEqual(selected, [ort.get_ep_devices()[0]])
                ort.register_execution_provider_library.assert_called_once_with(
                    provider.name,
                    provider.library_path,
                )
            finally:
                library_dir.cleanup()

        def test_windows_ml_migraphx_rejects_stale_msix_before_registration(self) -> None:
            library_dir, _provider, winml, ort = self.windows_ml_fixture((1, 8, 56, 0))
            try:
                with (
                    mock.patch.object(module, "verify_runtime_artifact"),
                    mock.patch.object(importlib, "import_module", return_value=winml),
                    self.assertRaises(SystemExit),
                ):
                    initialize_windows_ml_migraphx(ort, 0)
                ort.register_execution_provider_library.assert_not_called()
            finally:
                library_dir.cleanup()

        def test_windows_ml_migraphx_rejects_uncertified_entry_before_download(self) -> None:
            ensure_ready = mock.Mock()
            provider = SimpleNamespace(
                name="MIGraphXExecutionProvider",
                certification="uncertified",
                ensure_ready_async=ensure_ready,
            )
            winml = SimpleNamespace(
                ExecutionProviderCertification=SimpleNamespace(CERTIFIED="certified"),
                ExecutionProviderCatalog=SimpleNamespace(
                    get_default=lambda: SimpleNamespace(find_all_providers=lambda: [provider])
                ),
            )
            with (
                mock.patch.object(module, "verify_runtime_artifact"),
                mock.patch.object(importlib, "import_module", return_value=winml),
                self.assertRaises(SystemExit),
            ):
                initialize_windows_ml_migraphx(SimpleNamespace(), 0)
            ensure_ready.assert_not_called()

        def test_bootstrap_rejects_wsl_before_downloading(self) -> None:
            with (
                mock.patch.object(module, "in_isolated_python_environment", return_value=True),
                mock.patch.object(module, "installed_ort_distributions", return_value=set()),
                mock.patch.object(platform, "system", return_value="Linux"),
                mock.patch.object(platform, "release", return_value="5.15.153.1-microsoft-standard-WSL2"),
                mock.patch.object(module, "run_command") as run_mock,
                self.assertRaises(SystemExit),
            ):
                pip_install_for_target("migraphx", None)
            run_mock.assert_not_called()

        def test_bootstrap_refuses_to_uninstall_an_existing_runtime(self) -> None:
            with (
                mock.patch.object(module, "in_isolated_python_environment", return_value=True),
                mock.patch.object(module, "installed_ort_distributions", return_value={"onnxruntime"}),
                mock.patch.object(platform, "system", return_value="Windows"),
                mock.patch.object(platform, "machine", return_value="AMD64"),
                mock.patch.object(sys, "version_info", (3, 12, 9)),
                mock.patch.object(module, "run_command") as run_mock,
                self.assertRaises(SystemExit),
            ):
                pip_install_for_target("dml", None)
            run_mock.assert_not_called()

        def test_nonfinite_output_is_rejected(self) -> None:
            try:
                np = importlib.import_module("numpy")
            except ImportError:
                self.skipTest("NumPy is not installed in this test environment")
            with self.assertRaises(SystemExit):
                validate_outputs([np.array([1.0, float("nan")], dtype=np.float32)], np)

        def test_object_output_is_rejected(self) -> None:
            try:
                np = importlib.import_module("numpy")
            except ImportError:
                self.skipTest("NumPy is not installed in this test environment")
            with self.assertRaises(SystemExit):
                validate_outputs([[{"unsafe": 1}]], np)

        def test_strict_offload_disables_cpu_ep_fallback(self) -> None:
            class FakeOptions:
                def __init__(self) -> None:
                    self.entries: dict[str, str] = {}

                def add_session_config_entry(self, key: str, value: str) -> None:
                    self.entries[key] = value

            options = FakeOptions()
            configure_strict_offload(options, "MIGraphXExecutionProvider", True)
            self.assertEqual(options.entries, {"session.disable_cpu_ep_fallback": "1"})
            cpu_options = FakeOptions()
            configure_strict_offload(cpu_options, "CPUExecutionProvider", True)
            self.assertEqual(cpu_options.entries, {})

    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(case)
        for case in (
            VersionAndSelectionTests,
            InputAndModelTests,
            HardwareDetectionTests,
            EvidenceParsingTests,
            SafetyTests,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    args = parse_args()
    if args.unit_tests:
        return run_unit_tests()
    if not args.windows_ml:
        return run_demo(args)
    if platform.system() != "Windows":
        fail("--windows-ml is available only on Windows.")
    if args.bootstrap:
        fail(
            "--windows-ml and --bootstrap cannot be combined. Install the pinned Windows ML "
            "packages and matching Windows App Runtime from section 10 first."
        )
    if args.target not in {"auto", "gpu", "migraphx"}:
        fail("--windows-ml currently verifies the Windows AMD MIGraphX GPU plugin only.")
    try:
        bootstrap = importlib.import_module(
            "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
        )
    except Exception as exc:
        fail(
            "Cannot import the Windows App SDK bootstrap package. Install the pinned Windows ML "
            f"packages from section 10: {exc}"
        )
    try:
        with bootstrap.initialize(options=bootstrap.InitializeOptions.ON_NO_MATCH_SHOW_UI):
            return run_demo(args)
    except Exception as exc:
        fail(
            "Windows App Runtime initialization failed. Install the runtime version matching the "
            f"pinned wasdk packages and do not use Microsoft Store Python: {exc}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
