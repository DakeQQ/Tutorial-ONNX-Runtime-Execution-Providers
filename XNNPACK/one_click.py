#!/usr/bin/env python3
"""Build or reuse an XNNPACK-enabled ONNX Runtime wheel and prove execution.

The default path creates an isolated environment, checks out the pinned ONNX
Runtime release, builds a Python wheel with ``--use_xnnpack``, generates a
deterministic MatMul model, and verifies XNNPACK assignment without CPU fallback.

Examples:
    python XNNPACK/one_click.py
    python XNNPACK/one_click.py --wheel /path/to/custom-onnxruntime.whl
    python XNNPACK/one_click.py --threads 4 --jobs 6
    python XNNPACK/one_click.py --unit-tests
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn, Sequence
import unittest
import venv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENV = SCRIPT_DIR / ".venv-xnnpack"
DEFAULT_WORK_DIR = SCRIPT_DIR / ".xnnpack-build"
DEFAULT_ARTIFACTS = SCRIPT_DIR / ".xnnpack-smoke"
ORT_REPOSITORY = "https://github.com/microsoft/onnxruntime.git"
ORT_REF = "v1.27.1"
ORT_COMMIT = "df2ba1cf8108aa63627cf4cdf8f807880b938616"
ORT_VERSION = "1.27.1"
ONNX_VERSION = "1.22.0"
XNNPACK_EP = "XnnpackExecutionProvider"
CPU_EP = "CPUExecutionProvider"
ALLOW_INTRA_OP_SPINNING = "session.intra_op.allow_spinning"
DISABLE_CPU_FALLBACK = "session.disable_cpu_ep_fallback"
RECORD_ASSIGNMENT = "session.record_ep_graph_assignment_info"


def info(message: str) -> None:
    print(f"[INFO/信息] {message}", flush=True)


def fail(message: str, exit_code: int = 2) -> NoReturn:
    print(f"[FAIL/失败] {message}", file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


def quote_command(command: Sequence[os.PathLike[str] | str]) -> str:
    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    normalized = [str(part) for part in command]
    info(f"Running/执行: {quote_command(normalized)}")
    try:
        return subprocess.run(
            normalized,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        fail(f"Command not found: {normalized[0]} ({exc})")
    except subprocess.CalledProcessError as exc:
        if capture:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
        fail(
            f"Command failed with exit code {exc.returncode}: "
            f"{quote_command(normalized)}"
        )


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def validate_python() -> None:
    if platform.python_implementation() != "CPython":
        fail("The pinned ONNX Runtime build requires CPython. / 请使用 CPython。")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        fail("Use 64-bit CPython 3.11, 3.12, 3.13, or 3.14.")
    if sys.maxsize <= 2**32:
        fail("Use a 64-bit Python process. / 请使用 64 位 Python。")


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def command_path(name: str) -> str | None:
    return shutil.which(name)


def validate_source_build_host() -> str | None:
    system = platform.system()
    if system not in {"Linux", "Windows"}:
        fail(
            "The one-click Python source build supports Linux and Windows. "
            "Use the official Android Maven or iOS CocoaPods package on mobile."
        )

    missing = [name for name in ("git", "cmake") if command_path(name) is None]
    if system == "Linux":
        missing.extend(
            name for name in ("cc", "c++") if command_path(name) is None
        )
    elif command_path("cl") is None:
        missing.append("cl")
    if missing:
        fail(
            "Missing source-build tools: "
            + ", ".join(sorted(set(missing)))
            + ". Install the prerequisites listed in XNNPACK/README.md."
        )

    cmake_result = run_command(["cmake", "--version"], capture=True)
    cmake_version = parse_version(cmake_result.stdout)
    if cmake_version is None or cmake_version < (3, 28):
        fail(
            f"CMake >= 3.28 is required; detected output: {cmake_result.stdout.strip()}"
        )

    if system == "Linux" and command_path("ninja") is None:
        if command_path("make") is None:
            fail("Install Ninja or Make before building ONNX Runtime.")
        return None
    return "Ninja" if system == "Linux" else None


def physical_core_count() -> int:
    if platform.system() == "Linux":
        topology_root = Path("/sys/devices/system/cpu")
        cores: set[tuple[str, str]] = set()
        for cpu_dir in topology_root.glob("cpu[0-9]*"):
            try:
                package_id = (cpu_dir / "topology/physical_package_id").read_text(
                    encoding="ascii"
                ).strip()
                core_id = (cpu_dir / "topology/core_id").read_text(
                    encoding="ascii"
                ).strip()
            except OSError:
                continue
            cores.add((package_id, core_id))
        if cores:
            return len(cores)

    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def build_info_matches_commit(build_info: str) -> bool:
    match = re.search(r"\bgit-commit-id=([0-9a-fA-F]+)", build_info)
    if match is None or len(match.group(1)) < 7:
        return False
    return ORT_COMMIT.startswith(match.group(1).lower())


def environment_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    check = (
        "import onnx, onnxruntime as ort, re; "
        f"assert ort.__version__ == {ORT_VERSION!r}; "
        f"assert onnx.__version__ == {ONNX_VERSION!r}; "
        f"assert {XNNPACK_EP!r} in ort.get_available_providers(); "
        "match=re.search(r'git-commit-id=([0-9a-fA-F]+)', ort.get_build_info()); "
        f"assert match and len(match.group(1)) >= 7 and {ORT_COMMIT!r}.startswith(match.group(1).lower())"
    )
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


def ensure_venv(venv_dir: Path) -> Path:
    python = venv_python(venv_dir)
    if python.is_file():
        return python
    info(f"Creating isolated environment: {venv_dir}")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    except Exception as exc:
        fail(
            "Could not create the virtual environment. On Debian/Ubuntu, "
            f"install the matching python3-venv package. ({exc})"
        )
    if not python.is_file():
        fail(f"Virtual-environment Python was not created: {python}")
    return python


def checkout_source(source_dir: Path, refresh: bool) -> None:
    if source_dir.exists() and refresh:
        info(f"Removing generated source checkout: {source_dir}")
        shutil.rmtree(source_dir)

    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "git",
                "clone",
                "--branch",
                ORT_REF,
                "--depth",
                "1",
                ORT_REPOSITORY,
                source_dir,
            ]
        )

    version_file = source_dir / "VERSION_NUMBER"
    build_script = source_dir / "tools/ci_build/build.py"
    if not version_file.is_file() or not build_script.is_file():
        fail(f"Not an ONNX Runtime source checkout: {source_dir}")
    source_version = version_file.read_text(encoding="ascii").splitlines()[0]
    if source_version != ORT_VERSION:
        fail(
            f"Source checkout reports ONNX Runtime {source_version}; "
            f"this demo requires {ORT_VERSION}. Use --refresh."
        )
    actual_commit = run_command(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, capture=True
    ).stdout.strip()
    if actual_commit != ORT_COMMIT:
        fail(
            f"Source checkout is {actual_commit}; expected audited commit {ORT_COMMIT}. "
            "Remove the checkout or use --refresh."
        )


def install_build_requirements(python: Path, source_dir: Path) -> None:
    run_command([python, "-m", "pip", "install", "--upgrade", "pip"])
    requirement_files = [source_dir / "requirements.txt"]
    pybind_requirements = (
        source_dir / "tools/ci_build/requirements/pybind/requirements.txt"
    )
    if pybind_requirements.is_file():
        requirement_files.append(pybind_requirements)

    command: list[os.PathLike[str] | str] = [
        python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "setuptools",
        "wheel",
    ]
    for requirement_file in requirement_files:
        command.extend(["-r", requirement_file])
    run_command(command)


def source_build_command(
    python: Path,
    source_dir: Path,
    build_dir: Path,
    jobs: int,
    generator: str | None,
) -> list[os.PathLike[str] | str]:
    command: list[os.PathLike[str] | str] = [
        python,
        source_dir / "tools/ci_build/build.py",
        "--config",
        "Release",
        "--build_dir",
        build_dir,
        "--update",
        "--build",
        "--build_wheel",
        "--use_xnnpack",
        "--cmake_extra_defines",
        "onnxruntime_BUILD_UNIT_TESTS=OFF",
        "--skip_tests",
        "--parallel",
        str(jobs),
    ]
    if generator:
        command.extend(["--cmake_generator", generator])
    return command


def find_built_wheel(build_dir: Path) -> Path:
    wheels = sorted(
        build_dir.glob("**/dist/onnxruntime-*.whl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not wheels:
        fail(f"The build completed but no ONNX Runtime wheel was found under {build_dir}.")
    return wheels[0]


def install_runtime(python: Path, wheel: Path) -> None:
    if not wheel.is_file():
        fail(f"Custom ONNX Runtime wheel does not exist: {wheel}")
    info(f"Installing XNNPACK-enabled wheel: {wheel}")
    run_command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            wheel,
            f"onnx=={ONNX_VERSION}",
        ]
    )
    run_command([python, "-m", "pip", "check"])
    if not environment_ready(python):
        fail(
            "The installed wheel does not expose the pinned XNNPACK stack. "
            "A stock PyPI onnxruntime wheel is not sufficient."
        )


def build_runtime(
    python: Path,
    work_dir: Path,
    jobs: int,
    refresh: bool,
) -> Path:
    generator = validate_source_build_host()
    source_dir = work_dir / f"onnxruntime-{ORT_VERSION}"
    build_dir = work_dir / "build"
    checkout_source(source_dir, refresh)
    if refresh and build_dir.exists():
        info(f"Removing generated build tree: {build_dir}")
        shutil.rmtree(build_dir)
    install_build_requirements(python, source_dir)
    run_command(
        source_build_command(python, source_dir, build_dir, jobs, generator),
        cwd=source_dir,
    )
    return find_built_wheel(build_dir)


def generate_smoke_model(model_path: Path) -> tuple[Any, Any, Any]:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model_path.parent.mkdir(parents=True, exist_ok=True)
    input_data = np.linspace(-1.5, 1.5, 12, dtype=np.float32).reshape(3, 4)
    weights = np.asarray(
        [
            [0.25, -0.50, 0.75, 1.00, -0.25],
            [1.00, 0.50, -0.25, 0.125, 0.75],
            [-0.50, 0.25, 0.50, 0.75, -1.00],
            [0.50, 0.125, -0.25, 1.25, 0.375],
        ],
        dtype=np.float32,
    )

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weights"], ["output"], name="xnnpack_matmul")],
        "xnnpack_strict_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 5])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-xnnpack-tutorial",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return input_data, weights, model


def matmul_reference(input_data: Any, weights: Any) -> Any:
    import numpy as np

    return np.matmul(input_data, weights).astype(np.float32, copy=False)


def profile_provider_counts(profile_path: Path) -> Counter[str]:
    with profile_path.open("r", encoding="utf-8") as stream:
        events = json.load(stream)
    counts: Counter[str] = Counter()
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[str(provider)] += 1
    return counts


def assignment_provider_counts(session: Any) -> tuple[Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    details: list[str] = []
    getter = getattr(session, "get_provider_graph_assignment_info", None)
    if getter is None:
        return counts, details
    for assignment in getter():
        nodes = list(assignment.get_nodes())
        counts[str(assignment.ep_name)] += len(nodes)
        for node in nodes:
            details.append(f"{assignment.ep_name}: {node.name} ({node.op_type})")
    return counts, details


def xnnpack_count(counts: Counter[str]) -> int:
    return sum(
        count for provider, count in counts.items() if "xnnpack" in provider.lower()
    )


def cpu_count(counts: Counter[str]) -> int:
    return sum(
        count
        for provider, count in counts.items()
        if provider == CPU_EP or provider.lower().startswith("cpu")
    )


def run_worker(artifacts_dir: Path, threads: int, warmups: int, runs: int) -> int:
    import numpy as np
    import onnxruntime as ort

    if XNNPACK_EP not in ort.get_available_providers():
        fail(
            f"{XNNPACK_EP} is absent from {ort.get_available_providers()}. "
            "Build ONNX Runtime with --use_xnnpack; the stock PyPI wheel is not enough."
        )
    build_info = ort.get_build_info()
    if ort.__version__ != ORT_VERSION or not build_info_matches_commit(build_info):
        fail(
            f"Unaudited ONNX Runtime build: version={ort.__version__}, "
            f"build_info={build_info!r}. Expected {ORT_VERSION} from {ORT_COMMIT}."
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "xnnpack_matmul.onnx"
    input_data, weights, _ = generate_smoke_model(model_path)
    reference = matmul_reference(input_data, weights)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.add_session_config_entry(ALLOW_INTRA_OP_SPINNING, "0")
    options.add_session_config_entry(DISABLE_CPU_FALLBACK, "1")
    options.add_session_config_entry(RECORD_ASSIGNMENT, "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(artifacts_dir / "xnnpack_profile")

    providers = [(XNNPACK_EP, {"intra_op_num_threads": str(threads)})]
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=providers,
        enable_fallback=0,
    )
    assignment_counts, assignment_details = assignment_provider_counts(session)

    output = None
    for _ in range(warmups):
        output = session.run(None, {"input": input_data})[0]

    timings_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        output = session.run(None, {"input": input_data})[0]
        timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    profile_path = Path(session.end_profiling())
    profile_counts = profile_provider_counts(profile_path)
    if output is None:
        fail("Inference produced no output.")

    np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-6)
    if xnnpack_count(assignment_counts) == 0 and xnnpack_count(profile_counts) == 0:
        fail(
            "Inference completed, but neither graph assignment nor the current profile "
            f"proved XNNPACK execution. Assignment={dict(assignment_counts)}, "
            f"profile={dict(profile_counts)}"
        )
    if cpu_count(assignment_counts) or cpu_count(profile_counts):
        fail(
            "CPU graph execution was observed in a no-fallback session. "
            f"Assignment={dict(assignment_counts)}, profile={dict(profile_counts)}"
        )

    print("=" * 76)
    print("ONNX Runtime + XNNPACK EP strict proof")
    print("=" * 76)
    print(f"Platform            : {platform.platform()} / {platform.machine()}")
    print(f"Python              : {platform.python_version()}")
    print(f"ONNX Runtime        : {ort.__version__}")
    print(f"ORT build info      : {build_info}")
    print(f"Available providers : {ort.get_available_providers()}")
    print(f"Session providers   : {session.get_providers()}")
    print(f"XNNPACK threads     : {threads}")
    print(f"Assignment evidence : {dict(assignment_counts) or 'API unavailable'}")
    for detail in assignment_details:
        print(f"  - {detail}")
    print(f"Profile evidence    : {dict(profile_counts)}")
    print(f"Max abs error       : {float(np.max(np.abs(output - reference))):.8g}")
    print(f"Median latency      : {statistics.median(timings_ms):.4f} ms")
    print(f"Mean latency        : {statistics.fmean(timings_ms):.4f} ms")
    print(f"Profile             : {profile_path}")
    print("[PASS/通过] XNNPACK executed the model with correct output and no CPU fallback.")
    print("This smoke test qualifies the path; it is not a performance benchmark.")
    return 0


class LauncherTests(unittest.TestCase):
    def test_parse_version(self) -> None:
        self.assertEqual(parse_version("cmake version 3.31.6"), (3, 31, 6))
        self.assertIsNone(parse_version("unknown"))

    def test_profile_provider_counts(self) -> None:
        events = [
            {"cat": "Node", "args": {"provider": XNNPACK_EP}},
            {"cat": "Session", "args": {"provider": CPU_EP}},
            {"cat": "Node", "args": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            self.assertEqual(profile_provider_counts(path), Counter({XNNPACK_EP: 1}))

    def test_provider_counters(self) -> None:
        counts = Counter({XNNPACK_EP: 2, CPU_EP: 1})
        self.assertEqual(xnnpack_count(counts), 2)
        self.assertEqual(cpu_count(counts), 1)

    def test_source_build_command(self) -> None:
        command = source_build_command(
            Path("python"), Path("source"), Path("build"), 3, "Ninja"
        )
        rendered = [str(part) for part in command]
        self.assertIn("--use_xnnpack", rendered)
        self.assertIn("--build_wheel", rendered)
        self.assertIn("onnxruntime_BUILD_UNIT_TESTS=OFF", rendered)
        self.assertEqual(rendered[-2:], ["--cmake_generator", "Ninja"])

    def test_physical_core_count_is_positive(self) -> None:
        self.assertGreaterEqual(physical_core_count(), 1)

    def test_build_info_matches_abbreviated_commit(self) -> None:
        self.assertTrue(
            build_info_matches_commit(
                "ORT Build Info: git-branch=HEAD, git-commit-id=df2ba1c, build type=Release"
            )
        )
        self.assertFalse(
            build_info_matches_commit(
                "ORT Build Info: git-branch=main, git-commit-id=0000000, build type=Release"
            )
        )


def run_unit_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LauncherTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and strictly verify ONNX Runtime XNNPACK EP."
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Install an existing custom --use_xnnpack wheel instead of building source.",
    )
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--threads",
        type=int,
        default=physical_core_count(),
        help="XNNPACK intra-op threads (default: detected physical cores).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Parallel source-build jobs (default: min(logical CPUs, 8)).",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recreate the generated source/build tree and reinstall the runtime.",
    )
    parser.add_argument("--unit-tests", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("threads", "jobs", "warmups", "runs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.unit_tests:
        return run_unit_tests()
    if args.worker:
        return run_worker(args.artifacts_dir.resolve(), args.threads, args.warmups, args.runs)

    validate_python()
    venv_dir = args.venv.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    python = ensure_venv(venv_dir)

    if args.wheel is not None or args.refresh or not environment_ready(python):
        if args.wheel is not None:
            wheel = args.wheel.expanduser().resolve()
        else:
            wheel = build_runtime(python, work_dir, args.jobs, args.refresh)
        install_runtime(python, wheel)
    else:
        info(f"Reusing pinned XNNPACK environment: {venv_dir}")

    worker_command = [
        python,
        Path(__file__).resolve(),
        "--worker",
        "--artifacts-dir",
        artifacts_dir,
        "--threads",
        str(args.threads),
        "--warmups",
        str(args.warmups),
        "--runs",
        str(args.runs),
    ]
    run_command(worker_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())