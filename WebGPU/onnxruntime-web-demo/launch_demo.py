#!/usr/bin/env python3
"""Serve and launch the local ORT Web demo, or run the native WebGPU demo."""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NATIVE_VENV_DIR = SCRIPT_DIR / ".venv-webgpu"
NATIVE_REQUIREMENTS = SCRIPT_DIR / "requirements-native-webgpu.txt"


class DemoRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler with headers needed by multi-threaded ORT Web WASM."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")


def _browser_candidates(prefer_preview: bool = False) -> list[str]:
    system = platform.system()
    if system == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        program_files = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        program_files_x86 = Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        preview = [
            str(local / "Google/Chrome SxS/Application/chrome.exe"),
            str(local / "Microsoft/Edge SxS/Application/msedge.exe"),
        ]
        stable = [
            str(local / "Google/Chrome/Application/chrome.exe"),
            str(program_files / "Google/Chrome/Application/chrome.exe"),
            str(program_files / "Microsoft/Edge/Application/msedge.exe"),
            str(program_files_x86 / "Google/Chrome/Application/chrome.exe"),
            str(program_files_x86 / "Microsoft/Edge/Application/msedge.exe"),
        ]
        return [*preview, *stable] if prefer_preview else [*stable, *preview]
    if system == "Darwin":
        preview = [
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Microsoft Edge Canary.app/Contents/MacOS/Microsoft Edge Canary",
        ]
        stable = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
        return [*preview, *stable] if prefer_preview else [*stable, *preview]
    preview = [
        "google-chrome-unstable",
        "microsoft-edge-dev",
    ]
    stable = [
        "google-chrome",
        "google-chrome-stable",
        "microsoft-edge",
        "chromium",
        "chromium-browser",
    ]
    return [*preview, *stable] if prefer_preview else [*stable, *preview]


def _resolve_browser(explicit: str | None, prefer_preview: bool = False) -> str | None:
    candidates = [explicit] if explicit else _browser_candidates(prefer_preview)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _browser_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return (result.stdout or result.stderr).strip() or "version unavailable"
    except (OSError, subprocess.SubprocessError):
        return "version unavailable"


def _webnn_launch_flags(requested_backend: str) -> tuple[str, list[str], list[str]]:
    effective_backend = requested_backend
    if requested_backend == "auto":
        effective_backend = "platform"
        if platform.system() == "Windows":
            try:
                if sys.getwindowsversion().build < 26100:
                    effective_backend = "litert"
            except AttributeError:
                pass

    enabled_features = ["WebMachineLearningNeuralNetwork"]
    disabled_features: list[str] = []
    if effective_backend == "litert":
        # Chromium checks platform backends before LiteRT. Disable them and
        # explicitly enable LiteRT. WebNNDirectML was removed in Chromium 149,
        # but keeping that legacy feature name makes the policy work on older
        # builds too; unknown feature names are ignored by Chromium.
        enabled_features.append("WebNNLiteRT")
        disabled_features = [
            "WebNNOnnxRuntime",
            "WebNNDirectML",
            "WebNNCoreML",
        ]
    return effective_backend, enabled_features, disabled_features


def _print_webnn_launch_flags(requested_backend: str) -> tuple[list[str], list[str]]:
    effective_backend, enabled_features, disabled_features = _webnn_launch_flags(
        requested_backend
    )
    print(f"[Browser] WebNN backend policy: {effective_backend}")
    print(f"[Browser] Enabled features: {','.join(enabled_features)}")
    if disabled_features:
        print(f"[Browser] Disabled features: {','.join(disabled_features)}")
    return enabled_features, disabled_features


def _launch_webnn_browser(
    url: str,
    explicit: str | None,
    requested_backend: str,
    browser_args: list[str],
) -> tuple[
    subprocess.Popen[bytes] | None,
    tempfile.TemporaryDirectory[str] | None,
]:
    enabled_features, disabled_features = _print_webnn_launch_flags(
        requested_backend
    )
    executable = _resolve_browser(explicit, prefer_preview=True)
    if not executable:
        print("[Browser] Chrome/Edge was not found; open this URL in a WebNN-enabled Chromium browser:")
        print(f"          {url}")
        if browser_args:
            print(
                "[Browser] Apply these extra arguments manually: "
                + " ".join(browser_args)
            )
        return None, None

    profile_dir = tempfile.TemporaryDirectory(prefix="ort_webnn_browser_")
    command = [
        executable,
        "--new-window",
        f"--user-data-dir={profile_dir.name}",
        f"--enable-features={','.join(enabled_features)}",
    ]
    if disabled_features:
        command.append(f"--disable-features={','.join(disabled_features)}")
    command.extend(browser_args)
    command.append(url)
    print(f"[Browser] Launching WebNN with: {executable}")
    print(f"[Browser] Version: {_browser_version(executable)}")
    if browser_args:
        print(f"[Browser] Extra arguments: {' '.join(browser_args)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, profile_dir


def _launch_browser(
    provider: str,
    url: str,
    explicit: str | None,
    no_open: bool,
    webnn_backend: str,
    browser_args: list[str],
) -> tuple[subprocess.Popen[bytes] | None, tempfile.TemporaryDirectory[str] | None]:
    if no_open:
        print(f"[Browser] Automatic launch disabled. Open: {url}")
        if browser_args:
            print(
                "[Browser] Apply these extra arguments manually: "
                + " ".join(browser_args)
            )
        if provider == "webnn":
            _print_webnn_launch_flags(webnn_backend)
            print("[Browser] Apply the listed WebNN features to the browser you open manually.")
        return None, None
    if provider == "webnn":
        return _launch_webnn_browser(
            url, explicit, webnn_backend, browser_args
        )

    if provider == "webgpu":
        executable = _resolve_browser(explicit, prefer_preview=False)
        if executable:
            print(f"[Browser] Launching WebGPU with: {executable}")
            print(f"[Browser] Version: {_browser_version(executable)}")
            if browser_args:
                print(f"[Browser] Extra arguments: {' '.join(browser_args)}")
            return subprocess.Popen(
                [executable, "--new-window", *browser_args, url]
            ), None
        print("[Browser] Chrome/Edge was not found; open this URL in a WebGPU-enabled browser:")
        print(f"          {url}")
        if browser_args:
            print(
                "[Browser] Apply these extra arguments manually: "
                + " ".join(browser_args)
            )
        return None, None

    print(f"[Browser] Opening: {url}")
    if explicit:
        executable = _resolve_browser(explicit)
        if executable:
            return subprocess.Popen(
                [executable, "--new-window", *browser_args, url]
            ), None
        print(f"[Browser] Requested browser was not found: {explicit}")
    webbrowser.open_new(url)
    return None, None


def _native_venv_python() -> Path:
    if platform.system() == "Windows":
        return NATIVE_VENV_DIR / "Scripts" / "python.exe"
    return NATIVE_VENV_DIR / "bin" / "python"


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts)


def _validate_native_host() -> bool:
    system = platform.system()
    machine = platform.machine().lower()
    if (
        platform.python_implementation() != "CPython"
        or not (3, 11) <= sys.version_info[:2] < (3, 15)
        or sys.maxsize <= 2**32
    ):
        print(
            "ERROR: the pinned native WebGPU stack requires 64-bit CPython "
            "3.11, 3.12, 3.13, or 3.14.",
            file=sys.stderr,
        )
        return False
    if system in {"Windows", "Linux"} and machine not in {"amd64", "x86_64"}:
        print(
            f"ERROR: plugin 0.1.0 has no public {system} wheel for {machine}.",
            file=sys.stderr,
        )
        return False
    if system == "Darwin":
        if machine != "arm64":
            print(
                "ERROR: the plugin wheel is universal2, but the separately "
                "required onnxruntime 1.27.0 wheel is macOS arm64 only. "
                f"The pinned native demo cannot install on macOS {machine}.",
                file=sys.stderr,
            )
            return False
        mac_version = platform.mac_ver()[0]
        if mac_version and _version_tuple(mac_version) < (14,):
            print("ERROR: plugin 0.1.0 requires macOS 14 or newer.", file=sys.stderr)
            return False
    elif system == "Linux":
        libc_name, libc_version = platform.libc_ver()
        if libc_name and libc_name.lower() != "glibc":
            print(
                "ERROR: the public Linux wheel targets glibc, not "
                f"{libc_name or 'this libc'}.",
                file=sys.stderr,
            )
            return False
        if libc_version and _version_tuple(libc_version) < (2, 27):
            print("ERROR: the public Linux wheel requires glibc 2.27+.", file=sys.stderr)
            return False
    elif system != "Windows":
        print(f"ERROR: no public plugin 0.1.0 wheel for {system}.", file=sys.stderr)
        return False
    return True


def _native_python_supported(executable: Path) -> bool:
    try:
        result = subprocess.run(
            [
                str(executable),
                "-c",
                "import platform, sys; "
                "raise SystemExit(platform.python_implementation() != 'CPython' "
                "or not (3, 11) <= sys.version_info[:2] < (3, 15) "
                "or sys.maxsize <= 2**32)",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _native_environment_ready(executable: Path) -> bool:
    check = (
        "import sys; "
        "import platform; "
        "from importlib.metadata import version; "
        "import numpy, onnxruntime, onnxruntime_ep_webgpu; "
        "assert platform.python_implementation() == 'CPython'; "
        "assert (3, 11) <= sys.version_info[:2] < (3, 15); "
        "assert sys.maxsize > 2**32; "
        "assert int(numpy.__version__.split('.')[0]) < 3; "
        "assert version('onnxruntime') == '1.27.0'; "
        "assert version('onnxruntime-ep-webgpu') == '0.1.0'"
    )
    try:
        result = subprocess.run(
            [str(executable), "-c", check],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_native_environment() -> Path | None:
    if not _validate_native_host():
        return None
    if not NATIVE_REQUIREMENTS.is_file():
        print(f"ERROR: missing {NATIVE_REQUIREMENTS}", file=sys.stderr)
        return None

    executable = _native_venv_python()
    if executable.is_file() and not _native_python_supported(executable):
        print("[Native 1/3] Replacing an incompatible .venv-webgpu...")
        try:
            shutil.rmtree(NATIVE_VENV_DIR)
        except OSError as error:
            print(
                f"ERROR: could not remove incompatible {NATIVE_VENV_DIR}: {error}",
                file=sys.stderr,
            )
            return None
    if not executable.is_file():
        print("[Native 1/3] Creating isolated .venv-webgpu...")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(NATIVE_VENV_DIR)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            print(
                "ERROR: could not create .venv-webgpu. On Ubuntu, install "
                f"the matching python3-venv package. ({error})",
                file=sys.stderr,
            )
            return None
    else:
        print("[Native 1/3] Reusing isolated .venv-webgpu.")

    if not _native_environment_ready(executable):
        print("[Native 2/3] Installing pinned ONNX Runtime WebGPU packages...")
        try:
            subprocess.run(
                [str(executable), "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
            )
            subprocess.run(
                [
                    str(executable),
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(NATIVE_REQUIREMENTS),
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            print(
                "ERROR: native package installation failed. Check Internet "
                "access and the documented OS/architecture/Python wheel matrix. "
                f"({error})",
                file=sys.stderr,
            )
            return None
    else:
        print("[Native 2/3] Pinned packages are already installed.")

    return executable


def _run_native(extra_args: list[str]) -> int:
    executable = _ensure_native_environment()
    if executable is None:
        return 2
    print("[Native 3/3] Starting strict native WebGPU plugin validation...")
    command = [str(executable), str(SCRIPT_DIR / "native_webgpu_validator.py"), *extra_args]
    try:
        return subprocess.call(command)
    except OSError as error:
        print(f"ERROR: could not start the native demo: {error}", file=sys.stderr)
        return 2


def _serve(args: argparse.Namespace) -> int:
    required_assets = (
        "browser-demo.html",
        "browser-demo.js",
        "execution_provider_demo.onnx",
    )
    missing_assets = [name for name in required_assets if not (SCRIPT_DIR / name).is_file()]
    if missing_assets:
        print(f"ERROR: missing demo asset(s): {', '.join(missing_assets)}", file=sys.stderr)
        return 2

    handler = functools.partial(DemoRequestHandler, directory=str(SCRIPT_DIR))
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        if args.port == 0:
            raise
        print(f"[HTTP] Port {args.port} is unavailable ({error}); selecting a free port.")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True

    actual_port = int(server.server_address[1])
    effective_fallback = args.allow_wasm_fallback and args.provider != "wasm"
    local_ort_script = (
        SCRIPT_DIR / "node_modules" / "onnxruntime-web" / "dist" / "ort.all.min.js"
    )
    asset_policy = "local" if local_ort_script.is_file() else "cdn"
    query = urllib.parse.urlencode(
        {
            "ep": args.provider,
            "device": args.device,
            "iterations": args.iterations,
            "fallback": "1" if effective_fallback else "0",
            "profile": "0" if args.no_webgpu_profile else "1",
            "assets": asset_policy,
            "autorun": "1",
        }
    )
    url = f"http://127.0.0.1:{actual_port}/browser-demo.html?{query}"

    print("=" * 72)
    print("ONNX Runtime Web local-device demo server")
    print("=" * 72)
    print(f"Provider:        {args.provider}")
    if args.provider == "webnn":
        print(f"WebNN device:     {args.device}")
        print(f"WebNN backend:    {args.webnn_backend}")
    if args.provider == "webgpu":
        print(f"WebGPU profiling: {'disabled' if args.no_webgpu_profile else 'enabled (diagnostic overhead)'}")
    fallback_status = (
        "not applicable"
        if args.provider == "wasm"
        else "enabled"
        if effective_fallback
        else "strict EP-only"
    )
    print(f"WASM fallback:    {fallback_status}")
    print(
        "ORT Web assets:   "
        + ("local npm package" if asset_policy == "local" else "jsDelivr CDN (pinned 1.27.0)")
    )
    print(f"Document root:    {SCRIPT_DIR}")
    print(f"URL:              {url}")
    print("Security:         loopback secure context + COOP/COEP headers")
    print("Stop:             press Ctrl+C")

    browser_process: subprocess.Popen[bytes] | None = None
    profile_dir: tempfile.TemporaryDirectory[str] | None = None
    serving_started = False
    try:
        try:
            browser_process, profile_dir = _launch_browser(
                args.provider,
                url,
                args.browser,
                args.no_open,
                args.webnn_backend,
                args.browser_arg,
            )
        except OSError as error:
            print(
                f"[Browser] Automatic launch failed ({error}). Open manually: {url}",
                file=sys.stderr,
            )
        if args.seconds > 0:
            timer = threading.Timer(args.seconds, server.shutdown)
            timer.daemon = True
            timer.start()
        serving_started = True
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n[HTTP] Stopping...")
    finally:
        if serving_started:
            server.shutdown()
        server.server_close()
        if browser_process is not None and browser_process.poll() is None:
            browser_process.terminate()
            try:
                browser_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                browser_process.kill()
        if profile_dir is not None:
            try:
                profile_dir.cleanup()
            except OSError as error:
                print(
                    f"WARNING: temporary browser profile cleanup failed: {error}",
                    file=sys.stderr,
                )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click launcher: browser WASM/WebGPU/WebNN, or native Python "
            "WebGPU plugin inference."
        )
    )
    parser.add_argument(
        "provider",
        nargs="?",
        choices=("wasm", "webgpu", "webnn", "native-webgpu"),
        default="webgpu",
    )
    parser.add_argument("--device", choices=("cpu", "gpu", "npu"), default="gpu")
    parser.add_argument("--port", type=int, default=8000, help="0 selects a free port.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--browser", help="Explicit Chrome/Edge executable.")
    parser.add_argument(
        "--browser-arg",
        action="append",
        default=[],
        help=(
            "Extra browser argument; repeat as needed. Use the "
            "--browser-arg=--switch form for values beginning with '--'."
        ),
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--webnn-backend",
        choices=("auto", "platform", "litert"),
        default="auto",
        help=(
            "WebNN browser backend policy. 'auto' uses LiteRT on Windows "
            "before build 26100 and Chromium's platform default elsewhere."
        ),
    )
    parser.add_argument(
        "--allow-wasm-fallback",
        action="store_true",
        help="Append WASM after WebGPU/WebNN for unsupported operators.",
    )
    parser.add_argument(
        "--no-webgpu-profile",
        action="store_true",
        help="Disable WebGPU timestamp profiling for lower benchmark overhead.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["native-webgpu"]:
        return _run_native(raw_args[1:])
    args = _parser().parse_args(raw_args)
    if not 1 <= args.iterations <= 1000 or not 0 <= args.port <= 65535 or args.seconds < 0:
        print(
            "ERROR: iterations must be 1..1000, port must be 0..65535, "
            "and duration cannot be negative.",
            file=sys.stderr,
        )
        return 2
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
