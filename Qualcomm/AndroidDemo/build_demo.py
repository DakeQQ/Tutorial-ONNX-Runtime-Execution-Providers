#!/usr/bin/env python3
"""Build, optionally install, and launch the QNN Android demo.

This stdlib-only launcher supplies the role normally played by a binary Gradle
wrapper JAR: it downloads a checksum-pinned Gradle distribution into the user
cache, creates an isolated model-generation environment, and invokes the Android
build without modifying the machine-wide Python or Gradle installation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
import venv
import zipfile

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_VENV = PROJECT_DIR / ".venv-models"
MODEL_REQUIREMENTS = PROJECT_DIR / "requirements-models.txt"
PREPARE_MODELS = PROJECT_DIR / "prepare_models.py"
APK_PATH = PROJECT_DIR / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
CPU_BACKEND_TARGET = (
    PROJECT_DIR / "app" / "src" / "main" / "jniLibs" / "arm64-v8a" / "libQnnCpu.so"
)
APPLICATION_ID = "io.github.ortqnn.demo"
ACTIVITY = f"{APPLICATION_ID}/.MainActivity"
GRADLE_VERSION = "8.9"
GRADLE_SHA256 = "d725d707bfabd4dfdc958c624003b3c80accc03f7037b5122c4b1d0ef15cecab"
GRADLE_URL = f"https://downloads.gradle.org/distributions/gradle-{GRADLE_VERSION}-bin.zip"


def _validate_python_host() -> None:
    if platform.python_implementation() != "CPython":
        raise RuntimeError("Use 64-bit CPython to build the Android demo.")
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Use 64-bit CPython 3.11, 3.12, 3.13, or 3.14.")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Use a 64-bit Python installation.")


def _cache_root() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))


def _venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _model_environment_ready(python: Path) -> bool:
    if not python.is_file():
        return False
    command = (
        "import importlib.metadata as m; "
        "raise SystemExit(m.version('onnx') != '1.22.0' or "
        "m.version('onnxruntime') != '1.26.0' or "
        "m.version('sympy') != '1.14.0')"
    )
    try:
        return subprocess.run(
            [str(python), "-c", command],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _prepare_models(offline: bool) -> None:
    python = _venv_python(MODEL_VENV)
    if not _model_environment_ready(python):
        if offline:
            raise RuntimeError(
                "--offline requires an existing pinned .venv-models. Run once online first."
            )
        print("[1/4] Creating the isolated model-generation environment...")
        if not python.is_file():
            venv.EnvBuilder(with_pip=True).create(MODEL_VENV)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "-r",
                str(MODEL_REQUIREMENTS),
            ],
            check=True,
        )
        if not _model_environment_ready(python):
            raise RuntimeError("Model-generation environment did not match its pins.")
    else:
        print("[1/4] Reusing the isolated model-generation environment.")

    print("[2/4] Generating static FP32 and QDQ ONNX assets...")
    subprocess.run([str(python), str(PREPARE_MODELS)], check=True, cwd=PROJECT_DIR)


def _qairt_roots(explicit: Path | None) -> list[Path]:
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


def _find_android_cpu_backend(root: Path) -> Path | None:
    if root.is_file():
        return root if root.name == "libQnnCpu.so" else None
    for abi in ("aarch64-android", "aarch64-android-clang6.0"):
        candidate = root / "lib" / abi / "libQnnCpu.so"
        if candidate.is_file():
            return candidate.resolve()
    lib_root = root / "lib"
    if lib_root.is_dir():
        for candidate in sorted(lib_root.glob("**/libQnnCpu.so")):
            if "android" in str(candidate.parent).lower():
                return candidate.resolve()
    return None


def _prepare_qnn_cpu_backend(explicit: Path | None) -> None:
    roots = _qairt_roots(explicit)
    for root in roots:
        source = _find_android_cpu_backend(root)
        if source is None:
            continue
        CPU_BACKEND_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, CPU_BACKEND_TARGET)
        print(f"      QNN CPU backend: {source} -> {CPU_BACKEND_TARGET}")
        return

    if explicit is not None:
        raise RuntimeError(
            f"--qnn-sdk does not contain an Android arm64 libQnnCpu.so: {explicit}"
        )
    if CPU_BACKEND_TARGET.is_file():
        print(f"      QNN CPU backend: reusing {CPU_BACKEND_TARGET}")
    else:
        print(
            "      QNN CPU backend: not packaged (optional reference backend). "
            "Pass --qnn-sdk <QAIRT-2.48.40-root> to enable the CPU button."
        )


def _candidate_android_sdks(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value).expanduser())
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        candidates.append(local / "Android" / "Sdk")
    elif platform.system() == "Darwin":
        candidates.append(Path.home() / "Library" / "Android" / "sdk")
    else:
        candidates.extend((Path.home() / "Android" / "Sdk", Path.home() / "Android" / "sdk"))
    return candidates


def _find_android_sdk(explicit: Path | None) -> Path:
    for candidate in _candidate_android_sdks(explicit):
        candidate = candidate.resolve()
        if (candidate / "platform-tools").is_dir() or (candidate / "platforms").is_dir():
            platform_jar = candidate / "platforms" / "android-35" / "android.jar"
            if not platform_jar.is_file():
                raise RuntimeError(
                    f"Android SDK found at {candidate}, but platform API 35 is missing. "
                    "Install 'Android SDK Platform 35' in Android Studio > SDK Manager."
                )
            return candidate
    raise RuntimeError(
        "Android SDK not found. Install Android Studio with SDK Platform 35, then set "
        "ANDROID_SDK_ROOT/ANDROID_HOME or pass --android-sdk."
    )


def _candidate_java_homes(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]).expanduser())
    if os.name == "nt":
        program_files = Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        candidates.append(program_files / "Android" / "Android Studio" / "jbr")
    elif platform.system() == "Darwin":
        candidates.append(Path("/Applications/Android Studio.app/Contents/jbr/Contents/Home"))
    else:
        candidates.extend(
            (
                Path("/opt/android-studio/jbr"),
                Path.home() / "android-studio" / "jbr",
            )
        )
    java_on_path = shutil.which("java")
    if java_on_path:
        candidates.append(Path(java_on_path).resolve().parent.parent)
    return candidates


def _find_java_home(explicit: Path | None) -> Path:
    executable_name = "java.exe" if os.name == "nt" else "java"
    for candidate in _candidate_java_homes(explicit):
        candidate = candidate.resolve()
        java = candidate / "bin" / executable_name
        if not java.is_file():
            continue
        probe = subprocess.run(
            [str(java), "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
        text = probe.stderr or probe.stdout
        match = re.search(r'version "(\d+)', text)
        if match and 17 <= int(match.group(1)) <= 22:
            return candidate
    raise RuntimeError(
        "JDK 17-22 not found. Install Android Studio's bundled JBR/JDK 17, set "
        "JAVA_HOME, or pass --java-home."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"Unsafe path in Gradle archive: {member.filename}")
        bundle.extractall(destination)


def _download(url: str, destination: Path) -> None:
    expected_total = 0
    last_error: Exception | None = None
    for attempt in range(1, 7):
        received = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "ort-qnn-tutorial/1.0", "Accept-Encoding": "identity"}
        if received:
            headers["Range"] = f"bytes={received}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_range = response.headers.get("Content-Range", "")
                range_match = re.search(r"/(\d+)$", content_range)
                if range_match:
                    expected_total = int(range_match.group(1))
                elif status == 200:
                    expected_total = int(response.headers.get("Content-Length", "0"))

                # A server may ignore Range. Restart rather than appending a
                # second complete ZIP to the interrupted prefix.
                if received and status != 206:
                    received = 0
                    mode = "wb"
                else:
                    mode = "ab" if received else "wb"

                with destination.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
                        if expected_total:
                            percent = min(100, received * 100 // expected_total)
                            print(
                                f"\r      Gradle download: {percent:3d}%",
                                end="",
                                flush=True,
                            )
            if expected_total and received >= expected_total:
                print()
                return
            if not expected_total and received:
                print()
                return
            last_error = RuntimeError(
                f"transfer ended at {received} of {expected_total or 'unknown'} bytes"
            )
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        print(f"\n      Transfer interrupted; resume attempt {attempt}/6...")
    raise RuntimeError(f"Could not download Gradle completely: {last_error}")


def _ensure_gradle(explicit: Path | None, offline: bool) -> Path:
    executable_name = "gradle.bat" if os.name == "nt" else "gradle"
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        executable = candidate if candidate.is_file() else candidate / "bin" / executable_name
        if executable.is_file():
            return executable
        raise RuntimeError(f"Gradle executable not found under {candidate}")

    cache_dir = _cache_root() / "ort-qnn-tutorial" / "gradle"
    distribution_dir = cache_dir / f"gradle-{GRADLE_VERSION}"
    executable = distribution_dir / "bin" / executable_name
    if executable.is_file():
        return executable
    if offline:
        raise RuntimeError(f"Pinned Gradle {GRADLE_VERSION} is not cached at {cache_dir}.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Downloading checksum-pinned Gradle {GRADLE_VERSION}...")
    with tempfile.NamedTemporaryFile(
        prefix="gradle-",
        suffix=".zip",
        dir=cache_dir,
        delete=False,
    ) as temporary:
        archive = Path(temporary.name)
    try:
        actual_hash = ""
        for integrity_attempt in range(1, 3):
            _download(GRADLE_URL, archive)
            actual_hash = _sha256(archive)
            if actual_hash == GRADLE_SHA256:
                break
            archive.unlink(missing_ok=True)
            print(
                "      Downloaded bytes failed the official checksum; "
                f"retrying cleanly ({integrity_attempt}/2)."
            )
        if actual_hash != GRADLE_SHA256:
            raise RuntimeError(
                f"Gradle checksum mismatch: expected {GRADLE_SHA256}, got {actual_hash}"
            )
        _safe_extract(archive, cache_dir)
    finally:
        archive.unlink(missing_ok=True)
    if not executable.is_file():
        raise RuntimeError("Gradle archive extracted without the expected executable.")
    if os.name != "nt":
        executable.chmod(executable.stat().st_mode | 0o111)
    return executable


def _adb_path(android_sdk: Path) -> Path:
    name = "adb.exe" if os.name == "nt" else "adb"
    path = android_sdk / "platform-tools" / name
    if not path.is_file():
        raise RuntimeError("Android SDK Platform-Tools/adb is missing.")
    return path


def _choose_device(adb: Path, serial: str | None) -> list[str]:
    if serial:
        adb_command = [str(adb), "-s", serial]
        subprocess.run([*adb_command, "get-state"], check=True, capture_output=True)
        return adb_command
    result = subprocess.run(
        [str(adb), "devices"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one authorized Android device, found {devices}. Use --device SERIAL."
        )
    return [str(adb), "-s", devices[0]]


def _device_property(adb_command: list[str], name: str) -> str:
    result = subprocess.run(
        [*adb_command, "shell", "getprop", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_device(adb_command: list[str], backend: str | None) -> None:
    if _device_property(adb_command, "ro.kernel.qemu") == "1":
        raise RuntimeError("An Android emulator cannot prove QNN GPU/HTP execution.")

    abi = _device_property(adb_command, "ro.product.cpu.abi")
    api_text = _device_property(adb_command, "ro.build.version.sdk")
    if abi != "arm64-v8a":
        raise RuntimeError(
            f"The APK requires an arm64-v8a device; ADB reported {abi or 'unknown'}."
        )
    try:
        api_level = int(api_text)
    except ValueError as exc:
        raise RuntimeError(f"ADB returned an invalid Android API level: {api_text!r}") from exc
    if api_level < 27:
        raise RuntimeError(
            f"QNN HTP requires Android API 27 or newer; device reports API {api_level}."
        )

    soc = _device_property(adb_command, "ro.soc.model")
    hardware = _device_property(adb_command, "ro.hardware")
    manufacturer = _device_property(adb_command, "ro.soc.manufacturer")
    identity = " ".join((manufacturer, soc, hardware)).lower()
    print(
        "Device preflight: "
        f"ABI={abi}, API={api_level}, SoC={soc or hardware or 'unknown'}"
    )
    if backend in {"gpu", "htp"} and not any(
        marker in identity
        for marker in ("qualcomm", "qcom", "qti", "snapdragon")
    ):
        print(
            "WARNING: Android properties did not identify a Qualcomm SoC. "
            "The strict QNN session remains the final hardware gate."
        )


def _build_and_maybe_install(args: argparse.Namespace) -> int:
    _validate_python_host()
    _prepare_models(args.offline)
    _prepare_qnn_cpu_backend(args.qnn_sdk)
    android_sdk = _find_android_sdk(args.android_sdk)
    java_home = _find_java_home(args.java_home)
    gradle = _ensure_gradle(args.gradle, args.offline)

    environment = os.environ.copy()
    environment["ANDROID_HOME"] = str(android_sdk)
    environment["ANDROID_SDK_ROOT"] = str(android_sdk)
    environment["JAVA_HOME"] = str(java_home)
    environment["PATH"] = str(java_home / "bin") + os.pathsep + environment.get("PATH", "")

    print("[4/4] Building the arm64-v8a debug APK...")
    tasks = ["--no-daemon", "--stacktrace"]
    if args.clean:
        tasks.append("clean")
    tasks.append("assembleDebug")
    if args.offline:
        tasks.insert(0, "--offline")
    subprocess.run(
        [str(gradle), *tasks],
        check=True,
        cwd=PROJECT_DIR,
        env=environment,
    )
    if not APK_PATH.is_file():
        raise RuntimeError(f"Gradle succeeded but APK is missing: {APK_PATH}")
    print(f"APK: {APK_PATH} ({APK_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")

    if args.install:
        if args.backend == "cpu" and not CPU_BACKEND_TARGET.is_file():
            raise RuntimeError(
                "Cannot auto-run QNN CPU: rebuild with --qnn-sdk <QAIRT-2.48.x-root>."
            )
        adb = _adb_path(android_sdk)
        adb_command = _choose_device(adb, args.device)
        _validate_device(adb_command, args.backend)
        subprocess.run([*adb_command, "install", "-r", str(APK_PATH)], check=True)
        launch = [*adb_command, "shell", "am", "start", "-n", ACTIVITY]
        if args.backend:
            launch.extend(("--es", "backend", args.backend))
        subprocess.run(launch, check=True)
        print(f"Installed and launched {APPLICATION_ID}.")
        print(f"Logcat tag/filter: {adb_command[0]} logcat | grep -iE 'onnxruntime|qnn'")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/install the Qualcomm QNN Android demo.")
    parser.add_argument("--install", action="store_true", help="install and launch after building")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu", "htp"),
        help="with --install, automatically run this backend after launch",
    )
    parser.add_argument("--device", help="adb serial when more than one device is connected")
    parser.add_argument("--android-sdk", type=Path, help="Android SDK root")
    parser.add_argument(
        "--qnn-sdk",
        type=Path,
        help="QAIRT 2.48.40 root; packages optional libQnnCpu.so for QNN CPU testing",
    )
    parser.add_argument("--java-home", type=Path, help="JDK/JBR 17-22 root")
    parser.add_argument("--gradle", type=Path, help="existing Gradle 8.9 executable/root")
    parser.add_argument("--offline", action="store_true", help="forbid downloads and use caches only")
    parser.add_argument("--clean", action="store_true", help="run Gradle clean before assembleDebug")
    parser.add_argument("--verbose", action="store_true", help="show Python traceback on failure")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.backend and not args.install:
        raise SystemExit("--backend requires --install")
    try:
        return _build_and_maybe_install(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nFAIL: {type(error).__name__}: {error}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
