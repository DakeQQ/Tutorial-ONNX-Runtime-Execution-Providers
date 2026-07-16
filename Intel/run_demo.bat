@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python Launcher was not found. Install 64-bit CPython 3.11, 3.12, or 3.13 first.
    exit /b 1
)

set "PY=.venv\Scripts\python.exe"
set "NEW_VENV="
if exist "%PY%" (
    call :is_supported_python "%PY%"
    if errorlevel 1 (
        echo [1/4] Existing .venv uses an unsupported Python; recreating it...
        rmdir /s /q .venv
    )
)

if not exist "%PY%" (
    echo [1/4] Creating .venv with a supported 64-bit Python...
    call :create_venv
    if errorlevel 1 exit /b 1
    set "NEW_VENV=1"
) else (
    echo [1/4] Reusing the supported .venv interpreter.
)

call :stack_matches
if errorlevel 1 (
    echo [2/4] Preparing a clean, matched environment...
    if not defined NEW_VENV (
        rmdir /s /q .venv
        call :create_venv
        if errorlevel 1 exit /b 1
    )
    echo [3/4] Installing the pinned ONNX Runtime/OpenVINO stack...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
) else (
    echo [2/4] Existing environment matches the pinned stack.
    echo [3/4] No package changes needed.
)

"%PY%" -m pip check >nul
if errorlevel 1 (
    echo ERROR: Python package dependencies are inconsistent. Delete .venv and rerun.
    exit /b 1
)

echo [4/4] Running the strict demo...
"%PY%" Test.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo Demo failed. Read English\README.md or Chinese\README.md, then check the driver and device list.
)
exit /b %RC%

:is_supported_python
"%~1" -c "import platform,struct,sys; ok=sys.implementation.name=='cpython' and sys.version_info[:2] in ((3,11),(3,12),(3,13)) and struct.calcsize('P')==8 and platform.machine().lower() in {'amd64','x86_64'}; raise SystemExit(0 if ok else 1)" >nul 2>nul
exit /b %ERRORLEVEL%

:create_venv
for %%V in (3.12 3.13 3.11) do (
    py -%%V -c "import platform,struct,sys; ok=sys.implementation.name=='cpython' and struct.calcsize('P')==8 and platform.machine().lower() in {'amd64','x86_64'}; raise SystemExit(0 if ok else 1)" >nul 2>nul
    if not errorlevel 1 (
        py -%%V -m venv .venv
        if not errorlevel 1 exit /b 0
    )
)
echo ERROR: install 64-bit CPython 3.11, 3.12, or 3.13. Python 3.10 and 3.14 have no compatible wheel.
exit /b 1

:stack_matches
"%PY%" -c "import importlib.metadata as m,re; norm=lambda name:re.sub(r'[-_.]+','-',name).lower(); wanted={'onnxruntime-openvino':'1.24.1','openvino':'2025.4.1','onnx':'1.22.0','numpy':'2.3.5'}; names={norm(d.metadata['Name']) for d in m.distributions() if d.metadata['Name']}; owners={norm(name) for name in m.packages_distributions().get('onnxruntime',[])}; forbidden={'onnxruntime','onnxruntime-gpu','onnxruntime-directml'}; ok=owners=={'onnxruntime-openvino'} and names.isdisjoint(forbidden) and all(m.version(name)==version for name,version in wanted.items()); raise SystemExit(0 if ok else 1)" >nul 2>nul
exit /b %ERRORLEVEL%