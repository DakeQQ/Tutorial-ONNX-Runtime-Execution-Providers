@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MODE=%~1"
if not defined MODE set "MODE=webgpu"

if /I "%MODE%"=="native-webgpu" goto NATIVE
if /I "%MODE%"=="webgpu" goto BROWSER
if /I "%MODE%"=="webnn" goto BROWSER
if /I "%MODE%"=="wasm" goto BROWSER

echo Usage: run_demo.bat [native-webgpu^|webgpu^|webnn^|wasm] [options]
exit /b 2

:NATIVE
call :FIND_NATIVE_PYTHON
if not defined PYTHON_COMMAND (
    echo ERROR: install 64-bit CPython 3.11, 3.12, 3.13, or 3.14 for native WebGPU.
    exit /b 1
)
%PYTHON_COMMAND% %PYTHON_SELECTOR% launch_demo.py %*
exit /b %ERRORLEVEL%

:BROWSER
call :FIND_BROWSER_PYTHON
if not defined PYTHON_COMMAND (
    echo ERROR: the browser launcher requires Python 3.10 or newer for its local server.
    exit /b 1
)
%PYTHON_COMMAND% %PYTHON_SELECTOR% launch_demo.py %*
exit /b %ERRORLEVEL%

:FIND_NATIVE_PYTHON
set "PYTHON_COMMAND="
set "PYTHON_SELECTOR="
where py >nul 2>nul
if errorlevel 1 goto TRY_NATIVE_PATH
call :TRY_NATIVE_PY 3.12
call :TRY_NATIVE_PY 3.13
call :TRY_NATIVE_PY 3.11
call :TRY_NATIVE_PY 3.14
if defined PYTHON_COMMAND exit /b 0
:TRY_NATIVE_PATH
where python >nul 2>nul
if errorlevel 1 exit /b 0
python -c "import platform, sys; raise SystemExit(platform.python_implementation() != 'CPython' or sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13), (3, 14)) or sys.maxsize.bit_length() != 63)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_COMMAND=python"
exit /b 0

:TRY_NATIVE_PY
if defined PYTHON_COMMAND exit /b 0
py -%~1 -c "import platform, sys; raise SystemExit(platform.python_implementation() != 'CPython' or sys.maxsize.bit_length() != 63)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_COMMAND=py"
set "PYTHON_SELECTOR=-%~1"
exit /b 0

:FIND_BROWSER_PYTHON
set "PYTHON_COMMAND="
set "PYTHON_SELECTOR="
where py >nul 2>nul
if errorlevel 1 goto TRY_BROWSER_PATH
py -3 -c "import sys; raise SystemExit(sys.version_info ^< (3, 10))" >nul 2>nul
if errorlevel 1 goto TRY_BROWSER_PATH
set "PYTHON_COMMAND=py"
set "PYTHON_SELECTOR=-3"
exit /b 0
:TRY_BROWSER_PATH
where python >nul 2>nul
if errorlevel 1 exit /b 0
python -c "import sys; raise SystemExit(sys.version_info ^< (3, 10))" >nul 2>nul
if errorlevel 1 exit /b 0
set "PYTHON_COMMAND=python"
exit /b 0
