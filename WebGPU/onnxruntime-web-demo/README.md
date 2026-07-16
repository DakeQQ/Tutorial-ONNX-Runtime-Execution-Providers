# ONNX Runtime local provider demo

This folder contains the supported one-click smoke test for browser WASM/WebGPU/WebNN and the native Python WebGPU plugin.

Read the complete guides before testing production models:

- [English guide](../README.md)
- [简体中文教程](../README.zh-CN.md)

## Quick start

Run these commands **from this folder**:

| Route | Windows | Linux/macOS |
|---|---|---|
| Browser WASM baseline | `run_demo.bat wasm` | `bash run_demo.sh wasm` |
| Browser WebGPU | `run_demo.bat webgpu` | `bash run_demo.sh webgpu` |
| Browser WebNN GPU request | `run_demo.bat webnn --device gpu` | `bash run_demo.sh webnn --device gpu` |
| Native Python WebGPU | `run_demo.bat native-webgpu` | `bash run_demo.sh native-webgpu` |

The browser launcher requires Python 3.10+ only for its local HTTP server. Node.js is optional: install current Node.js LTS and run `npm ci` to use local ORT Web assets; otherwise the page downloads the pinned 1.27.0 assets from jsDelivr.

The pinned native stack requires 64-bit CPython 3.11–3.14 and supports Windows x64, Linux x86-64 with glibc 2.27+, and macOS 14+ on Apple Silicon. Intel macOS is not supported by the pinned native route because `onnxruntime 1.27.0` has no macOS x86-64 core wheel.

## What a pass proves

The checked-in `execution_provider_demo.onnx` model is a static float32 `MatMul → Add → Relu` graph.

- Every browser route verifies the exact ORT Web version, model contract, and an independent JavaScript math reference.
- WebGPU/WebNN browser routes additionally compare against a separate WASM session.
- Strict browser mode disables implicit CPU EP fallback.
- Native WebGPU compares with CPU ORT, disables CPU fallback by default, and checks the ORT profile for WebGPU compute events and zero CPU node events.

`--allow-wasm-fallback` only permits unsupported **model nodes** to use WASM after the requested browser API/context initializes. It does not hide a missing WebGPU adapter or unavailable WebNN API.

For WebNN, the launcher enables the API in an isolated temporary Chromium profile. `--webnn-backend auto` chooses LiteRT on Windows versions before build 26100 and the Chromium platform default elsewhere. `--webnn-backend litert` explicitly enables `WebNNLiteRT` and disables higher-priority platform backends; it also disables the legacy `WebNNDirectML` feature for compatibility with Chromium 148 and older builds.

## Supported entry points

The maintained rookie path consists of:

- `run_demo.bat` / `run_demo.sh`
- `launch_demo.py`
- `browser-demo.html` / `browser-demo.js`
- `native_webgpu_validator.py`
- `execution_provider_demo.onnx`
- `package.json` / `package-lock.json`
- `requirements-native-webgpu.txt`

---

## 中文快速说明

请在本目录运行上表命令。浏览器路径只用 Python 3.10+ 启动本地服务器；执行 `npm ci` 可准备离线文件，否则页面从固定版本 jsDelivr 下载 ORT Web 1.27.0。

固定原生组合要求 64 位 CPython 3.11–3.14，支持 Windows x64、glibc 2.27+ 的 Linux x86-64，以及 Apple Silicon 上的 macOS 14+。由于 ORT 1.27.0 核心没有 macOS x86-64 wheel，Intel Mac 不支持该原生路径。

浏览器通过必须完成独立数学参考检查；WebGPU/WebNN 还会与 WASM 对比。原生 WebGPU 默认禁止 CPU 回退，并检查 ORT 性能文件中的 WebGPU 计算节点。完整细节和故障排查请阅读上方中英文教程。
