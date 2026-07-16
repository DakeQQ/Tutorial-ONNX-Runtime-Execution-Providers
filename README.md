# ONNX Runtime Execution Provider Tutorials

Practical, reproducible setup guides and strict smoke tests for running ONNX models on AMD, Intel, NVIDIA, Qualcomm, WebGPU, WebNN, and WebAssembly backends.

> **Documentation snapshot:** all maintained platform guides were last rechecked on **2026-07-16**. Exact package pins, hardware gates, tested environments, and validation limits are documented in each platform guide.

[English](#english) · [简体中文](#简体中文)

---

<a id="english"></a>
## English

### What this repository provides

This repository is a platform index plus runnable qualification tests. The maintained demos are designed to do more than confirm that an Execution Provider (EP) can be imported. Where the provider API permits, they:

1. use a pinned, internally compatible software stack;
2. create or include a deterministic local smoke model;
3. compare outputs with an independent CPU or mathematical reference;
4. disable or detect unintended CPU fallback; and
5. inspect current-run graph-assignment or profiling evidence.

> **Important:** `onnxruntime.get_available_providers()` only shows that a provider is exposed by the installed runtime. It does **not** prove that model nodes executed on the requested CPU, GPU, or NPU.
>
> Use a separate virtual environment for each route. Packages such as `onnxruntime`, `onnxruntime-gpu`, `onnxruntime-openvino`, and `onnxruntime-directml` provide the same Python module and must not be mixed casually.

### Choose a tutorial

| Area | Hardware and execution route | Covered hosts | Complete guides | Proof entry point after prerequisites |
|---|---|---|---|---|
| **AMD** | AMD GPU through DirectML or MIGraphX; Ryzen AI NPU through Vitis AI | Windows and Ubuntu; exact GPU/NPU support gates vary | [English](AMD/README.md) · [简体中文](AMD/README.zh-CN.md) | Use the [`amd_ort_hardware_test.py`](AMD/amd_ort_hardware_test.py) command matrix and select `dml`, `migraphx`, or `npu` |
| **Intel** | Intel CPU, integrated/discrete GPU, or integrated NPU through OpenVINO EP | Windows 11 and Ubuntu x86-64 | [English](Intel/README.md) · [简体中文](Intel/README.zh-CN.md) | Windows: `Intel\run_demo.bat --device CPU`<br>Linux: `bash Intel/run_demo.sh --device CPU` |
| **NVIDIA** | CUDA EP, classic TensorRT EP, or standalone TensorRT RTX plugin | Windows 10/11 and Ubuntu x86-64 | [English](NVIDIA/README.md) · [简体中文](NVIDIA/README.zh-CN.md) | Start with `python NVIDIA/provider_test.py --provider cuda` |
| **Qualcomm** | QNN GPU or HTP/NPU; optional QNN CPU reference backend | Native Windows ARM64 on Snapdragon and physical Android ARM64 devices | [English](Qualcomm/README.md) · [简体中文](Qualcomm/README.zh-CN.md) · [Android app](Qualcomm/AndroidDemo/README.md) | Windows: `python Qualcomm/one_click.py htp`<br>Android: `python Qualcomm/AndroidDemo/build_demo.py --install --backend htp` |
| **Web and native WebGPU** | Browser WASM, browser WebGPU, browser WebNN, or native Python WebGPU plugin | Browser-dependent Windows/Linux/macOS routes; native wheel support is narrower | [English](WebGPU/README.md) · [简体中文](WebGPU/README.zh-CN.md) · [Demo](WebGPU/onnxruntime-web-demo/README.md) | Windows: `WebGPU\onnxruntime-web-demo\run_demo.bat wasm`<br>Linux/macOS: `bash WebGPU/onnxruntime-web-demo/run_demo.sh wasm` |

### Route order matters

- **AMD:** GPU and NPU use different software stacks. Choose the exact hardware/OS path in the guide; new Linux GPU deployments use MIGraphX rather than the removed ROCm EP.
- **Intel:** qualify `CPU` first, then explicitly test `GPU`, `GPU.x`, or `NPU`. An `AUTO:...` run is useful for deployment but is not proof of one physical device.
- **NVIDIA:** pass CUDA first. Add classic TensorRT only after CUDA works. Keep the standalone TensorRT RTX plugin in a separate environment from `onnxruntime-gpu`.
- **Qualcomm:** local Windows GPU/HTP execution requires a native ARM64 process. HTP should start with a static QDQ model; Android testing requires a physical Snapdragon device.
- **Web:** run WASM first, then WebGPU, then experimental WebNN. Native Python WebGPU is a separate plugin route and does not make WebNN a native Python API.

### How to interpret a result

| Signal | What it proves |
|---|---|
| Provider appears in the available-provider list | The installed runtime exposes or can load that provider |
| Session creation succeeds | The provider accepted the session configuration and model initialization |
| Output matches an independent reference | The smoke-test result is numerically sane within its documented tolerance |
| Graph assignment or profile names the target EP | The current run executed graph work through the requested provider |
| Strict test reports no CPU nodes/events | No CPU graph fallback was observed through the evidence channel used by that test |
| Low latency alone | **Not proof** of accelerator execution or production performance |

The included models are qualification workloads, not benchmarks. After obtaining a strict pass, repeat the checks with the production model, real shapes, representative inputs, warm-up policy, precision mode, and application-level accuracy metrics.

### Repository map

| Path | Purpose |
|---|---|
| [AMD](AMD/README.md) | DirectML, Windows ML MIGraphX, ROCm/MIGraphX, and Ryzen AI/Vitis AI setup and verification |
| [Intel](Intel/README.md) | OpenVINO EP setup for Intel CPU, GPU, NPU, and meta-devices |
| [NVIDIA](NVIDIA/README.md) | CUDA, classic TensorRT, and standalone TensorRT RTX setup and strict profiling tests |
| [Qualcomm](Qualcomm/README.md) | QNN 2.x plugin setup for Snapdragon Windows and Android |
| [Qualcomm/AndroidDemo](Qualcomm/AndroidDemo/README.md) | Complete Kotlin CPU/GPU/HTP application and one-click build/install launcher |
| [WebGPU](WebGPU/README.md) | Browser WASM/WebGPU/WebNN and native Python WebGPU guidance |
| [WebGPU/onnxruntime-web-demo](WebGPU/onnxruntime-web-demo/README.md) | Runnable cross-provider browser/native smoke test |

### License

This repository is licensed under the [Apache License 2.0](LICENSE).

---

<a id="简体中文"></a>
## 简体中文

### 本仓库提供什么

本仓库既是各平台教程的总入口，也提供可直接运行的硬件资格验证脚本。维护中的演示不仅检查执行提供程序（Execution Provider，EP）能否导入；在 Provider API 允许时，还会：

1. 使用固定且彼此兼容的软件版本；
2. 本地生成或附带确定性的冒烟模型；
3. 与独立 CPU 或数学参考结果比较；
4. 禁止或检测意外的 CPU 回退；
5. 检查本次运行的计算图分配或性能分析证据。

> **重要：** `onnxruntime.get_available_providers()` 只表示当前运行时暴露了某个 Provider，**不能**证明模型节点已经在目标 CPU、GPU 或 NPU 上执行。
>
> 每条路线应使用独立虚拟环境。`onnxruntime`、`onnxruntime-gpu`、`onnxruntime-openvino`、`onnxruntime-directml` 等软件包会提供同名 Python 模块，不能随意混装。

### 选择教程

| 分类 | 硬件与执行路线 | 教程覆盖平台 | 完整指南 | 完成前置配置后的验证入口 |
|---|---|---|---|---|
| **AMD** | AMD GPU 使用 DirectML 或 MIGraphX；Ryzen AI NPU 使用 Vitis AI | Windows 与 Ubuntu；GPU/NPU 支持范围不同 | [English](AMD/README.md) · [简体中文](AMD/README.zh-CN.md) | 查看 [`amd_ort_hardware_test.py`](AMD/amd_ort_hardware_test.py) 命令表，并选择 `dml`、`migraphx` 或 `npu` |
| **Intel** | Intel CPU、集成/独立 GPU、集成 NPU 使用 OpenVINO EP | Windows 11 与 Ubuntu x86-64 | [English](Intel/README.md) · [简体中文](Intel/README.zh-CN.md) | Windows：`Intel\run_demo.bat --device CPU`<br>Linux：`bash Intel/run_demo.sh --device CPU` |
| **NVIDIA** | CUDA EP、传统 TensorRT EP、TensorRT RTX 独立插件 | Windows 10/11 与 Ubuntu x86-64 | [English](NVIDIA/README.md) · [简体中文](NVIDIA/README.zh-CN.md) | 先运行 `python NVIDIA/provider_test.py --provider cuda` |
| **Qualcomm** | QNN GPU 或 HTP/NPU；可选 QNN CPU 参考后端 | Snapdragon 原生 Windows ARM64 与 Android ARM64 真机 | [English](Qualcomm/README.md) · [简体中文](Qualcomm/README.zh-CN.md) · [Android 应用](Qualcomm/AndroidDemo/README.md) | Windows：`python Qualcomm/one_click.py htp`<br>Android：`python Qualcomm/AndroidDemo/build_demo.py --install --backend htp` |
| **Web 与原生 WebGPU** | 浏览器 WASM、WebGPU、WebNN，或原生 Python WebGPU 插件 | 依赖浏览器的 Windows/Linux/macOS；原生 wheel 范围更窄 | [English](WebGPU/README.md) · [简体中文](WebGPU/README.zh-CN.md) · [演示](WebGPU/onnxruntime-web-demo/README.md) | Windows：`WebGPU\onnxruntime-web-demo\run_demo.bat wasm`<br>Linux/macOS：`bash WebGPU/onnxruntime-web-demo/run_demo.sh wasm` |

### 路线顺序很重要

- **AMD：** GPU 与 NPU 属于不同软件栈。必须按指南选择与硬件和系统完全匹配的路径；新的 Linux GPU 项目使用 MIGraphX，不再使用已移除的 ROCm EP。
- **Intel：** 先验证 `CPU`，再显式测试 `GPU`、`GPU.x` 或 `NPU`。`AUTO:...` 适合部署测试，但不能证明某一块物理设备执行了模型。
- **NVIDIA：** 必须先通过 CUDA。CUDA 正常后再增加传统 TensorRT；TensorRT RTX 独立插件不能与 `onnxruntime-gpu` 混在同一环境。
- **Qualcomm：** Windows 本机 GPU/HTP 推理要求原生 ARM64 进程。HTP 应先使用静态 QDQ 模型；Android 验证必须使用 Snapdragon 真机。
- **Web：** 先运行 WASM，再测试 WebGPU，最后测试实验性的 WebNN。原生 Python WebGPU 是独立插件路线，并不会把 WebNN 变成原生 Python API。

### 如何理解验证结果

| 信号 | 能证明什么 |
|---|---|
| Provider 出现在可用列表 | 当前运行时暴露或能够加载该 Provider |
| 会话创建成功 | Provider 接受了会话配置并完成模型初始化 |
| 输出与独立参考一致 | 冒烟测试结果在文档容差内数值有效 |
| Graph Assignment 或 Profile 明确标记目标 EP | 本次运行确实通过目标 Provider 执行了计算图工作 |
| 严格测试没有 CPU 节点/事件 | 在该测试采用的证据渠道中，没有观察到 CPU Graph 回退 |
| 仅仅延迟较低 | **不能证明**硬件加速，也不能代表生产性能 |

仓库中的小模型用于资格验证，不是性能基准。严格验证通过后，还应使用生产模型、真实 Shape、代表性输入、预热策略、精度模式和应用级准确率指标重新验证。

### 仓库目录

| 路径 | 用途 |
|---|---|
| [AMD](AMD/README.zh-CN.md) | DirectML、Windows ML MIGraphX、ROCm/MIGraphX 与 Ryzen AI/Vitis AI 配置和验证 |
| [Intel](Intel/README.zh-CN.md) | Intel CPU、GPU、NPU 与 Meta-device 的 OpenVINO EP 配置 |
| [NVIDIA](NVIDIA/README.zh-CN.md) | CUDA、传统 TensorRT、TensorRT RTX 独立插件配置与严格性能分析验证 |
| [Qualcomm](Qualcomm/README.zh-CN.md) | Snapdragon Windows 与 Android 的 QNN 2.x 插件配置 |
| [Qualcomm/AndroidDemo](Qualcomm/AndroidDemo/README.md) | 完整 Kotlin CPU/GPU/HTP 应用与一键构建/安装脚本 |
| [WebGPU](WebGPU/README.zh-CN.md) | 浏览器 WASM/WebGPU/WebNN 与原生 Python WebGPU 指南 |
| [WebGPU/onnxruntime-web-demo](WebGPU/onnxruntime-web-demo/README.md) | 可运行的浏览器/原生跨 Provider 冒烟测试 |

### 许可证

本仓库采用 [Apache License 2.0](LICENSE)。
