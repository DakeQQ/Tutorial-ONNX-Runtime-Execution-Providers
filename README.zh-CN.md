# ONNX Runtime 执行提供程序教程

本仓库提供可复现的配置指南和**严格**的冒烟测试，帮助你确认 ONNX 模型确实运行在 **Apple、AMD、Intel、NVIDIA、Qualcomm、Web、Windows 或 XNNPACK CPU** 后端上，而不只是某个执行提供程序（EP）能够加载。

[English](README.md)  ·  **最近验证：2026-07-17。** 每份平台指南都单独列出锁定的软件版本、硬件要求、已测试环境和验证范围。

---

## 目录

- [ONNX Runtime 执行提供程序教程](#onnx-runtime-执行提供程序教程)
  - [目录](#目录)
  - [一张图看懂本仓库](#一张图看懂本仓库)
  - [四步完成验证](#四步完成验证)
  - [必须记住的一件事](#必须记住的一件事)
  - [Plugin EP 是加载模型，不是硬件](#plugin-ep-是加载模型不是硬件)
  - [选择适合的方案](#选择适合的方案)
  - [获得可信结果的步骤](#获得可信结果的步骤)
  - [如何解读结果](#如何解读结果)
  - [目录说明](#目录说明)
  - [许可证](#许可证)

---

## 一张图看懂本仓库

六类平台家族，加上跨厂商 Windows 和 XNNPACK CPU 路线，采用同一套验证流程，并遵循同一个原则：**用证据代替假设**。

```mermaid
flowchart LR
    ROOT(["ONNX Runtime<br/>EP 教程"])
    ROOT --> APPLE["Apple"]
    ROOT --> WIN["Windows"]
    ROOT --> AMD["AMD"]
    ROOT --> INTEL["Intel"]
    ROOT --> NV["NVIDIA"]
    ROOT --> QC["Qualcomm"]
    ROOT --> WEB["Web"]
    ROOT --> XNN["XNNPACK CPU"]

    APPLE --> APPa["CoreML CPU"] & APPb["GPU"] & APPc["神经网络引擎"]
    WIN --> WINa["DirectML GPU"] & WINb["Windows ML EP 目录"] & WINc["自动 EP 策略"]
    AMD --> AMDa["DirectML"] & AMDb["MIGraphX"] & AMDc["Ryzen AI / Vitis AI"]
    INTEL --> INTa["OpenVINO CPU"] & INTb["GPU / GPU.x"] & INTc["NPU"] & INTd["Meta-device"]
    NV --> NVa["CUDA EP"] & NVb["传统 TensorRT"] & NVc["TensorRT RTX"]
    QC --> QCa["QNN HTP / NPU"] & QCb["QNN GPU"] & QCc["Android 应用"]
    WEB --> WEBa["WASM"] & WEBb["WebGPU"] & WEBc["WebNN"] & WEBd["原生 Python WebGPU"]
    XNN --> XNNa["Android / iOS"] & XNNb["Windows / Linux"] & XNNc["Arm · x86 · WASM"]

    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b;
    class APPa,APPb,APPc,WINa,WINb,WINc,AMDa,AMDb,AMDc,INTa,INTb,INTc,INTd,NVa,NVb,NVc,QCa,QCb,QCc,WEBa,WEBb,WEBc,WEBd,XNNa,XNNb,XNNc leaf;

    style ROOT fill:#455a64,stroke:#cfd8dc,stroke-width:1px,color:#ffffff
    style APPLE fill:#212121,stroke:#bdbdbd,color:#ffffff
    style WIN fill:#0067b8,stroke:#8dc8f4,color:#ffffff
    style AMD fill:#c62828,stroke:#ff8a80,color:#ffffff
    style INTEL fill:#1565c0,stroke:#82b1ff,color:#ffffff
    style NV fill:#2e7d32,stroke:#b9f6ca,color:#ffffff
    style QC fill:#5e35b1,stroke:#b388ff,color:#ffffff
    style WEB fill:#e65100,stroke:#ffab40,color:#ffffff
    style XNN fill:#00695c,stroke:#64ffda,color:#ffffff
```

---

## 四步完成验证

掌握下面四步后，就可以在不同平台上重复使用同一套方法。

```mermaid
flowchart LR
    P["1 · 选择<br/>目标硬件"] --> S["2 · 配置<br/>独立环境 + 锁定版本"]
    S --> R["3 · 验证<br/>ORT → EP → CPU / GPU / NPU"]
    R --> T["4 · 确认<br/>输出正确 + 节点分配正确 + 无回退"]
```

| 步骤 | 你要做 | 你会得到 |
|---|---|---|
| **1 · 选择** | 根据具体设备、系统和驱动找到对应指南 | 适合当前硬件的方案及其最低要求 |
| **2 · 配置** | 按指南锁定的版本创建干净的虚拟环境 | 环境中只有一种 ONNX Runtime 发行包，避免相互冲突 |
| **3 · 验证** | 运行该方案中禁用回退的冒烟测试 | 确认推理确实经过目标 EP |
| **4 · 确认** | 查看测试输出的各项验证信息 | 确认目标 EP 处理了计算图；EP 支持时再确认设备层证据 |

---

## 必须记住的一件事

> [!IMPORTANT]
> 某个 EP 出现在 `get_available_providers()` 中，只能说明 ONNX Runtime **能够加载**它，并不代表模型已经**在 GPU 或 NPU 上执行**。

很多看似“能运行”的示例会在没有明显提示的情况下回退到 CPU。为避免这种误判，本仓库的测试会同时执行以下四项相互独立的检查：

```mermaid
flowchart TD
    Q["get_available_providers()<br/>列出了我的 EP"]
        Q -->|只能说明| A["Runtime 能加载它"]
        Q -.->|不能说明| B["节点已经在目标设备上执行"]
        subgraph Need["确认硬件执行：四项缺一不可"]
      direction LR
      M["确定性<br/>冒烟模型"] --> R["独立<br/>参考结果"]
            R --> G["节点分配 /<br/>性能分析记录"]
      G --> F["无 CPU<br/>回退"]
    end
    B --> M
```

| 检查 | 可以确认 | 可以排除 |
|---|---|---|
| 确定性冒烟模型 | 每次都使用相同输入 | 结果不稳定或无法复现 |
| 独立参考结果 | 输出数值正确 | 运行成功但计算结果有误 |
| 节点分配 / 性能分析记录 | 节点由目标 EP 执行 | 计算实际落在 CPU 却未被发现 |
| 无 CPU 回退 | 计算由加速器完成 | 回退到 CPU 后仍被误判为成功 |

> [!NOTE]
> 某些 EP 内部还有自己的调度器。例如，已分配给 `CoreMLExecutionProvider` 的分区仍可能在 CPU、GPU 或 ANE 上执行。ORT 的节点分配只能证明 EP 边界；当物理计算单元很重要时，还要使用该 Provider 的设备级分析工具。

> [!TIP]
> 每种方案都应使用**独立的虚拟环境**。`onnxruntime`、`onnxruntime-gpu`、`onnxruntime-openvino` 和 `onnxruntime-directml` 导入后都叫 `onnxruntime`，不能安装在同一个环境中。

---

## Plugin EP 是加载模型，不是硬件

> [!IMPORTANT]
> **Plugin EP 不是另一种硬件 Provider。** 它是 ONNX Runtime 面向执行提供程序的公共 C ABI、动态加载边界和设备发现模型。

厂商既可以通过它实现全新的树外 EP，也可以让现有 EP 接入该接口并独立发布。ORT 源码同时支持这两种方式，还会把内置 EP 和传统 Provider Bridge EP 适配到同一套工厂与设备模型中。

```mermaid
flowchart LR
    EP["EP 身份<br/>CUDA · QNN · WebGPU · 厂商 EP"] --> LOAD{"如何加载？"}
    LOAD --> IN["ORT 内置"]
    LOAD --> BR["传统 Provider Bridge<br/>+ Plugin EP 工厂"]
    LOAD --> PL["纯 Plugin EP<br/>仅使用公共 C ABI"]
    IN --> RUN["相同的 ORT 计算图分区与执行契约"]
    BR --> RUN
    PL --> RUN
```

本仓库已经覆盖四条插件路线：Windows ML MIGraphX、QNN 2.x、独立 TensorRT RTX 和原生 WebGPU。仅完成注册或设备发现并不代表执行通过；每个平台测试仍要求同时提供输出、节点分配/性能分析和无回退证据。

请阅读 [Plugin EP 源码深度解析](PluginEP/README.zh-CN.md)，了解加载器调用链、`OrtEpFactory` / `OrtEp` 生命周期、编译与内核执行路径、ABI 演进、打包规则和固定版本源码链接。

---

## 选择适合的方案

先确认手头的硬件，再打开对应指南并运行其中的第一条命令。

```mermaid
flowchart TD
    A{"准备在哪类硬件上运行？"}
    A -->|M 系列 Mac、iPhone 或 iPad| APPLE["Apple<br/>CoreML EP"]
    A -->|跨厂商 Windows 部署| WIN["Windows<br/>DirectML · EP 目录 · 自动选择"]
    A -->|AMD GPU 或 Ryzen AI NPU| AMD["AMD<br/>DirectML · MIGraphX · Vitis AI"]
    A -->|Intel CPU / GPU / NPU| INT["Intel<br/>OpenVINO EP"]
    A -->|NVIDIA GPU| NV["NVIDIA<br/>CUDA · TensorRT"]
    A -->|Snapdragon 或 Android| QC["Qualcomm<br/>QNN HTP · GPU"]
    A -->|浏览器或原生 WebGPU| WEB["Web<br/>WASM · WebGPU · WebNN"]
    A -->|移动端 / 桌面端 CPU 推理| XNN["XNNPACK<br/>Arm · x86 · WASM CPU"]
    APPLE --> G["打开指南 →<br/>运行第一条命令 →<br/>查看验证结果"]
    WIN --> G
    AMD --> G["打开指南 →<br/>运行第一条命令 →<br/>查看验证结果"]
    INT --> G
    NV --> G
    QC --> G
    WEB --> G
    XNN --> G
```

| 平台 | 适用硬件 | 支持的系统 | 起步命令 | 指南 |
|---|---|---|---|---|
| **Apple** | Apple Silicon Mac，或通过 CoreML 使用 iPhone/iPad | macOS · iOS | `python3 Apple/one_click.py`<br/><sub>当前 Python 路线：macOS 14+ arm64</sub> | [中文](Apple/README.zh-CN.md) · [EN](Apple/README.md) |
| **Windows** | 任意受支持的 DirectX 12 GPU，或 Windows ML 目录中的 CPU/GPU/NPU | Windows 10/11；目录 EP：Windows 11 24H2+ | `py -3.12 DirectML\one_click.py directml`<br/><sub>Windows ML：把 `directml` 换成 `windowsml --allow-download`</sub> | [中文](DirectML/README.zh-CN.md) · [EN](DirectML/README.md) |
| **AMD** | AMD GPU（DirectML / MIGraphX）或 Ryzen AI NPU（Vitis AI） | Windows · Ubuntu | `python AMD/provider_test.py --target dml`<br/><sub>按主机把 `dml` 换成 `migraphx` 或 `npu`</sub> | [中文](AMD/README.zh-CN.md) · [EN](AMD/README.md) |
| **Intel** | Intel CPU、集成/独立 GPU 或 NPU（OpenVINO） | Windows 11 · Ubuntu x86-64 | `bash Intel/run_demo.sh --device CPU`<br/><sub>Windows：`Intel\run_demo.bat --device CPU`</sub> | [中文](Intel/README.zh-CN.md) · [EN](Intel/README.md) |
| **NVIDIA** | NVIDIA GPU（CUDA / 传统 TensorRT / TensorRT RTX） | Windows 10/11 · Ubuntu x86-64 | `python NVIDIA/provider_test.py --provider cuda` | [中文](NVIDIA/README.zh-CN.md) · [EN](NVIDIA/README.md) |
| **Qualcomm** | Snapdragon HTP/NPU 或 GPU（QNN） | Windows ARM64 · Android ARM64 | `python Qualcomm/one_click.py htp`<br/><sub>Android：`python Qualcomm/AndroidDemo/build_demo.py --install --backend htp`</sub> | [中文](Qualcomm/README.zh-CN.md) · [EN](Qualcomm/README.md) · [应用](Qualcomm/AndroidDemo/README.zh-CN.md) |
| **Web** | 浏览器 WASM/WebGPU/WebNN 或原生 Python WebGPU | 取决于浏览器；原生 wheel 更窄 | `bash WebGPU/onnxruntime-web-demo/run_demo.sh wasm`<br/><sub>Windows：`WebGPU\onnxruntime-web-demo\run_demo.bat wasm`</sub> | [中文](WebGPU/README.zh-CN.md) · [EN](WebGPU/README.md) · [演示](WebGPU/onnxruntime-web-demo/README.zh-CN.md) |
| **XNNPACK** | 跨平台 Arm/x86 CPU 微内核 | Android · iOS · Windows · Linux · WASM | `python XNNPACK/one_click.py`<br/><sub>桌面 Python 会构建锁定版本的自定义 ORT wheel</sub> | [中文](XNNPACK/README.zh-CN.md) · [EN](XNNPACK/README.md) |

---

## 获得可信结果的步骤

先从最基础的方案开始，再逐步提高验证要求。如果某一步失败，请先解决当前问题，再继续下一步。

```mermaid
flowchart TD
    S1["1 · 检查要求<br/>硬件 · 系统 · 驱动"] --> S2["2 · 搭建环境<br/>单一 ORT 发行包、锁定版本"]
    S2 --> S3["3 · 运行基础方案<br/>先验证最简单的路径"]
    S3 --> S4["4 · 运行禁用回退的测试"]
    S4 --> D{"输出 + 节点分配<br/>+ 回退策略全部通过？"}
    D -->|是| S5["5 · 换生产模型<br/>重复验证"]
    D -->|否| FIX["修复驱动、依赖版本或模型，<br/>再回到第 3 步"]
    FIX --> S3
```

| 步骤 | 操作 | 通过条件 |
|---:|---|---|
| 1 | 对照平台指南检查硬件、系统和驱动要求 | 目标设备在官方支持范围内 |
| 2 | 为所选方案创建独立环境并安装锁定版本 | 依赖检查通过，且没有互相冲突的 ORT 包 |
| 3 | 先运行最基础的方案 | Apple：CoreML `CPUOnly` · Windows：DirectML 适配器 0 · AMD：匹配的 GPU/NPU 环境 · Intel：`CPU` · NVIDIA：CUDA · Web：WASM · XNNPACK：源码构建的 `MatMul` 证明 |
| 4 | 运行仓库中禁用回退的验证程序 | 输出、节点分配和回退检查全部通过 |
| 5 | 使用生产模型和真实输入再次验证 | 算子、形状、精度和应用指标均符合要求 |

**各平台推荐顺序：**

| 平台 | 推荐顺序 |
|---|---|
| Apple | CoreML `CPUOnly` → `ALL` → `CPUAndGPU` / `CPUAndNeuralEngine` → MLComputePlan 或 Instruments |
| Windows | DirectML 适配器 0 → 每个目标 DXGI 适配器 → 已安装的 Windows ML EP → 目录下载和生产 Provider |
| AMD | 先区分 GPU 与 NPU，再按硬件和系统选择 DirectML、MIGraphX 或 Vitis AI |
| Intel | `CPU` → 明确的 `GPU` / `GPU.x` / `NPU` → 部署所需 Meta-device |
| NVIDIA | CUDA → 传统 TensorRT；TensorRT RTX 插件需要使用独立环境 |
| Qualcomm | Windows 使用原生 ARM64；HTP 先验证静态 QDQ 模型；Android 使用 Snapdragon 真机 |
| Web | WASM → WebGPU → WebNN；原生 Python WebGPU 是独立插件路线 |
| XNNPACK | 严格桌面 `MatMul` → 生产模型且不允许 ORT CPU EP 回退 → 在目标 CPU 上恢复回退并调优 |

---

## 如何解读结果

看到绿色通过标记，并不一定表示加速器已经执行了计算。下面列出各类结果分别可以确认什么。

```mermaid
flowchart LR
    L["EP 出现在列表"] --> W1["Runtime 能够加载"]
    C["会话创建成功"] --> W2["配置和初始化已被接受"]
    O["输出与参考一致"] --> W3["数值结果有效"]
    G["节点分配/性能分析标记 EP"] --> W4["计算由目标 EP 执行"]
    N["无 CPU 节点/事件"] --> W5["没有发生 CPU 回退"]
    Speed["仅延迟较低"] -.-> X["不能据此判断执行设备"]
```

| 信号 | 可以确认什么 |
|---|---|
| EP 出现在可用列表 | 当前运行时提供或能够加载该 EP |
| 会话创建成功 | EP 接受了配置并完成模型初始化 |
| 输出与独立参考一致 | 冒烟测试结果在文档给定的误差范围内正确 |
| 计算图分配或性能分析记录标记目标 EP | 本次运行中的计算图工作确实交给了目标 EP |
| 禁用回退的测试中没有 CPU 节点/事件 | 该测试所检查的记录中没有发现计算图回退到 CPU |
| 仅延迟较低 | **不能据此确认**硬件加速，也不能代表生产环境性能 |

对于 CoreML 以及其他内部还包含调度器的 EP，这张表只能证明 EP 边界。要区分已接受分区内部实际使用的是 CPU、GPU 还是 NPU/ANE，请使用对应的设备级工具。

> [!NOTE]
> 仓库自带的模型只用于**验证配置和执行路径，不是性能基准**。通过测试后，还需要使用生产模型、真实输入形状、代表性数据、实际预热策略、精度模式和应用级准确率指标重新验证。

---

## 目录说明

| 路径 | 内容 |
|---|---|
| [Apple](Apple/README.zh-CN.md) | CoreML EP 配置、严格 macOS 证明、CPU/GPU/ANE 放置边界、iOS 路线和源码深度解析 |
| [DirectML](DirectML/README.zh-CN.md) | 跨厂商 DirectML 和 Windows ML 配置、严格一键证明、EP 目录流程以及 DML/WinML 源码深度解析 |
| [AMD](AMD/README.zh-CN.md) | DirectML、Windows ML MIGraphX、ROCm/MIGraphX 与 Ryzen AI/Vitis AI |
| [Intel](Intel/README.zh-CN.md) | Intel CPU、GPU、NPU 与 Meta-device 的 OpenVINO EP |
| [NVIDIA](NVIDIA/README.zh-CN.md) | CUDA、传统 TensorRT 与 TensorRT RTX 插件 |
| [Qualcomm](Qualcomm/README.zh-CN.md) | Snapdragon Windows 与 Android 的 QNN 2.x 插件 |
| [Qualcomm/AndroidDemo](Qualcomm/AndroidDemo/README.zh-CN.md) | Kotlin CPU/GPU/HTP 应用与一键构建/安装脚本 |
| [WebGPU](WebGPU/README.zh-CN.md) | 浏览器 WASM/WebGPU/WebNN 与原生 Python WebGPU |
| [WebGPU/onnxruntime-web-demo](WebGPU/onnxruntime-web-demo/README.zh-CN.md) | 浏览器/原生跨 Provider 冒烟测试 |
| [XNNPACK](XNNPACK/README.zh-CN.md) | XNNPACK EP 源码指南、移动端包、桌面源码构建、线程、算子限制和严格 CPU 路径证明 |
| [PluginEP](PluginEP/README.zh-CN.md) | Plugin EP ABI、加载器、工厂/设备、执行路径、兼容性与打包的源码级指南 |

---

## 许可证

本仓库采用 [Apache License 2.0](LICENSE)。
