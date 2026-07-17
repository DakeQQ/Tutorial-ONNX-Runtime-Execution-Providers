# ONNX Runtime + XNNPACK：跨平台 CPU 推理指南

[English](README.md) · [仓库首页](../README.zh-CN.md) · [XNNPACK EP 官方文档](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html) · [已核验源码 `bf6aa006`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack)

| 项目 | 本文基线 |
|---|---|
| 最近核验 | `2026-07-17`；源码以 ONNX Runtime `main` 的 [`bf6aa006`](https://github.com/microsoft/onnxruntime/commit/bf6aa0063d1c178c4a4d33ed6770425834147e2a) 提交为准，稳定版示例则以 `v1.27.1` 的 [`df2ba1cf`](https://github.com/microsoft/onnxruntime/commit/df2ba1cf8108aa63627cf4cdf8f807880b938616) 提交为准 |
| 加速范围 | XNNPACK 会根据 CPU 架构选择相应的微内核来加速推理；它**不使用** GPU 或 NPU |
| 官方预编译包 | Android 使用 Maven 包 `onnxruntime-android`；iOS 使用 CocoaPods 包 `onnxruntime-c` / `onnxruntime-objc` |
| 桌面端 Python | 需要通过 `--use_xnnpack` 从源码构建 ONNX Runtime；普通 PyPI wheel 不保证包含该 EP |
| 入口 | [`one_click.py`](one_click.py) |
| 验证方式 | 运行结果确定的 `MatMul`，与独立的 NumPy 结果对比，同时检查当前 Session 的计算图分配和/或 profile，并禁用回退到 ORT CPU EP |
| 已验证范围 | 启动脚本的单元测试已在 Linux 上通过；若要完成源码构建和严格推理，还需能访问 GitHub/codeload，并安装下文列出的构建依赖 |

### 如何理解本文结论

| 结论类型 | 本文采用的依据 | 可以说明什么 |
|---|---|---|
| 软件包与公开 API | [XNNPACK 官方页面](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)和[官方构建指南](https://onnxruntime.ai/docs/build/eps.html#xnnpack) | 官方提供哪些软件包、API 名称以及公开配置项 |
| 稳定版行为 | 固定版本的 ORT [`v1.27.1` 源码](https://github.com/microsoft/onnxruntime/tree/df2ba1cf8108aa63627cf4cdf8f807880b938616/onnxruntime/core/providers/xnnpack) | 一键脚本的行为，以及第 7 节所列的 capability 判断规则 |
| 新版行为 | 已核验的 `main` [提交 `bf6aa006`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack) | 稳定版发布后的修复和源码变化 |
| 本仓库中的行为 | `one_click.py` 单元测试，以及严格的节点分配/profile 检查 | 脚本在实际运行环境中的表现 |
| 性能 | 在目标设备上使用生产模型实测 | 速度、内存占用、功耗和温度；这些指标无法仅靠阅读源码得出 |

> [!IMPORTANT]
> XNNPACK 本来就是在 **CPU** 上运行的。本文所说的“没有回退到 CPU”，特指计算图节点没有交给 ONNX Runtime 通用的 `CPUExecutionProvider`，并不是说推理没有使用 CPU。严格测试验证的是 XNNPACK 这条软件执行路径，而不是某个独立的硬件设备。

---

## 目录

- [1. 了解 XNNPACK 的工作方式](#1-了解-xnnpack-的工作方式)
- [2. 选择预编译包或源码构建](#2-选择预编译包或源码构建)
- [3. 运行一键严格验证](#3-运行一键严格验证)
- [4. 源码结构](#4-源码结构)
- [5. 分图、布局转换与融合](#5-分图布局转换与融合)
- [6. 线程与并发](#6-线程与并发)
- [7. 从源码确认算子支持范围](#7-从源码确认算子支持范围)
- [8. 内核与内存生命周期](#8-内核与内存生命周期)
- [9. API 示例](#9-api-示例)
- [10. 从冒烟测试走向生产](#10-从冒烟测试走向生产)
- [11. 故障排查](#11-故障排查)
- [12. 主要参考资料](#12-主要参考资料)

---

## 1. 了解 XNNPACK 的工作方式

[XNNPACK](https://github.com/google/XNNPACK) 是一个面向 Arm、x86 和 WebAssembly CPU 的神经网络算子优化库。ONNX Runtime 的 XNNPACK Execution Provider（EP）会判断哪些 ONNX 节点可以由 XNNPACK 执行，必要时转换数据布局，然后创建 XNNPACK operator 并调用相应的微内核。

```mermaid
flowchart LR
    APP["应用程序"] --> ORT["ONNX Runtime Session"]
    ORT --> CAP["XNNPACK GetCapability"]
    CAP --> CHECK["检查 NodeUnit 是否受支持"]
    CHECK --> LAYOUT["布局敏感算子<br/>NCHW 转为内部 NHWC"]
    LAYOUT --> KR["静态 XNNPACK 内核注册表"]
    CHECK --> KR
    KR --> OP["create / reshape / setup / run"]
    OP --> MK["Arm · x86 · WASM<br/>CPU 微内核"]
    CHECK -. 不支持的节点 .-> CPU["ORT CPUExecutionProvider"]
```

### XNNPACK 是什么，不是什么

| 问题 | 答案 |
|---|---|
| XNNPACK 是 CPU 后端吗？ | 是。输入、输出和内存分配都在 CPU 内存中完成，计算也由 CPU 执行。 |
| 需要 GPU/NPU 驱动吗？ | 不需要。XNNPACK 会在构建和运行时选择合适的 CPU 微内核。 |
| 它等同于 ORT CPU EP 吗？ | 不等同。两者虽然都使用 CPU，但内核注册表、数据布局、线程模型和算子支持范围均不相同。 |
| 一定更快吗？ | 不一定。不受支持的节点、布局转换、小张量、线程争用和模型 shape 都可能抵消加速收益，必须使用生产模型实测。 |
| `get_available_providers()` 能确认已使用 XNNPACK 吗？ | 不能。它只能说明二进制在构建时包含了 XNNPACK。还需要检查当前会话的节点分配或 profile。 |
| 生产环境可以保留 CPU EP 作为回退吗？ | 可以，而且通常有助于扩大算子覆盖范围。本文在严格验证中禁用它，是为了确保任何回退都会让验证失败。 |

```mermaid
%%{init: {"themeCSS": ".mindmap-node text { fill: #20242b !important; } .mindmap-node span { color: #20242b !important; } .mindmap-node.root rect, .mindmap-node.root circle { fill: #ffffff !important; stroke: #20242b !important; } .mindmap-node.root text { fill: #000000 !important; } .mindmap-node.root span { color: #000000 !important; }"}}%%
mindmap
    root((XNNPACK EP))
        在 CPU 上运行
            Arm
            x86
            WebAssembly
        ORT 负责决策
            判断节点支持
            NHWC 改写
            QDQ 或激活融合
        XNNPACK 负责执行
            create
            reshape
            setup
            run
        验证依据
            输出正确
            XNNPACK 分配或 profile
            CPU EP 节点为零
```

---

## 2. 选择预编译包或源码构建

### 2.1 支持与打包矩阵

| 目标平台 | 官方发行方式 | 启用方式 | 本仓库提供的内容 |
|---|---|---|---|
| Android | Maven [`com.microsoft.onnxruntime:onnxruntime-android`](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android) 已含 XNNPACK | Java `SessionOptions.addXnnpack(...)` | 提供配置说明；Android 应用应使用移动端官方包 |
| iOS | CocoaPods `onnxruntime-c` 和 `onnxruntime-objc` 已含 XNNPACK | C/C++ 或 Objective-C 封装接口 | 提供配置说明；构建和打包需要 macOS 与 Xcode |
| Windows | 使用 `--use_xnnpack` 自定义构建 ORT | C、C++ 或自定义 Python wheel | 一键脚本支持 |
| Linux | 使用 `--use_xnnpack` 自定义构建 ORT | C、C++ 或自定义 Python wheel | 一键脚本支持 |
| WebAssembly | XNNPACK 和 ORT 都支持 WASM 构建 | 取决于具体构建所提供的 JavaScript/C API | 本文只介绍源码；浏览器演示请参阅 Web 目录 |

> [!WARNING]
> 在桌面端，不能仅靠 `pip install onnxruntime` 安装 XNNPACK。如果 `onnxruntime.get_available_providers()` 中没有 `XnnpackExecutionProvider`，请改用通过 `--use_xnnpack` 构建的自定义 wheel。看到 `CPUExecutionProvider` 并不代表 XNNPACK 已启用。

### 2.2 桌面构建依赖

一键脚本固定使用 ONNX Runtime `v1.27.1`，并在构建前核对提交 `df2ba1cf8108aa63627cf4cdf8f807880b938616`，确保源码版本没有发生变化。

| 启动脚本检查项 | 要求 |
|---|---|
| 主机 | Linux 或 Windows；64 位进程 |
| Python | CPython 3.11–3.14 |
| CMake | 3.28 或更新版本 |
| 编译器 | Linux `cc` + `c++`（ORT 会拒绝低于 11.1 的 GCC）；Windows 使用 Visual Studio 2022 的 `cl` |
| 构建工具 | 优先使用 Ninja；Linux 没有 Ninja 时可回退到 Make |
| 隔离的 Python 环境 | 使用指定提交构建的 ORT `1.27.1` 自定义 wheel，以及 `onnx==1.22.0`；脚本会自动安装并再次核对版本 |

**Ubuntu 24.04 / Debian 系基线：**

```bash
sudo apt update
sudo apt install -y build-essential git python3-dev python3-venv ninja-build
cmake --version
python3 --version
```

Ubuntu 24.04 自带的软件包可以满足这些要求。较旧的发行版需要从组织认可的来源安装新版 CMake 和编译器。建议安装 `ninja-build`；如果系统中没有 Ninja，脚本会改用 Make。

**Windows 基线：**

1. 安装 64 位 CPython 3.11–3.14。
2. 安装 Git for Windows。
3. 安装 CMake 3.28+ 并加入 `PATH`。
4. 安装 Visual Studio 2022，勾选 **Desktop development with C++**、MSVC 和当前 Windows SDK。
5. 在 **x64 Native Tools Command Prompt for VS 2022** 中运行。

桌面端启动脚本不支持 macOS；如需部署到 Apple 移动设备，请使用官方 iOS 软件包。

---

## 3. 运行一键严格验证

在仓库根目录执行：

```bash
python XNNPACK/one_click.py
```

首次运行会从源码完整构建 ONNX Runtime，因此可能需要较长时间。脚本会依次完成以下工作：

1. 创建 `XNNPACK/.venv-xnnpack`；
2. 获取指定版本的 ONNX Runtime 源码，并核对准确的提交 ID；
3. 使用 `--use_xnnpack` 构建 Release 版 Python wheel，同时关闭与本示例无关的单元测试目标；
4. 安装该 wheel 和指定版本的模型生成依赖，再核对 wheel 中记录的 ORT 构建提交；
5. 生成一个静态 FP32 `MatMul` 模型，其中右侧矩阵是常量 initializer；
6. 只请求 `XnnpackExecutionProvider`，关闭 Python 的重试回退，并禁止 ORT 隐式注册的 CPU EP 接收任何计算图节点；
7. 将输出与独立计算的 NumPy 结果比较；
8. 确认计算图分配或 profile 中出现 XNNPACK，同时确保没有节点交给 ORT CPU EP。

常用参数：

```bash
# 复用已经构建好的自定义 wheel。
python XNNPACK/one_click.py --wheel /path/to/onnxruntime-1.27.1-*.whl

# 重新创建源码和构建目录。
python XNNPACK/one_click.py --refresh

# 分别设置源码构建并行数与 XNNPACK 请求的线程数。
python XNNPACK/one_click.py --jobs 4 --threads 8

# 快速离线测试启动脚本；不构建 ONNX Runtime。
python XNNPACK/one_click.py --unit-tests
```

验证成功时会看到类似以下内容：

```text
Assignment evidence : {'XnnpackExecutionProvider': 1}
Profile evidence    : {'XnnpackExecutionProvider': 3}
Max abs error       : ...
[PASS/通过] XNNPACK executed the model with correct output and no CPU fallback.
```

Profile 事件数会随预热次数和正式运行次数变化，因此通过条件不依赖某个固定数字。脚本要求当前会话中至少出现一次 XNNPACK 节点分配或 profile 事件、输出结果正确，并且分配给 `CPUExecutionProvider` 的图节点数为零。

### 为什么冒烟模型选择 `MatMul`

在该 EP 中，`MatMul` 不受数据布局影响，并且直接对应一个静态 XNNPACK 内核。卷积虽然能测试更多布局转换逻辑，但 ORT 可能会在内部 NHWC 区域的边界插入转置。使用单个受支持的 `MatMul`，更容易判断禁用回退后的结果。验证生产模型时，仍需覆盖模型实际使用的算子和布局。

---

## 4. 源码结构

### 4.1 文件职责

| 源码 | 职责 |
|---|---|
| [`xnnpack_provider_factory.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/xnnpack_provider_factory.cc) | 保存 Provider 和会话参数，并创建 `XnnpackExecutionProvider` |
| [`xnnpack_execution_provider.h`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/xnnpack_execution_provider.h) | 声明首选的 NHWC 布局、选择性融合方式、会话内 `Run()` 串行化、内存分配器和私有线程池 |
| [`xnnpack_execution_provider.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/xnnpack_execution_provider.cc) | 注册静态内核、管理线程池、初始化 XNNPACK，并实现两轮 `GetCapability` 检查 |
| [`detail/node_support_checker.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/detail/node_support_checker.cc) | 调用各 ONNX/NodeUnit 支持检查，并判断 `Clip`/`Relu` 能否融合 |
| [`detail/utils.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/detail/utils.cc) | 处理 QDQ 分类与融合、激活函数 MetaDef、量化参数解析和 padding 模式判断 |
| [`xnnpack_kernel.h`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/xnnpack_kernel.h) | 内核基类，用于保存私有线程池和可选的 XNNPACK 缓存 |
| [`xnnpack_init.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/xnnpack_init.cc) | 将 ORT CPU 内存分配器适配为 XNNPACK 所需的分配回调表 |
| [`nn/`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/nn) | Conv、ConvTranspose、AveragePool 和 MaxPool 的支持检查与内核实现 |
| [`math/`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/math) | Gemm、MatMul 和 Softmax 的支持检查与内核实现 |
| [`tensor/resize.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack/tensor/resize.cc) | 双线性 Resize 的支持检查和 XNNPACK operator 生命周期管理 |
| [`onnxruntime_providers_xnnpack.cmake`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/cmake/onnxruntime_providers_xnnpack.cmake) | 把 EP 构建为静态库并定义 `USE_XNNPACK` |
| [`external/xnnpack.cmake`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/cmake/external/xnnpack.cmake) | 获取 XNNPACK/pthreadpool/fxdiv、选择目标架构，并在适用的 Arm 平台加入 KleidiAI |

### 4.2 创建调用链

```mermaid
sequenceDiagram
    participant API as C / C++ / Java / Python
    participant REG as Provider 注册层
    participant FAC as XnnpackProviderFactory
    participant EP as XnnpackExecutionProvider
    participant XNN as XNNPACK

    API->>REG: 添加 XNNPACK + Provider 参数
    REG->>FAC: Create(options, SessionOptions)
    FAC->>EP: XnnpackExecutionProviderInfo
    EP->>EP: 解析 intra_op_num_threads
    EP->>EP: 大于 1 时创建私有 pthreadpool
    EP->>EP: CreatePreferredAllocators()
    EP->>XNN: xnn_initialize(ORT allocator callbacks)
```

该 Provider 会静态链接到 ONNX Runtime 主二进制中，不会像采用 provider bridge 的 EP 那样单独发布 `onnxruntime_providers_*.so/.dll`。只有在编译时启用 `USE_XNNPACK`，`get_available_providers()` 才会列出它。

### 4.3 使用静态内核，而不是编译子图

当前的 `GetCapability` 会返回与静态 `KernelRegistry` 对应的 `ComputeCapability`，并不负责把子图交给编译器。源码预留了将来加入编译内核的位置；在现有实现中，分图后最终使用的是已注册的 `OpKernel` 类。

---

## 5. 分图、布局转换与融合

### 5.1 为什么要执行两轮支持检查

`XnnpackExecutionProvider::GetPreferredLayout()` 返回 `NHWC`，而标准 ONNX Conv/Pool 模型通常使用 NCHW，因此 XNNPACK EP 需要与 ORT 的布局转换器配合：

```mermaid
flowchart TD
    G["原始计算图"] --> U["构建 NodeUnit<br/>其中包括 QDQ 组"]
    U --> P1["第一轮支持检查<br/>检查尚未分配的目标节点"]
    P1 --> EACH["逐个返回 NodeUnit 中的节点"]
    EACH --> LT["ORT 布局转换<br/>NCHW 转为内部 NHWC 域"]
    LT --> P2["第二轮支持检查<br/>节点已标记为 XNNPACK"]
    P2 --> QDQ["将支持的 QDQ 组<br/>融合为一个 MetaDef"]
    P2 --> ACT["必要时融合唯一的<br/>Clip 或 Relu 消费节点"]
    QDQ --> K["匹配静态 XNNPACK 内核"]
    ACT --> K
```

关键点：

- `QDQ::GetAllNodeUnits` 会把量化算子及其周围的 Quantize/Dequantize 节点作为一个整体处理。
- 第一轮根据目标算子判断是否支持，并分别申领组内节点，以便后续进行布局转换。
- 第二轮将支持的 QDQ 组融合成 `QLinearConv`，或动态域中的 `QLinearSoftmax` 等 MetaDef。
- 布局敏感内核注册在 ORT 的内部 NHWC 域；Gemm、MatMul 和 Softmax 仍位于 ONNX 域。
- 不支持的节点会留给优先级较低的 EP。即使应用没有主动添加 CPU EP，ORT 仍会隐式注册默认的 CPU EP。本文的严格验证只请求 XNNPACK，同时关闭 Python 创建会话时的重试回退，并设置 `session.disable_cpu_ep_fallback=1`。只要有节点未分配，或被交给隐式注册的 CPU EP，会话初始化就会失败。

### 5.2 激活函数融合

该 Provider 没有为独立的 `Relu` 或 `Clip` 注册 XNNPACK 内核，但可以将它们融合到前面的内部 NHWC `Conv`、`MaxPool` 或 `AveragePool` 中。具体做法是在前驱算子的 MetaDef 中记录输出的最小值和最大值。量化 QDQ 组不参与这种激活函数融合。

> [!WARNING]
> 稳定版 `v1.27.1` 早于修复提交 [`86cbd205`](https://github.com/microsoft/onnxruntime/commit/86cbd2052540c59ad54f5ca135f9b0f58453557a)。如果激活函数之前的输出同时也是计算图输出，或还有其他消费节点，该修复会拒绝融合。`v1.27.1` 仍可能融合这类分支图，随后因为输入悬空而无法创建会话。一键 `MatMul` 验证不受影响。如果生产模型的分支还会使用 Conv/Pool 在激活前的输出，请升级到包含该修复的版本，或回移该提交，然后重新检查节点分配。

---

## 6. 线程与并发

XNNPACK 和 ORT 各自维护一套 intra-op 线程池。如果两套线程池申请的线程总数超过硬件承载能力，性能可能明显下降。

| 配置 | 所有者 | 源码行为 |
|---|---|---|
| `SessionOptions.intra_op_num_threads` | ORT | 控制 ORT 的 intra-op 线程池 |
| `session.intra_op.allow_spinning` | ORT | 等待时自旋会占用 CPU；XNNPACK 负责计算线程时应关闭 |
| XNNPACK `intra_op_num_threads` | XNNPACK EP | 公开接口要求该值 >= `1`，默认值及实际效果均为 `1`。内部用 `0` 表示调用方未提供配置，此时会复制 ORT 会话原先的线程设置；只有最终结果大于 `1` 时才创建私有 pthreadpool。调优时建议显式设置。 |
| `ConcurrentRunSupported()` | XNNPACK EP | 返回 `false`；因此 ORT 会通过会话级锁，让同一会话中的 `Run()` 调用串行执行 |

推荐起点：

```python
import os
import onnxruntime as ort

physical_cores = max(1, (os.cpu_count() or 1) // 2)
options = ort.SessionOptions()
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.intra_op_num_threads = 1
options.add_session_config_entry("session.intra_op.allow_spinning", "0")

session = ort.InferenceSession(
    "model.onnx",
    sess_options=options,
    providers=[
        ("XnnpackExecutionProvider", {"intra_op_num_threads": str(physical_cores)}),
        "CPUExecutionProvider",  # 普通生产回退；严格验证时删除
    ],
)
```

接下来应使用真实的并发量和模型 shape 进行测试。如果计算量较大的不支持节点交给了 ORT CPU EP，增大 ORT 线程池反而可能更快；官方文档也明确建议分别测试这两种配置。

```mermaid
flowchart TD
    S["起点：ORT 线程数设为 1<br/>关闭 spinning"] --> A{"计算量大的节点<br/>由哪个 EP 执行？"}
    A -->|XNNPACK| X["扫描 XNNPACK 线程数<br/>1 到物理核心数"]
    A -->|CPU EP 回退| C["测试更大的<br/>ORT 线程池"]
    X --> Q["测试真实形状和<br/>请求并发量"]
    C --> Q
    Q --> R["采用实测表现更好的配置"]
```

> [!NOTE]
> 在已核验的 `v1.27.1` 中，Gemm 和 MatMul 虽然在 reshape 阶段使用私有线程池，但调用 `xnn_run_operator` 时传入的是 `nullptr`。因此，不能认为增加 XNNPACK 线程数一定会加速这两个内核。Conv、Pool、Softmax 和 Resize 的执行路径则会传入私有线程池。所有性能结论都必须注明所用模型和源码版本。

---

## 7. 从源码确认算子支持范围

官方页面只提供摘要。下表结合 `v1.27.1` 的检查器、内核注册表和内核实现，整理出适合生产环境采用的保守支持范围。对于检查器接受、但实际实现仍存在风险的情况，后文会单独列出。

| 算子 / 模式 | 生产环境中建议采用的支持范围 |
|---|---|
| `Conv` | 支持检查要求 opset >= 11；rank 为 3（1D，内部将高度视为 1）或 4（2D）；C 和空间维度已知；权重及可选 bias 为常量；使用 FP32，或在兼容平台上使用 FP16；支持 group/depthwise；auto-pad 仅限 `NOTSET`、`VALID`、`SAME_UPPER` |
| `QLinearConv`、QDQ `Conv` | UINT8 或 INT8，输入/权重/输出类型需匹配；scale、zero point、权重和可选 bias 为常量；INT8 权重按输出通道量化时，权重 zero point 必须为 0 或省略；不支持 U8/S8 混合 |
| `ConvTranspose`、QDQ/`QLinearConvTranspose` | 沿用 ConvBase 对 rank、静态维度和常量的检查，并使用反卷积执行路径；建议采用按张量量化，因为显式的 `QLinearConvTranspose` 会拒绝 INT8 按通道权重 |
| `AveragePool` | opset >= 7；rank 为 4；C/H/W 已知；使用二维且非 1x1 的内核；`ceil_mode=0`；使用默认 dilation；支持 FP32，或在兼容平台上使用 FP16；浮点路径要求 `count_include_pad=0` |
| 量化 AveragePool | 内核注册表中虽然存在 `QLinearAveragePool`，但兼容性检查会拒绝所有量化 AveragePool，因此该版本实际不支持此模式。 |
| `MaxPool` | opset >= 8；rank 为 4；C/H/W 已知；不能请求可选的 indices 输出；使用二维且非 1x1 的内核；`ceil_mode=0`；auto-pad 仅限 `NOTSET`、`VALID`、`SAME_UPPER`；支持 FP32，以及兼容平台上的 FP16/UINT8/INT8 |
| QDQ `MaxPool` | 输入输出量化类型必须相同，且为 UINT8 或 INT8；Pool 计算本身不使用单独量化参数 |
| 普通 ONNX `Resize` | opset >= 10；rank 4；FP32、opset 10–18 上合格平台的 FP16、UINT8 或 INT8；`scales`/`sizes` 为常量；N/C 不变且 H/W 与输出 H/W 已知；`mode=linear`；禁止 antialias/axes/exclude-outside；aspect policy 为 stretch；extrapolation 为 0；坐标模式受限；降采样需通过因数检查 |
| QDQ `Resize` | 检查器会识别该模式，但融合后生成的是 ONNX 域中的 `Resize`，而对应的静态内核注册在内部 NHWC 域中；上游测试也处于禁用状态。因此，已核验版本不应使用此模式。 |
| `Gemm` | 使用 FP32，或在兼容平台上使用 FP16；A/B 为二维；`alpha=1`、`beta=1`、`transA=0`；B 为常量；C 省略，或使用常量的一维输出通道 bias；支持 `transB` |
| `MatMul` | 使用 FP32，或在兼容平台上使用 FP16；A 的 rank >= 1；B 是 rank 1 或 rank 2 的非空常量；运行时会将 N-D A 的外层维度展平为 batch |
| `Softmax` | 使用 FP32，或在兼容平台上使用 FP16；reduction 维度必须是静态值；opset >= 13 时只支持最后一个 axis；opset <= 12 时保留从 axis 开始展平的语义 |
| QDQ `Softmax` | 只支持 UINT8；输出 scale 必须约为 `1/256`、输出 zero point 为 `0`；融合成动态内部 `QLinearSoftmax` schema |
| `Relu` / `Clip` | 没有独立内核；可以融合到受支持的内部 NHWC Conv/MaxPool/AveragePool 之后，其中 Clip 的边界必须为常量。为了保证融合正确，前驱算子的输出不能同时作为图输出，并且该激活函数必须是唯一的消费节点。 |

### 源码核验发现的文档差异

| 官方简表 | 源码中的实际情况 |
|---|---|
| Conv 只支持 2D | 源码还接受 rank-3 的 1D Conv/ConvTranspose，并把高度视为 1 |
| MatMul 只支持 2D | A 可以是 N-D；常量 B 仍只能是 rank 1 或 rank 2 |
| 注册算子即可使用 | 量化 AveragePool 虽然已经注册，但兼容性检查仍会无条件拒绝它 |
| 内核注册表包含较旧的 Conv 版本 | `ConvBase::IsOnnxNodeSupported` 会在布局转换前拒绝 opset 11 以下的 ONNX Conv/ConvTranspose |
| XNNPACK 线程可加速 Gemm/MatMul | 在 `v1.27.1` 中，两者在 reshape 时会使用线程池，但调用 `xnn_run_operator` 时传入的却是 `nullptr` |

### 已知的检查器与内核实现缺口

前六项问题在已核验的 `main` 提交 `bf6aa006` 中仍然存在；最后一项激活函数融合问题已由 `86cbd205` 修复。

| 特殊情况 | 源码实际行为 | 建议做法 |
|---|---|---|
| INT8 Conv 按通道权重 zero point 非零 | 循环记录日志并 `break`，随后仍把类型设为按通道 INT8 | 权重 zero point 使用 0 或省略 |
| `Gemm transA=1` | 支持检查会接受该配置，但内核不会转置 A，而是直接把原始指针交给 fully connected operator | 使用 `transA=0` |
| 任意 `Gemm` C broadcasting | 支持检查会接受部分 rank-1/rank-2 shape，但 XNNPACK 只把 C 当作 bias 指针 | 省略 C，或使用长度等于输出宽度 N 的一维 bias |
| `Resize` 改变 batch，或 H/W 为动态值 | 检查器会检查 C，却不检查 N；`Compute` 强制输出 N 等于输入 N，而 operator 创建时会固定输出 H/W | 保持 N/C 不变，并固定 H/W |
| opset 19 的 FP16 `Resize` | 检查器接受 FP16，但 opset 19 注册的内核不包含 FP16 | FP16 使用 opset 10–18，或改用 FP32/UINT8/INT8 |
| opset <= 12 的 `Softmax` reduction 维度未知 | 内层循环的 `break` 不会拒绝节点；Session 创建时会预计算 channel 数 | `axis` 起的维度保持静态 |
| `v1.27.1` 中激活函数的前驱算子还有其他消费节点，或其输出同时也是图输出 | 稳定版不会拒绝融合该分支，因而可能留下悬空输入 | 升级到包含 `86cbd205` 的版本、回移该提交，或避免在分支中继续使用激活前的输出 |

### FP16 支持条件

Provider 会为 Arm/Arm64 以及非移动端的 x86/x64 目标编译 FP16 支持，随后再由 XNNPACK 判断当前硬件是否兼容。内核声明支持某种数据类型，并不代表每款处理器都能执行。建议先用 FP32 确认节点分配正确，再测试 FP16 的精度和设备兼容性。

### 动态 shape 的影响

XNNPACK 的执行流程是 create → reshape → setup → run，不过 create 阶段仍会固定一些关键的模型信息：

| 算子 | Session 创建时固定 | 仍可安全变化的运行时维度 |
|---|---|---|
| Conv / Pool | C/H/W、属性、常量权重 | Batch N |
| MatMul | rank-1/rank-2 常量 B 和 reduction 宽度 | A 的外层 batch 维度 |
| Softmax | 参与 reduction 的维度 | reduction 区域之前的维度 |
| Resize | 常量 scales/sizes 和输出 H/W；C 不变 | N 本身不做 resize 时，Batch N 可变化 |

在移动端部署时，应尽量固定 channel 和空间维度；也可以先设置 ORT free-dimension override（自由维度覆盖），再检查节点能否分配给 XNNPACK。

---

## 8. 内核与内存生命周期

### 8.1 通用执行模式

大多数内核都遵循同一套 XNNPACK API 调用流程：

1. **Create：**验证属性并一次性创建 `xnn_operator`；
2. **Reshape：**传入运行时 batch 和空间维度，并查询所需工作空间；
3. **Setup：**绑定本次输入输出指针；
4. **Run：**如果该内核提供了私有 pthreadpool，就使用该线程池执行。

`XnnpackOperatorDeleter` 通过 RAII 自动调用 `xnn_delete_operator`。Conv 和 Resize 使用经 ORT 适配的 XNNPACK 内存分配器，为每次运行分配满足对齐要求的工作空间。

### 8.2 布局和权重预打包

布局转换器会把 Conv/Pool/Resize 的激活张量转换为内部 NHWC 布局。常量卷积权重需要单独预打包，因为这并不是简单地转置激活张量：

- Conv：ONNX `M,C/group,kH,kW` → XNNPACK `M,kH,kW,C/group`；
- Group ConvTranspose 会显式加入 group 维，并把输入 channel 移到最内层；
- Gemm/MatMul 使用常量 B 创建 XNNPACK 全连接 operator。

因此，常量权重是支持检查中必须满足的条件：operator 创建和权重打包都发生在 `Compute` 之前。

### 8.3 内存分配器与缓存

| 项目 | 已核验版本的行为 |
|---|---|
| 内存分配器 | `CreatePreferredAllocators` 会延迟创建标记为 `XnnpackExecutionProvider` 的 CPU 内存分配器，安装回调表并调用 `xnn_initialize`；所有内存仍位于主机 CPU |
| 重新分配 | 空指针会按新分配处理；对已有指针执行 reallocate 会触发 `ORT_NOT_IMPLEMENTED` |
| 权重缓存 | Conv/Gemm/MatMul 只有在定义 `XNN_CACHE_ENABLE` 时才会请求权重缓存；已核验的两份 CMake 都没有定义该宏，一键脚本也未额外添加 |
| 代码缓存 | 公开的 `xnnpack.h` 没有提供 create/free API，因此代码缓存处于禁用状态 |

---

## 9. API 示例

### 9.1 使用自定义 wheel 的 Python

```python
import onnxruntime as ort

assert "XnnpackExecutionProvider" in ort.get_available_providers()

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.intra_op_num_threads = 1
options.add_session_config_entry("session.intra_op.allow_spinning", "0")

session = ort.InferenceSession(
    "model.onnx",
    sess_options=options,
    providers=[
        ("XnnpackExecutionProvider", {"intra_op_num_threads": "4"}),
        "CPUExecutionProvider",
    ],
)
```

进行严格验证时，请删除 `CPUExecutionProvider`，设置 `session.disable_cpu_ep_fallback=1`，启用 `session.record_ep_graph_assignment_info`，并参照本目录脚本检查当前会话的 profile。

### 9.2 C++

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "xnnpack"};
Ort::SessionOptions options;
options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
options.SetIntraOpNumThreads(1);
options.AddConfigEntry("session.intra_op.allow_spinning", "0");
options.AppendExecutionProvider(
    "XNNPACK", {{"intra_op_num_threads", "4"}});
Ort::Session session{env, model_path, options};
```

通用注册 API 同时接受短名称 `XNNPACK` 和规范名称 `XnnpackExecutionProvider`。

### 9.3 Android Java

```java
try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
    options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
    options.setIntraOpNumThreads(1);
    options.addConfigEntry("session.intra_op.allow_spinning", "0");
    options.addXnnpack(java.util.Collections.singletonMap("intra_op_num_threads", "4"));

    try (OrtSession session = environment.createSession(modelPath, options)) {
        // 执行推理。
    }
}
```

请使用官方 `onnxruntime-android` Maven artifact，其中的 native library 已包含 XNNPACK。按照 Java 绑定的要求，`SessionOptions` 必须一直存活到使用它的 Session 被销毁为止。

---

## 10. 从冒烟测试走向生产

### 10.1 分阶段验证

```mermaid
flowchart TD
    A["运行严格 MatMul 验证"] --> B{"XNNPACK 分配<br/>和输出都通过？"}
    B -->|否| C["修复构建 / Provider / 主机环境"]
    C --> A
    B -->|是| D["禁用 CPU 回退<br/>运行生产模型"]
    D --> E{"所需节点全部支持？"}
    E -->|否| F["检查算子、类型、形状、<br/>常量与布局限制"]
    E -->|是| G["按需要恢复 CPU 回退"]
    F --> G
    G --> H["测试线程数、预热、<br/>形状和并发"]
    H --> I["在目标设备验证应用精度<br/>并完成打包"]
```

### 10.2 模型检查表

| 检查 | 原因 |
|---|---|
| 使用受支持的算子/opset 组合 | 上游 XNNPACK 库的支持范围大于该 ORT EP 的内核注册表和检查器实现 |
| 固定 Conv/Pool 的 C/H/W | 创建内核前的支持检查需要这些维度 |
| 把 Conv/Gemm/MatMul 权重设为 initializer | 运行时传入权重会被拒绝 |
| 检查 NCHW↔NHWC 边界 | 对于小而分散的子图，转置开销可能超过内核带来的收益 |
| 核对 QDQ scale、zero point 和 tensor type | 混合类型或非常量量化参数会被拒绝 |
| 先验证 FP32，再测 FP16/INT8 | 把集成故障与精度/硬件兼容问题分开 |
| 有意识地测试两套线程池 | 默认配置可能竞争或超额订阅物理核心 |
| 在真实手机/CPU 上测试 | 模拟环境和桌面平台无法代表移动端的微内核选择及温控表现 |
| 计时前预热 | 首次权重打包、内存分配和缓存状态会影响延迟 |
| 在 CI 保留分配证据 | 包或构建变化可能移除 XNNPACK，而 CPU EP 仍让推理“成功” |

仓库自带的模型只用于确认 XNNPACK 执行路径是否正常，不能作为性能基准。生产环境的测试报告应使用有代表性的输入，并覆盖端到端延迟、吞吐量、内存占用、温度、功耗和任务级准确率。

---

## 11. 故障排查

| 现象 | 常见原因 | 处理方法 |
|---|---|---|
| 没有 `XnnpackExecutionProvider` | 使用普通 wheel，或构建未启用 `--use_xnnpack` | 运行一键脚本或安装已核验自定义 wheel；移动端使用官方包 |
| 源码构建无法下载 | GitHub/codeload 或依赖站点被网络策略阻断 | 配置组织认可的代理或缓存；不要改用未经核验的压缩包 |
| CMake 被拒绝 | CMake 低于 3.28 或工具链不支持 | 安装当前 CMake/compiler，然后用 `--refresh` 重试 |
| 禁用 CPU 回退后无法创建 Session | 至少一个节点不满足 XNNPACK 的支持条件 | 临时开启回退，检查详细日志和节点分配，再对照第 7 节排查 |
| Conv/Pool 仍由 CPU EP 执行 | C/H/W 为动态维度、padding 或其他属性不受支持、请求了可选输出，或者权重/bias 不是常量 | 固定相关维度并使用 initializer，或保留 CPU EP 作为回退 |
| 量化模型仍由 CPU EP 执行 | U8/S8 组合不匹配、量化参数是动态值、按通道 zero point 不合规，或 QDQ 模式不受支持 | 按照第 7 节核对 scale、zero point 和数据类型 |
| `v1.27.1` 中 `Clip`/`Relu` 分支导致图无效 | 该稳定版早于分支消费节点的融合修复 | 升级到包含 `86cbd205` 的版本、回移该提交，或避免在分支中使用激活前的输出 |
| 增加线程后反而变慢 | ORT 与 XNNPACK 线程池相互争用、拓扑识别不准确，或工作负载太小 | 将 ORT intra-op 设为 `1`、spinning 设为 `0`，再从 1 到物理核心数逐一测试 XNNPACK 线程数 |
| `--threads` 没有加速 MatMul/Gemm | 已核验版本的内核在执行阶段向 `xnn_run_operator` 传入 `nullptr`，而不是线程池 | 这是该版本的已知行为；请实测其他版本，或改用受支持且以 Conv 为主的工作负载 |
| 输出正确，但没有 XNNPACK 事件 | Provider 已加载，但节点没有分配给它；也可能是分配信息 API/profile 不可用 | 不能据此判定通过；请使用完整的自定义构建，并检查当前会话的节点分配和 profile |
| Android 模拟器很慢 | 模拟架构/微内核与目标设备不同 | 在真实 Arm 设备验证 |

---

## 12. 主要参考资料

- [XNNPACK Execution Provider 官方页面](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)
- [官方 EP 构建说明](https://onnxruntime.ai/docs/build/eps.html#xnnpack)
- [已核验的 `main` 源码快照（`bf6aa006`）](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/xnnpack)
- [实时 `main` 源码（可能变化）](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/xnnpack)
- [固定版本的 Provider 源码（`v1.27.1`，`df2ba1cf`）](https://github.com/microsoft/onnxruntime/tree/df2ba1cf8108aa63627cf4cdf8f807880b938616/onnxruntime/core/providers/xnnpack)
- [已核验 `main` 中的 Provider 注册实现](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/provider_registration.cc)
- [已核验 `main` 中的 Python Provider Factory](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/python/onnxruntime_pybind_state.cc)
- [已核验 `main` 中的 XNNPACK CMake 集成](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/cmake/external/xnnpack.cmake)
- [XNNPACK 上游项目](https://github.com/google/XNNPACK)
- [`v1.27.1` 之后的 activation side-consumer 修复](https://github.com/microsoft/onnxruntime/commit/86cbd2052540c59ad54f5ca135f9b0f58453557a)
