# ONNX Runtime Execution Provider Tutorials

Reproducible setup guides and **strict** smoke tests that prove your ONNX model truly runs through **Apple, AMD, Intel, NVIDIA, Qualcomm, Web, cross-vendor Windows, or XNNPACK CPU** backends — not just that a provider *could* load.

[简体中文](README.zh-CN.md)  ·  **Last verified: 2026-07-17.** Every platform guide pins its own packages, hardware gates, tested environments, and validation limits.

---

## Contents

- [The whole repo in one picture](#the-whole-repo-in-one-picture)
- [Your journey in four moves](#your-journey-in-four-moves)
- [The one idea to remember](#the-one-idea-to-remember)
- [Plugin EP is a loading model, not hardware](#plugin-ep-is-a-loading-model-not-hardware)
- [Pick your path](#pick-your-path)
- [The road to a trustworthy pass](#the-road-to-a-trustworthy-pass)
- [How to read your results](#how-to-read-your-results)
- [Repository map](#repository-map)
- [License](#license)

---

## The whole repo in one picture

Six platform families plus cross-vendor Windows and XNNPACK CPU routes, one workflow, one promise: **evidence over assumptions**.

```mermaid
flowchart LR
    ROOT(["ONNX Runtime<br/>EP Tutorials"])
    ROOT --> APPLE["Apple"]
    ROOT --> WIN["Windows"]
    ROOT --> AMD["AMD"]
    ROOT --> INTEL["Intel"]
    ROOT --> NV["NVIDIA"]
    ROOT --> QC["Qualcomm"]
    ROOT --> WEB["Web"]
    ROOT --> XNN["XNNPACK CPU"]

    APPLE --> APPa["CoreML CPU"] & APPb["GPU"] & APPc["Neural Engine"]
    WIN --> WINa["DirectML GPU"] & WINb["Windows ML EP Catalog"] & WINc["Automatic EP policy"]
    AMD --> AMDa["DirectML"] & AMDb["MIGraphX"] & AMDc["Ryzen AI / Vitis AI"]
    INTEL --> INTa["OpenVINO CPU"] & INTb["GPU / GPU.x"] & INTc["NPU"] & INTd["Meta-devices"]
    NV --> NVa["CUDA EP"] & NVb["Classic TensorRT"] & NVc["TensorRT RTX"]
    QC --> QCa["QNN HTP / NPU"] & QCb["QNN GPU"] & QCc["Android app"]
    WEB --> WEBa["WASM"] & WEBb["WebGPU"] & WEBc["WebNN"] & WEBd["Native Python WebGPU"]
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

## Your journey in four moves

Learn this loop once and reuse it on every platform.

```mermaid
flowchart LR
    P["1 · PICK<br/>your hardware"] --> S["2 · SET UP<br/>isolated env + pinned stack"]
    S --> R["3 · PROVE<br/>ORT → EP → CPU / GPU / NPU"]
    R --> T["4 · TRUST<br/>output + assignment + no fallback"]
```

| Move | You do | You get |
|---|---|---|
| **1 · Pick** | Match your exact device, OS, and driver to a guide | The right route and its support gates |
| **2 · Set up** | Create a clean venv with the guide's pinned stack | One ONNX Runtime distribution, no conflicts |
| **3 · Prove** | Run the route's strict smoke test | A real inference through the requested EP |
| **4 · Trust** | Read the evidence the test prints | Proof the requested EP handled the graph, with device-level proof where that EP exposes it |

---

## The one idea to remember

> [!IMPORTANT]
> A provider appearing in `get_available_providers()` only means the runtime **can load** it. It does **not** mean your model's nodes actually **ran through that EP**.

That single gap is why so many "it works" demos quietly fall back to CPU. Every test here closes it with four independent checks:

```mermaid
flowchart TD
    Q["get_available_providers()<br/>lists my EP"]
    Q -->|only proves| A["the runtime can load it"]
    Q -.->|does NOT prove| B["your nodes ran through that EP"]
    subgraph Need["Real proof = all four together"]
      direction LR
      M["Deterministic<br/>smoke model"] --> R["Independent<br/>reference"]
      R --> G["Assignment /<br/>profile evidence"]
      G --> F["No CPU<br/>fallback"]
    end
    B --> M
```

| Check | Confirms | Guards against |
|---|---|---|
| Deterministic smoke model | The same input every run | Flaky, unreproducible results |
| Independent reference | The output is numerically correct | Silently wrong math |
| Assignment / profile evidence | Nodes ran on the target EP | Invisible ORT CPU execution |
| No CPU fallback | ORT did not move unsupported graph work to its CPU EP | Quiet ORT CPU fallback that looks like success |

> [!NOTE]
> Some EPs contain another scheduler. CoreML, for example, may use CPU, GPU, or ANE *inside* a partition already assigned to `CoreMLExecutionProvider`. ORT assignment proves the EP boundary; use that provider's device-level profiler when the physical compute unit matters.

> [!TIP]
> Use a **separate virtual environment per route**. `onnxruntime`, `onnxruntime-gpu`, `onnxruntime-openvino`, and `onnxruntime-directml` all import as the same `onnxruntime` module and must never be mixed.

---

## Plugin EP is a loading model, not hardware

> [!IMPORTANT]
> **Plugin EP is not another hardware provider.** It is ONNX Runtime's public C ABI, dynamic-loading boundary, and device-discovery model for execution providers.

A vendor can implement a brand-new out-of-tree EP with it, or an existing EP can add the interface and ship independently. ORT source supports both cases and also adapts built-in and legacy provider-bridge EPs into the same factory/device model.

```mermaid
flowchart LR
    EP["EP identity<br/>CUDA · QNN · WebGPU · vendor EP"] --> LOAD{"How is it loaded?"}
    LOAD --> IN["Built into ORT"]
    LOAD --> BR["Legacy provider bridge<br/>+ Plugin EP factory"]
    LOAD --> PL["Pure Plugin EP<br/>public C ABI only"]
    IN --> RUN["Same ORT graph partition + execution contract"]
    BR --> RUN
    PL --> RUN
```

This repository already exercises four plugin routes: Windows ML MIGraphX, QNN 2.x, standalone TensorRT RTX, and native WebGPU. Registration or device discovery alone is not an execution pass; each platform test still requires output, assignment/profile, and no-fallback evidence.

Read the [Plugin EP source deep dive](PluginEP/README.md) for the loader call chain, `OrtEpFactory` / `OrtEp` lifecycle, compile-versus-kernel execution paths, ABI evolution, packaging rules, and source links.

---

## Pick your path

Start from the hardware in front of you, open its guide, and run the first command.

```mermaid
flowchart TD
    A{"What are you targeting?"}
    A -->|M-series Mac, iPhone or iPad| APPLE["Apple<br/>CoreML EP"]
    A -->|Cross-vendor Windows deployment| WIN["Windows<br/>DirectML · EP Catalog · Auto selection"]
    A -->|AMD GPU or Ryzen AI NPU| AMD["AMD<br/>DirectML · MIGraphX · Vitis AI"]
    A -->|Intel CPU / GPU / NPU| INT["Intel<br/>OpenVINO EP"]
    A -->|NVIDIA GPU| NV["NVIDIA<br/>CUDA · TensorRT"]
    A -->|Snapdragon or Android| QC["Qualcomm<br/>QNN HTP · GPU"]
    A -->|Browser or native WebGPU| WEB["Web<br/>WASM · WebGPU · WebNN"]
    A -->|CPU inference on mobile / desktop| XNN["XNNPACK<br/>Arm · x86 · WASM CPU"]
    APPLE --> G["Open the guide →<br/>run the first command →<br/>read the proof"]
    WIN --> G
    AMD --> G["Open the guide →<br/>run the first command →<br/>read the proof"]
    INT --> G
    NV --> G
    QC --> G
    WEB --> G
    XNN --> G
```

| Platform | What you have | Hosts | First command | Guides |
|---|---|---|---|---|
| **Apple** | Apple Silicon Mac or iPhone/iPad via CoreML | macOS · iOS | `python3 Apple/one_click.py`<br/><sub>current Python route: macOS 14+ arm64</sub> | [EN](Apple/README.md) · [中文](Apple/README.zh-CN.md) |
| **Windows** | Any supported DirectX 12 GPU, or Windows ML catalog CPU/GPU/NPU | Windows 10/11; catalog EPs: Windows 11 24H2+ | `py -3.12 DirectML\one_click.py directml`<br/><sub>Windows ML: replace `directml` with `windowsml --allow-download`</sub> | [EN](DirectML/README.md) · [中文](DirectML/README.zh-CN.md) |
| **AMD** | AMD GPU (DirectML / MIGraphX) or Ryzen AI NPU (Vitis AI) | Windows · Ubuntu | `python AMD/provider_test.py --target dml`<br/><sub>swap `dml` → `migraphx` or `npu` for your host</sub> | [EN](AMD/README.md) · [中文](AMD/README.zh-CN.md) |
| **Intel** | Intel CPU, integrated/discrete GPU, or NPU via OpenVINO | Windows 11 · Ubuntu x86-64 | `bash Intel/run_demo.sh --device CPU`<br/><sub>Windows: `Intel\run_demo.bat --device CPU`</sub> | [EN](Intel/README.md) · [中文](Intel/README.zh-CN.md) |
| **NVIDIA** | NVIDIA GPU via CUDA, classic TensorRT, or TensorRT RTX | Windows 10/11 · Ubuntu x86-64 | `python NVIDIA/provider_test.py --provider cuda` | [EN](NVIDIA/README.md) · [中文](NVIDIA/README.zh-CN.md) |
| **Qualcomm** | Snapdragon HTP/NPU or GPU via QNN | Windows ARM64 · Android ARM64 | `python Qualcomm/one_click.py htp`<br/><sub>Android: `python Qualcomm/AndroidDemo/build_demo.py --install --backend htp`</sub> | [EN](Qualcomm/README.md) · [中文](Qualcomm/README.zh-CN.md) · [App](Qualcomm/AndroidDemo/README.md) |
| **Web** | Browser WASM/WebGPU/WebNN or native Python WebGPU | Browser-dependent; native wheels narrower | `bash WebGPU/onnxruntime-web-demo/run_demo.sh wasm`<br/><sub>Windows: `WebGPU\onnxruntime-web-demo\run_demo.bat wasm`</sub> | [EN](WebGPU/README.md) · [中文](WebGPU/README.zh-CN.md) · [Demo](WebGPU/onnxruntime-web-demo/README.md) |
| **XNNPACK** | Cross-platform Arm/x86 CPU microkernels | Android · iOS · Windows · Linux · WASM | `python XNNPACK/one_click.py`<br/><sub>desktop Python builds a pinned custom ORT wheel</sub> | [EN](XNNPACK/README.md) · [中文](XNNPACK/README.zh-CN.md) |

---

## The road to a trustworthy pass

Always climb from the simplest route to the strictest. Fix problems where they appear, then continue.

```mermaid
flowchart TD
    S1["1 · Check the gates<br/>hardware · OS · driver"] --> S2["2 · Build the env<br/>one ORT distro, pinned"]
    S2 --> S3["3 · Run the baseline<br/>simplest route first"]
    S3 --> S4["4 · Run the strict test"]
    S4 --> D{"Output + assignment<br/>+ fallback all pass?"}
    D -->|Yes| S5["5 · Repeat with the<br/>production model"]
    D -->|No| FIX["Fix driver / stack / model,<br/>then rerun step 3"]
    FIX --> S3
```

| Step | Do this | Pass condition |
|---:|---|---|
| 1 | Check the hardware, OS, and driver gates in the platform guide | The exact device is in the documented support scope |
| 2 | Create an isolated environment with the pinned stack | Dependency checks pass with one ORT distribution |
| 3 | Run the route's baseline | Apple: CoreML `CPUOnly` · Windows: DirectML adapter 0 · AMD: exact GPU/NPU stack · Intel: `CPU` · NVIDIA: CUDA · Web: WASM · XNNPACK: source-built `MatMul` proof |
| 4 | Run the repository's strict entry point | Output, assignment evidence, and fallback policy all pass |
| 5 | Repeat with the production model and real inputs | Operators, shapes, precision, and app metrics pass |

**Recommended order per platform:**

| Platform | Climb this ladder |
|---|---|
| Apple | CoreML `CPUOnly` → `ALL` → `CPUAndGPU` / `CPUAndNeuralEngine` → MLComputePlan or Instruments |
| Windows | DirectML adapter 0 → every intended DXGI adapter → installed Windows ML EP → catalog download and production provider |
| AMD | Separate GPU from NPU, then choose DirectML, MIGraphX, or Vitis AI for the exact host |
| Intel | `CPU` → explicit `GPU` / `GPU.x` / `NPU` → deployment meta-device |
| NVIDIA | CUDA → classic TensorRT; use a separate environment for the TensorRT RTX plugin |
| Qualcomm | Native Windows ARM64 → static QDQ first for HTP → physical Snapdragon device for Android |
| Web | WASM → WebGPU → WebNN; native Python WebGPU is a separate plugin route |
| XNNPACK | Strict desktop `MatMul` → production model without ORT CPU EP fallback → restore fallback and tune on the target CPU |

---

## How to read your results

Not every green checkmark means the accelerator ran. Here is what each signal actually proves.

```mermaid
flowchart LR
    L["Provider listed"] --> W1["Runtime can load it"]
    C["Session created"] --> W2["Config + init accepted"]
    O["Output matches reference"] --> W3["Numerically sane"]
    G["EP named in assignment/profile"] --> W4["Ran through the EP — real proof"]
    N["No CPU nodes/events"] --> W5["No CPU fallback — real proof"]
    Speed["Low latency alone"] -.-> X["Proves nothing"]
```

| Signal | What it proves |
|---|---|
| Provider appears in the available-provider list | The installed runtime exposes or can load that provider |
| Session creation succeeds | The provider accepted the configuration and model initialization |
| Output matches an independent reference | The result is numerically sane within its documented tolerance |
| Graph assignment or profile names the target EP | The current run executed graph work through the requested provider |
| Strict test reports no CPU nodes/events | No ORT CPU EP graph fallback was observed through that test's evidence channel |
| Low latency alone | **Not proof** of accelerator execution or production performance |

For CoreML and any other EP with an internal scheduler, this table stops at the EP boundary. Use its device-level tooling to distinguish CPU, GPU, and NPU/ANE placement within the accepted partition.

> [!NOTE]
> The included models are **qualification workloads, not benchmarks**. After a strict pass, repeat the checks with your production model, real shapes, representative inputs, warm-up policy, precision mode, and application-level accuracy metrics.

---

## Repository map

| Path | What lives here |
|---|---|
| [Apple](Apple/README.md) | CoreML EP setup, strict macOS proof, CPU/GPU/ANE placement boundary, iOS route, and source deep dive |
| [DirectML](DirectML/README.md) | Cross-vendor DirectML and Windows ML setup, strict one-click proof, EP catalog workflow, and DML/WinML source deep dive |
| [AMD](AMD/README.md) | DirectML, Windows ML MIGraphX, ROCm/MIGraphX, and Ryzen AI/Vitis AI setup and verification |
| [Intel](Intel/README.md) | OpenVINO EP setup for Intel CPU, GPU, NPU, and meta-devices |
| [NVIDIA](NVIDIA/README.md) | CUDA, classic TensorRT, and standalone TensorRT RTX setup and strict profiling tests |
| [Qualcomm](Qualcomm/README.md) | QNN 2.x plugin setup for Snapdragon Windows and Android |
| [Qualcomm/AndroidDemo](Qualcomm/AndroidDemo/README.md) | Complete Kotlin CPU/GPU/HTP application and one-click build/install launcher |
| [WebGPU](WebGPU/README.md) | Browser WASM/WebGPU/WebNN and native Python WebGPU guidance |
| [WebGPU/onnxruntime-web-demo](WebGPU/onnxruntime-web-demo/README.md) | Runnable cross-provider browser/native smoke test |
| [XNNPACK](XNNPACK/README.md) | Source-level XNNPACK EP guide, mobile packages, desktop source build, threading, operator guards, and strict CPU-path proof |
| [PluginEP](PluginEP/README.md) | Source-level guide to the Plugin EP ABI, loader, factories/devices, execution paths, compatibility, and packaging |

---

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
