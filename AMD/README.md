# ONNX Runtime + AMD: GPU and NPU

[简体中文](README.zh-CN.md) · [Repository index](../README.md) · [Official MIGraphX EP guide](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

ONNX Runtime reaches AMD hardware through four routes — **DirectML**, **Windows ML**, **ROCm/MIGraphX**, and **Ryzen AI/Vitis AI**. This guide picks the right one for your device, then *proves* it with real node placement, not just a provider name in a list.

| Item | Baseline |
|---|---|
| Last verified | `2026-07-17` against linked AMD, Microsoft, Canonical, ONNX Runtime, Docker Hub, and PyPI sources |
| Hosts | Windows and Ubuntu; exact gates vary by GPU/NPU generation |
| Routes | DirectML · Windows ML MIGraphX · ROCm/MIGraphX · Ryzen AI/Vitis AI |
| Entry point | [`provider_test.py`](provider_test.py) |
| Proof | Current-run node placement + output sanity; CPU parity for the built-in GPU model, or any model with `--compare-cpu` |
| Validation boundary | Script self-tests passed on Linux; final DirectML/Windows ML/MIGraphX/Vitis AI proof needs matching hardware |

### Files

| File | Purpose |
|---|---|
| [`README.md`](README.md) · [`README.zh-CN.md`](README.zh-CN.md) | This guide (English / 简体中文) |
| [`provider_test.py`](provider_test.py) | One-command setup, optional bootstrap, and strict proof test |

| You are... | Start at |
|---|---|
| On Windows with an AMD GPU, want the fastest test | [§9 DirectML](#9-simplest-python-path-directml) |
| On Windows, building a new app for Win 11 24H2+ | [§10 Windows ML + MIGraphX](#10-new-windows-path-windows-ml--amd-migraphx) |
| On Ubuntu with an AMD GPU | [§6 Install the matching ROCm track](#6-install-the-matching-rocm-track) |
| On a Ryzen AI laptop, Windows | [§12 Install Ryzen AI Software](#12-install-ryzen-ai-software-171) |
| On a Ryzen AI laptop, Ubuntu | [§15 Install the Linux NPU driver](#15-install-the-ubuntu-npu-driver-and-ryzen-ai) |
| Targeting a Zynq/Versal board | [§16 Embedded Linux targets](#16-embedded-linux-targets) |
| Asking "why did my node land on CPU?" | [§19 Verification flow](#19-verification-flow) + [§21 Troubleshooting](#21-troubleshooting) |
| Still not sure | [§1 Choose a route](#1-choose-a-route) |

> [!IMPORTANT]
> A provider appearing in `ort.get_available_providers()` only proves the library **can load**. It never proves a model node **actually executed** on that device. Every check in this guide closes that gap with current-run profile or assignment evidence — see [§19](#19-verification-flow).

---

## Contents

- [The whole AMD picture](#the-whole-amd-picture)
- [1. Choose a route](#1-choose-a-route)
- [2. Fundamentals](#2-fundamentals)
- [3. Version and support matrix](#3-version-and-support-matrix)
- [4. Zero-rookie preflight](#4-zero-rookie-preflight)
- [Part A — Ubuntu AMD GPU: ROCm + MIGraphX](#part-a--ubuntu-amd-gpu-rocm--migraphx)
  - [5. Hardware and OS gates](#5-hardware-and-os-gates)
  - [6. Install the matching ROCm track](#6-install-the-matching-rocm-track)
  - [7. Install MIGraphX and the ORT wheel](#7-install-migraphx-and-the-ort-wheel)
  - [8. Ubuntu Docker fast path](#8-ubuntu-docker-fast-path)
- [Part B — Windows AMD GPU](#part-b--windows-amd-gpu)
  - [9. Simplest Python path: DirectML](#9-simplest-python-path-directml)
  - [10. New Windows path: Windows ML + AMD MIGraphX](#10-new-windows-path-windows-ml--amd-migraphx)
- [Part C — Windows Ryzen AI NPU: Vitis AI](#part-c--windows-ryzen-ai-npu-vitis-ai)
  - [11. Supported scope](#11-supported-scope)
  - [12. Install Ryzen AI Software 1.7.1](#12-install-ryzen-ai-software-171)
  - [13. Vitis AI provider options by generation](#13-vitis-ai-provider-options-by-generation)
- [Part D — Ubuntu Ryzen AI NPU: Vitis AI](#part-d--ubuntu-ryzen-ai-npu-vitis-ai)
  - [14. Current Linux support gate](#14-current-linux-support-gate)
  - [15. Install the Ubuntu NPU driver and Ryzen AI](#15-install-the-ubuntu-npu-driver-and-ryzen-ai)
- [Part E — Vitis AI on AMD Adaptive SoCs](#part-e--vitis-ai-on-amd-adaptive-socs)
  - [16. Embedded Linux targets](#16-embedded-linux-targets)
- [Part F — One-click Python demo](#part-f--one-click-python-demo)
  - [17. Demo behavior](#17-demo-behavior)
  - [18. Minimal provider code](#18-minimal-provider-code)
  - [19. Verification flow](#19-verification-flow)
  - [20. Performance guidance](#20-performance-guidance)
  - [21. Troubleshooting](#21-troubleshooting)
  - [22. Production checklist](#22-production-checklist)
  - [23. References](#23-references)
- [Appendix: VSINPU is not an AMD provider](#appendix-vsinpu)

---

## The whole AMD picture

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#dbeafe","primaryTextColor":"#1e293b","primaryBorderColor":"#3b82f6","secondaryColor":"#e0f2fe","secondaryTextColor":"#1e293b","tertiaryColor":"#f1f5f9","tertiaryTextColor":"#1e293b","lineColor":"#94a3b8"},"themeCSS":".mindmap-node text{fill:#1e293b !important;} .mindmap-node span{color:#1e293b !important;}"}}%%
mindmap
  root((AMD and ONNX Runtime))
    GPU
      Windows DirectML
      Windows ML MIGraphX
      Ubuntu ROCm 7.14, gfx950 or gfx942
      Ubuntu ROCm 7.2.4 or 7.2.1
    Ryzen AI NPU
      Windows direct SDK
      Windows ML VitisAI
      Ubuntu, STX or KRK only
    Adaptive SoC
      Zynq UltraScale Plus
      Versal AI Core or Edge
    Prove it
      Profile shows target EP
      No CPU fallback
      Vitis assignment report
```

---

## 1. Choose a route

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart TD
    A["Start"] --> B{"Operating system?"}
    B -->|Windows| C{"Target device?"}
    B -->|Ubuntu| D{"Target device?"}

    C -->|"AMD GPU, quick Python test"| E["DirectML<br/>DmlExecutionProvider"]
    C -->|"AMD GPU, new Win 11 24H2+ app"| F["Windows ML<br/>MIGraphXExecutionProvider"]
    C -->|"Ryzen AI NPU"| G["Ryzen AI Software<br/>VitisAIExecutionProvider"]

    D -->|"AMD GPU in the ROCm matrix"| H["ROCm + MIGraphX<br/>MIGraphXExecutionProvider"]
    D -->|"Ryzen AI NPU, STX or KRK"| I["Ryzen AI for Linux<br/>VitisAIExecutionProvider"]
    D -->|"Zynq or Versal SoC"| J["Vitis AI target setup"]

    style A fill:#455a64,stroke:#cfd8dc,color:#ffffff
    style B fill:#e65100,stroke:#ffcc80,color:#ffffff
    style C fill:#e65100,stroke:#ffcc80,color:#ffffff
    style D fill:#e65100,stroke:#ffcc80,color:#ffffff
    style E fill:#0067b8,stroke:#8dc8f4,color:#ffffff
    style F fill:#5e35b1,stroke:#b388ff,color:#ffffff
    style G fill:#c62828,stroke:#ff8a80,color:#ffffff
    style H fill:#0067b8,stroke:#8dc8f4,color:#ffffff
    style I fill:#c62828,stroke:#ff8a80,color:#ffffff
    style J fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
```

| Scenario | Route | ONNX Runtime EP | Status |
|---|---|---|---|
| Windows, recent AMD GPU | DirectML for the simplest Python start; evaluate Windows ML for new apps | `DmlExecutionProvider` | Supported, sustained engineering; Windows ML is Microsoft's new direction |
| Windows 11 24H2+, supported AMD GPU | Dynamically acquire AMD MIGraphX through Windows ML | `MIGraphXExecutionProvider` | Available via catalog; `--windows-ml` supports this path |
| Ubuntu, AMD GPU in the ONNX matrix | ROCm + MIGraphX + AMD wheel | `MIGraphXExecutionProvider` | **Primary Linux GPU path**; exact GPU/ROCm/Python/wheel gates apply |
| Windows, Ryzen AI NPU | Ryzen AI Software 1.7.1; Windows ML is also catalog-available | `VitisAIExecutionProvider` | PHX/HPT/STX/KRK; `--windows-ml` is GPU-only, use the vendor env for NPU |
| Ubuntu 24.04, Ryzen AI NPU | Ryzen AI for Linux 1.7.1 | `VitisAIExecutionProvider` | **STX/KRK only, kernel >= 6.10, Python 3.12** |
| Linux, AMD/Xilinx Adaptive SoC | Vitis AI target image and runtime | `VitisAIExecutionProvider` | Embedded Linux path for Zynq and Versal |
| Native Windows ROCm Core SDK | Not a current ORT MIGraphX Python path | None | ROCm 7.14 grows Windows core support, but the validated MIGraphX/ORT stack stays Linux-only |

> [!IMPORTANT]
> `ROCMExecutionProvider` was **removed in ONNX Runtime 1.23**. ROCm 7.0 was the last AMD release carrying it — new projects must use `MIGraphXExecutionProvider`.

> [!NOTE]
> GPU and NPU are separate stacks. ROCm/MIGraphX or DirectML target the GPU; Vitis AI/Ryzen AI targets the XDNA NPU. Installing one never enables the other.

---

## 2. Fundamentals

ONNX Runtime assigns each graph node to the first capable EP in the `providers` list; `CPUExecutionProvider` is the usual fallback.

```python
providers = [
    "MIGraphXExecutionProvider",  # first choice
    "CPUExecutionProvider",      # fallback
]
```

| API or signal | Proves | Does not prove |
|---|---|---|
| `ort.get_available_providers()` | Which EPs the wheel can load | Node placement |
| `session.get_providers()` | Registered EPs and priority | Actual placement ratio |
| ORT verbose log | Init and node-placement detail | Hard to automate; format can change |
| `args.provider` on `*_kernel_time` events | Which EP ran those kernels | Utilization percentage |
| Vitis AI assignment report | CPU/NPU node counts and op types | GPU EP placement |
| Task Manager / `amd-smi` / `xrt-smi` | Device activity and driver visibility | Which ONNX node ran there |

| Item | AMD GPU | AMD Ryzen AI NPU |
|---|---|---|
| Hardware | RDNA/CDNA GPU | AMD XDNA NPU |
| Linux stack | ROCm + MIGraphX | XRT + `amdxdna` + Ryzen AI/Vitis AI |
| Windows stack | DirectML or Windows ML MIGraphX | Ryzen AI Software or Windows ML VitisAI |
| ORT EP | `MIGraphXExecutionProvider` / `DmlExecutionProvider` | `VitisAIExecutionProvider` |
| Typical precision | FP32, FP16, hardware-dependent BF16/INT8/FP8 | INT8, BF16 (model- and silicon-dependent) |
| First load | MIGraphX compile/tune can be slow | Vitis AI compile can take minutes |
| Cache | MIGraphX cache / compiled artifacts | Vitis AI cache or ORT EP Context |

---

## 3. Version and support matrix

### 3.1 Snapshot verified on 2026-07-17

| Component | Verified version | Notes |
|---|---:|---|
| Current ROCm Core SDK | 7.14.0 | Production release 2026-07-15; first after the TheRock versioning discontinuity |
| ROCm 7.14 validated ONNX stack | ORT 1.23.2 + MIGraphX 2.16 | Linux, Python 3.12, `gfx950`/`gfx942` only — not a drop-in for the 7.2.x recipes below |
| Audited AMD-hosted MIGraphX wheel route | ROCm 7.2.4 + ORT 1.23.2 | CPython 3.10/3.12; newest broad, reproducible route this guide enforces |
| Official ROCm ORT Docker | ROCm 7.2.4 + ORT 1.23 + PyTorch 2.10.0 | Newest `rocm/onnxruntime` tags, Ubuntu 22.04/24.04 |
| Consumer Radeon validation matrix | ROCm 7.2.1 + ORT 1.23.2 | Radeon/Ryzen pages update on a different schedule than core ROCm |
| Latest upstream ORT / PyPI MIGraphX package | 1.27.1 | Dated 2026-07-12; AMD has not published a matching ROCm row, so this guide keeps the audited route |
| Stable Ryzen AI Software | 1.7.1 | Windows + Ubuntu NPU; 1.8.0 beta is not for production |
| Minimum Ryzen AI Windows NPU driver | 32.0.203.280 | Compatibility floor for Ryzen AI EP 1.7 |
| PyPI ONNX Runtime DirectML | 1.24.4 | Current x64 wheel; Python >= 3.11 |
| DirectML operator library in ORT | DirectML 1.15.2, opset up to 20 | Sustained engineering, with some opset-20 exceptions |
| Python packaging | pip 26.1.2; NumPy pinned to 1.26.4 | AMD's Radeon 7.2.1 ORT wheel is documented as incompatible with NumPy 2.x |
| Latest PyPI Windows ML artifacts | `wasdk-*` 2.3.0 + `onnxruntime-windowsml` 1.27.1 | Independently serviced — do not hand-combine newest version numbers |
| Reproducible Windows ML Python recipe | `wasdk-*` 2.1.3 + `onnxruntime-windowsml` 1.24.6.202605042033 | Exact dependency published by the 2.1.3 wheel; keep this pair together |

> [!IMPORTANT]
> "Latest" is not a compatibility guarantee. ROCm, MIGraphX, and the ORT MIGraphX wheel must come from one vendor-validated release set. Windows ML's two `wasdk-*` packages and the Windows App Runtime must share a release line, and the exact ORT dependency the projection declares wins over any standalone "latest" ORT. Never install more than one `onnxruntime-*` distribution in one environment.

> [!NOTE]
> **Why Windows ML is pinned to 2.1.3:** the unpinned PyPI projection already resolves to 2.3.0, while Microsoft's public stable Windows App SDK download page still tops out at runtime 2.2.0. The 2.1.3 projection, its exact ORT dependency, and the 2.1.3 runtime are all still published and form one reproducible set — do not replace these pins with "latest".

### 3.2 Documentation skew

The generic ONNX Runtime Vitis AI page still describes Ryzen AI as Windows-only with Linux limited to Adaptive SoCs. Ryzen AI Software 1.7.1's own product documentation adds Ubuntu 24.04 NPU support for STX/KRK. Use the product-version docs for Ryzen AI PCs, and the Vitis AI target docs for Zynq/Versal.

### 3.3 Audited artifact fingerprints

> [!WARNING]
> The verifier enforces these SHA-256 values (downloaded from the stated Microsoft PyPI or AMD HTTPS source and rehashed 2026-07-17). A mismatch fails closed — it is not permission to bypass the check. Re-audit any new vendor artifact and update code/docs together.

| Artifact | SHA-256 |
|---|---|
| DirectML 1.24.4 CPython 3.12 x64 wheel | `f2ecb68b7b7b259d2ef3112ae760149f9b5a1e7c0fbb73d539da6250a648a614` |
| `DirectML.dll` inside that wheel | `b73972115320e906a49602f2027a3266622881b0d325ba685e0f165a9482a8d7` |
| AMD ROCm 7.2.1 MIGraphX 1.23.2 CPython 3.10 wheel | `07f485fbeb8fbd6a89fa42d24832b4e206057fca62654b0eb39eb1edf9d6e70a` |
| AMD ROCm 7.2.1 MIGraphX 1.23.2 CPython 3.12 wheel | `663bff4dc3f72582d69f12ad073eb5695dfb526d574376cc8e5b161c7d2f0f08` |
| MIGraphX provider SO inside both 7.2.1 wheels | `8079986332cdf12234635ed4f2b5abd1b49519f6592d6dfcd8afaf5000887b7b` |
| AMD ROCm 7.2.4 MIGraphX 1.23.2 CPython 3.10 wheel | `4886faab646a7ef12f33fb53f085208182fab8dac249ba199dc5d23f8bd128ec` |
| AMD ROCm 7.2.4 MIGraphX 1.23.2 CPython 3.12 wheel | `ee8edeb2ba6a8d99b3043b23e812423e6f10333b508e003fc77b0feda197449f` |
| MIGraphX provider SO inside both 7.2.4 wheels | `f3fb0b10996b2a2f94afc59edf6fab421bfa12842f09518339d1e0d8f3bd86c7` |
| AMD ROCm 7.14.0 MIGraphX 1.23.2 CPython 3.12 wheel | `67c32a5d8396c28da5efd3643c1ebcb55a03581aad089f7d99922ed5a51bc58b` |
| MIGraphX provider SO inside the 7.14.0 wheel | `447bb405de55dd7872a8e01a90405ff0f0397d5d562acc6f48711312971537c0` |

Windows ML is serviced dynamically, so the verifier instead requires certified catalog status, exact current MSIX `1.8.57.0`, the pinned Python distributions, and a valid Microsoft Authenticode signature on the Windows App Runtime installer.

---

## 4. Zero-rookie preflight

### 4.1 Windows: identify the GPU and NPU

```powershell
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, AdapterRAM

Get-PnpDevice -PresentOnly |
  Where-Object { $_.FriendlyName -match 'NPU|Neural|AMD' } |
  Format-Table -AutoSize

winver
```

Then check **Task Manager → Performance**: confirm the **GPU** name/driver/DX12 support, and confirm **NPU 0** appears for a correctly installed Ryzen AI driver. On an iGPU+dGPU machine, note the GPU numbering — DirectML `device_id=0` is not necessarily the fastest device.

### 4.2 Ubuntu: identify devices, OS, and permissions

```bash
cat /etc/os-release
uname -r
lspci -nnk | grep -EA3 'VGA|Display|3D|1022:17f0'
groups
ls -l /dev/kfd /dev/dri 2>/dev/null || true
```

After installing ROCm: `/opt/rocm/bin/rocminfo | grep -E 'Name:|Marketing Name:' | head -20` and `/opt/rocm/bin/amd-smi list`. After installing Ryzen AI NPU/XRT: `source /opt/xilinx/xrt/setup.sh && xrt-smi examine`.

| Observation | Meaning |
|---|---|
| `/dev/kfd` and `/dev/dri/renderD*` exist | Linux GPU compute device nodes exist |
| `rocminfo` shows a `gfx...` agent | ROCm sees the GPU; the official hardware matrix must still list it |
| `1022:17f0` + `xrt-smi` shows Strix/Krackan | Candidate for the Ryzen AI Linux NPU path |
| User not in `render,video` | Common `Permission denied` cause; log out or reboot after adding the groups |

---

## Part A — Ubuntu AMD GPU: ROCm + MIGraphX

## 5. Hardware and OS gates

AMD publishes three distinct ONNX Runtime tracks. The newest ROCm Core SDK is not automatically the right ONNX package for every GPU:

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart TD
    A["Ubuntu AMD GPU"] --> B{"Exact GPU model?"}
    B -->|"Instinct MI350X / MI355X / MI300X / MI325X"| C{"gfx950 or gfx942,<br/>confirmed by rocminfo?"}
    C -->|Yes| D["ROCm 7.14.0 track<br/>ORT 1.23.2 + MIGraphX 2.16<br/>Ubuntu 24.04, Python 3.12 only"]
    B -->|"Other Instinct: MI300A, MI200, MI100"| E["Not in the 7.14 ONNX row"]
    E --> F["Use only an explicit archived route<br/>or a validated source build"]
    B -->|"Radeon / Radeon PRO listed in AMD's Radeon ONNX matrix"| G["ROCm 7.2.1 track<br/>Radeon-focused, needs the exact HWE kernel"]
    B -->|"Retained archived Instinct matrix"| H["ROCm 7.2.4 track<br/>retained, not current production"]
    B -->|"Ryzen APU iGPU or unlisted GPU"| I["Stop — no ONNX row yet<br/>use DirectML or the Vitis AI NPU path"]

    classDef step fill:#e0f2fe,stroke:#0ea5e9,color:#0c2a3d;
    classDef dec fill:#fef3c7,stroke:#f59e0b,color:#713f12;
    classDef good fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef bad fill:#fee2e2,stroke:#ef4444,color:#7f1d1d;
    class A step;
    class B,C dec;
    class D,G,H good;
    class E,F,I bad;
```

Do not combine the driver, MIGraphX package, or wheel from different tracks.

| Family | Representative models | Mandatory check |
|---|---|---|
| Instinct `gfx950` / `gfx942` | MI355X, MI350X, MI325X, MI300X | Current ROCm 7.14 AI Ecosystem ONNX matrix; use the exact target `rocminfo` reports |
| Other Instinct | MI300A, MI200 family, MI100 | Not in the current 7.14 ONNX row; use only an explicitly supported archived route or a separately validated source build |
| Radeon PRO | AI PRO R9700/R9600D, W7900/W7800/W7700 families | Must appear in the Radeon-focused ONNX matrix — a core ROCm listing alone is not enough |
| Radeon RDNA4 | RX 9070/9060 families | Usually restricted to specific Ubuntu/RHEL releases |
| Radeon RDNA3 | RX 7900/7800/7700 families | Use only SKUs AMD explicitly lists |
| Unlisted GPU | Older Polaris/Vega/RDNA2 or another model | Might run, but is not officially supported; do not use for a production commitment |

> [!NOTE]
> An unlisted GPU appearing in `rocminfo` does not mean every prebuilt ROCm/MIGraphX library supports it — enumeration can succeed while a kernel launch later fails.
>
> **Ryzen APU iGPU:** ROCm 7.14 adds core GPU support for several `gfx115x` Ryzen APUs, but its AI Ecosystem ONNX row stays `gfx950/gfx942` only, and the Radeon 7.2.1 matrix does not cover Ryzen APUs either. On a Ryzen AI laptop use DirectML for the GPU and Vitis AI for the documented STX/KRK NPU path.

---

## 6. Install the matching ROCm track

| Before you install | Rule |
|---|---|
| Confirm the exact match | GPU SKU, LLVM target, OS point release, and kernel must all appear in the chosen track's matrix |
| Pick exactly one track | Use the flowchart in [§5](#5-hardware-and-os-gates) — never mix drivers/packages across tracks |
| Never overwrite an AMDGPU install | Run the matching AMD uninstall procedure first; Radeon Software for Linux has no in-place upgrade |
| Secure Boot enabled | Follow your org's DKMS module-signing policy — don't disable security controls just to run the demo |

> [!WARNING]
> Every route below installs or replaces GPU software and can require a reboot. Use only the route matching your exact hardware, release, and Ubuntu version.

### 6.1 Current ROCm 7.14.0 ONNX track — Ubuntu 24.04, `gfx950/gfx942` only

ROCm 7.14 uses the new TheRock packaging layout — do not adapt the older `amdgpu-install_7.2.x` commands below. Open AMD's current [ROCm install selector](https://rocm.docs.amd.com/en/latest/install/rocm.html), select your exact GPU and Ubuntu 24.04, and complete its driver/repository prerequisites. Then install exactly **one** architecture package:

```bash
# MI300X / MI325X only:
sudo apt install amdrocm7.14-gfx942

# OR MI350X / MI355X only (not both commands on a single-architecture host):
# sudo apt install amdrocm7.14-gfx950

sudo usermod -a -G render,video "$LOGNAME"
sudo reboot
```

Confirm the installed release and exact GPU target before continuing:

```bash
/opt/rocm/bin/hipconfig --version
/opt/rocm/bin/rocminfo | grep -E '^[[:space:]]*Name:[[:space:]]*gfx(942|950)$'
/opt/rocm/bin/amd-smi version
```

`rocminfo` must print the target matching your GPU. A different `gfx` target is not eligible for the 7.14 ONNX wheel even if core ROCm supports it.

### 6.2 Retained ROCm 7.2.4 track — Ubuntu 24.04

This older block is retained for AMD's release-matched 7.2.4 ORT artifacts. It is **not** the current ROCm release.

```bash
wget --https-only -O amdgpu-install_7.2.4.70204-1_all.deb \
  https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms

sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video "$LOGNAME"
sudo apt install rocm
sudo reboot
```

### 6.3 Retained ROCm 7.2.4 track — Ubuntu 22.04

```bash
wget --https-only -O amdgpu-install_7.2.4.70204-1_all.deb \
  https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/jammy/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms

sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video "$LOGNAME"
sudo apt install rocm
sudo reboot
```

### 6.4 Radeon-focused ONNX track — ROCm 7.2.1

The conservative, fully matrix-validated route for the discrete Radeon/Radeon PRO products on AMD's Radeon ONNX page. Install the HWE kernel the matrix requires, reboot, and verify the kernel before continuing.

**Ubuntu 24.04.4** (needs the 6.17 HWE line):

```bash
sudo apt update
sudo apt-get install --install-recommends linux-generic-hwe-24.04
sudo reboot
```

After reboot (`uname -r` must show 6.17 HWE):

```bash
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
wget --https-only -O amdgpu-install_7.2.1.70201-1_all.deb \
  https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video "$LOGNAME"
sudo reboot
```

**Ubuntu 22.04.5** (needs the 6.8 HWE line):

```bash
sudo apt update
sudo apt-get install --install-recommends linux-generic-hwe-22.04
sudo reboot
```

After reboot (`uname -r` must show 6.8 HWE):

```bash
sudo apt update
sudo apt install -y python3-setuptools python3-wheel
wget --https-only -O amdgpu-install_7.2.1.70201-1_all.deb \
  https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/jammy/amdgpu-install_7.2.1.70201-1_all.deb
sudo apt install ./amdgpu-install_7.2.1.70201-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video "$LOGNAME"
sudo reboot
```

Verify after reboot:

```bash
groups
/opt/rocm/bin/rocminfo | head -80
/opt/rocm/bin/amd-smi list
cat /opt/rocm/.info/version
```

| Expected result | Check |
|---|---|
| User groups | Belongs to `render` and `video` |
| GPU visible | `rocminfo` lists at least one GPU agent |
| GPU visible | `amd-smi list` shows the expected GPU |
| Version match | ROCm version matches the wheel repository you intend to use |

---

## 7. Install MIGraphX and the ORT wheel

### 7.1 MIGraphX runtime

For **ROCm 7.14**, install AMD's exact MIGraphX 2.16 packages:

```bash
wget --https-only \
  https://rocm.frameworks.amd.com/deb-multi-arch/amdrocm-migraphx/pool/main/amdrocm-migraphx_2.16.0-3.py312_amd64.deb
wget --https-only \
  https://rocm.frameworks.amd.com/deb-multi-arch/amdrocm-migraphx/pool/main/amdrocm-migraphx-dev_2.16.0-3.py312_amd64.deb
sudo apt install -y \
  ./amdrocm-migraphx_2.16.0-3.py312_amd64.deb \
  ./amdrocm-migraphx-dev_2.16.0-3.py312_amd64.deb

/opt/rocm/bin/migraphx-driver --version
/opt/rocm/bin/migraphx-driver perf --test
dpkg-query -W -f='${Package} ${Version}\n' amdrocm-migraphx amdrocm-migraphx-dev
```

For either **7.2.x** track, use the release repository package instead:

```bash
sudo apt update
sudo apt install -y migraphx

/opt/rocm/bin/migraphx-driver --version
/opt/rocm/bin/migraphx-driver perf --test
dpkg-query -W -f='${Package} ${Version}\n' migraphx half
```

`--test` runs a documented built-in single-layer GEMM model, so no model file is required. On 7.2.x, `half` should arrive as a dependency; install it manually if the last `dpkg-query` shows it missing. `migraphx-dev` is needed only for development/source builds on 7.2.x.

### 7.2 Create an isolated Python environment

Three release-matched `onnxruntime_migraphx-1.23.2` routes are used here: **7.14.0** is CPython 3.12-only and gated to `gfx950/gfx942`; the retained **7.2.4** and **Radeon 7.2.1** repositories carry CPython 3.10 and 3.12. Use Ubuntu's native Python — do not add an unofficial Python repository for this demo.

```bash
# Ubuntu 24.04
sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv .venv-amd-ort

# Ubuntu 22.04 (7.2.x tracks only)
sudo apt install -y python3.10 python3.10-venv
python3.10 -m venv .venv-amd-ort
```

Activate the environment and install from the source matching your installed ROCm release.

**Current ROCm 7.14.0** (`gfx950/gfx942`, Python 3.12 only):

```bash
source .venv-amd-ort/bin/activate
/opt/rocm/bin/hipconfig --version 2>&1 | grep -Eq '(^|[^0-9])7\.14(\.0)?([^0-9]|$)' || { echo "Installed ROCm is not 7.14.0" >&2; exit 1; }
python -m pip install --index-url https://pypi.org/simple "pip==26.1.2"
python -m pip install --index-url https://pypi.org/simple "numpy==1.26.4"
python -m pip install --index-url https://pypi.org/simple \
  "https://rocm.frameworks.amd.com/whl-multi-arch/onnxruntime-migraphx/onnxruntime_migraphx-1.23.2%2Brocm7.14.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

**Retained ROCm 7.2.4**:

```bash
source .venv-amd-ort/bin/activate
grep -Eq '(^|[^0-9])7\.2\.4([^0-9]|$)' /opt/rocm/.info/version || { echo "Installed ROCm is not 7.2.4" >&2; exit 1; }
python -m pip install --index-url https://pypi.org/simple "pip==26.1.2"
python -m pip install --index-url https://pypi.org/simple "numpy==1.26.4"
PYTAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
case "$PYTAG" in cp310|cp312) ;; *) echo "Unsupported Python ABI: $PYTAG" >&2; exit 1;; esac
python -m pip install --index-url https://pypi.org/simple \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/onnxruntime_migraphx-1.23.2-${PYTAG}-${PYTAG}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

**Radeon-focused ROCm 7.2.1** (same commands, 7.2.1 repository):

```bash
source .venv-amd-ort/bin/activate
grep -Eq '(^|[^0-9])7\.2\.1([^0-9]|$)' /opt/rocm/.info/version || { echo "Installed ROCm is not 7.2.1" >&2; exit 1; }
python -m pip install --index-url https://pypi.org/simple "pip==26.1.2"
python -m pip install --index-url https://pypi.org/simple "numpy==1.26.4"
PYTAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
case "$PYTAG" in cp310|cp312) ;; *) echo "Unsupported Python ABI: $PYTAG" >&2; exit 1;; esac
python -m pip install --index-url https://pypi.org/simple \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/onnxruntime_migraphx-1.23.2-${PYTAG}-${PYTAG}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
```

> [!NOTE]
> This is a freshly created, disposable venv. If `python -m pip list` already shows any `onnxruntime-*` package before you install, delete the venv and recreate it — don't uninstall packages in place.
>
> The direct AMD wheel URLs are deliberate: 7.14 comes from `rocm.frameworks.amd.com`, 7.2.x from their exact `repo.radeon.com` release directories. PyPI now hosts independently published same-named wheels (including 1.27.1) that AMD has not mapped to these tracks, so `--bootstrap` hash-verifies the selected wheel against [§3.3](#33-audited-artifact-fingerprints) before installing.
>
> NumPy is pinned to `1.26.4` because AMD's Radeon 7.2.1 ORT page documents an incompatibility with NumPy 2.x; this guide keeps one conservative baseline across all three ORT 1.23.2 routes.

Verify the wheel:

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected: `['MIGraphXExecutionProvider', 'CPUExecutionProvider']`

### 7.3 One-command GPU run

```bash
python AMD/provider_test.py --target migraphx --strict-all
```

If the wheel is not installed, let the script install one matching your installed ROCm release:

```bash
python AMD/provider_test.py --target migraphx --bootstrap --strict-all
```

> [!NOTE]
> `--bootstrap` never installs a kernel driver — it only manages Python packages in the active environment. It requires an activated venv or non-base Conda env, refuses to touch Ryzen AI/Windows ML vendor environments, checks the x86-64 + release-specific Python ABI, verifies MIGraphX and the installed ROCm release (accepting only 7.2.1, 7.2.4, or 7.14.0), and never uninstalls an existing ORT. The 7.14 path additionally reads `rocminfo` and rejects any `--device-id` that isn't `gfx942/gfx950` before downloading.

---

## 8. Ubuntu Docker fast path

Host prerequisites: the AMD kernel driver, `/dev/kfd`, `/dev/dri`, Docker Engine, and correct user permissions. The container carries ROCm user-space libraries, MIGraphX, and ORT.

> [!WARNING]
> As of 2026-07-17 AMD has not published a ROCm 7.14 `rocm/onnxruntime` image — the newest official tags remain 7.2.4. This shortcut covers only the 7.2.x tracks; use [§6.1](#61-current-rocm-7140-onnx-track--ubuntu-2404-gfx950gfx942-only) and [§7](#7-install-migraphx-and-the-ort-wheel) for 7.14.

```bash
# ROCm core 7.2.4, Ubuntu 24.04:
IMAGE=rocm/onnxruntime:rocm7.2.4_ub24.04_ort1.23_torch2.10.0
# Radeon-focused ROCm 7.2.1, Ubuntu 24.04 (use this instead on that track):
# IMAGE=rocm/onnxruntime:rocm7.2.1_ub24.04_ort1.23_torch2.9.1

docker pull "$IMAGE"

docker run --rm -it \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python3 AMD/provider_test.py --target migraphx --strict-all
```

Ubuntu 22.04 tags: `rocm7.2.4_ub22.04_ort1.23_torch2.10.0` (core) and `rocm7.2.1_ub22.04_ort1.23_torch2.9.1` (Radeon). Verify inside the container with `rocminfo` and `/opt/rocm/bin/amd-smi list`.

---

## Part B — Windows AMD GPU

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart TD
    A["Windows AMD GPU"] --> B{"What matters most?"}
    B -->|"Fastest Python bring-up, any Win 10/11"| C["DirectML<br/>onnxruntime-directml"]
    B -->|"New Win 11 24H2+ app,<br/>managed EP updates"| D["Windows ML<br/>AMD MIGraphX plugin"]
    C --> E["DmlExecutionProvider"]
    D --> F["MIGraphXExecutionProvider"]

    style A fill:#455a64,stroke:#cfd8dc,color:#ffffff
    style B fill:#e65100,stroke:#ffcc80,color:#ffffff
    style C fill:#0067b8,stroke:#8dc8f4,color:#ffffff
    style D fill:#5e35b1,stroke:#b388ff,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b;
    class E,F leaf;
```

## 9. Simplest Python path: DirectML

| Requirement | Minimum / guidance |
|---|---|
| OS | Introduced in Windows 10 1903; Windows 11 recommended |
| GPU | DirectX 12 capable; broadly supports AMD GCN 1st Gen and newer |
| Driver | Latest stable AMD Adrenalin/PRO driver |
| Python | x64 Python 3.12 from python.org or winget — not Microsoft Store Python |
| Package | `onnxruntime-directml==1.24.4` for this verified snapshot |

```powershell
winget install --id Python.Python.3.12 -e `
  --accept-package-agreements --accept-source-agreements
```

If this installed Python for the first time, close every PowerShell window and reopen one. Continue only when both checks below succeed and report AMD64/x86-64:

```powershell
py -3.12 --version
py -3.12 -c "import platform; print(platform.machine())"
py -3.12 -m venv .venv-amd-dml
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv-amd-dml\Scripts\Activate.ps1

python -m pip install --index-url https://pypi.org/simple "pip==26.1.2"
python -m pip install --index-url https://pypi.org/simple "numpy==1.26.4" "onnxruntime-directml==1.24.4"

python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected: `['DmlExecutionProvider', 'CPUExecutionProvider']`. `numpy==1.26.4` is a reproducibility pin shared with the audited AMD wheel path, not a DirectML hardware requirement.

```powershell
python AMD/provider_test.py --target dml --strict-all

# On a multi-GPU machine:
python AMD/provider_test.py --target dml --device-id 1 --strict-all
```

Run from the repository root. The demo enumerates DXGI adapters in the same order DirectML uses, and fails unless the selected `--device-id` has AMD PCI vendor ID `0x1002`.

Required session settings (DirectML does not support ORT parallel execution or memory-pattern optimization — use separate sessions for concurrency):

```python
options.enable_mem_pattern = False
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
```

| Limitation | Detail |
|---|---|
| Engineering status | Sustained engineering; new Windows development is moving to Windows ML |
| Opset ceiling | DirectML 1.15.2 supports up to opset 20, except unsupported configs like 5-D GridSample 20 and DeformConv |
| Shapes | Static input shapes generally improve folding, weight preprocessing, and scheduling |
| Adapter choice | `device_id=0` is the default DXGI adapter, not necessarily the fastest one |

---

## 10. New Windows path: Windows ML + AMD MIGraphX

Use this for a new Windows 11 24H2+ application that benefits from system-managed EP downloads and updates.

| Item | Requirement |
|---|---|
| OS | Windows 11 24H2, build 26100+ for dynamically acquired hardware EPs |
| Python | x64 Python 3.12 (this audited recipe); pinned ORT requires Python >= 3.11 — not Microsoft Store Python |
| Runtime | Windows App SDK Runtime matching the Python `wasdk-*` packages |
| AMD MIGraphX plugin | Acquired through the Windows ML EP Catalog |
| AMD VitisAI plugin | Requires a Ryzen AI NPU driver — see [Part C](#part-c--windows-ryzen-ai-npu-vitis-ai) |

```powershell
winget install --id Python.Python.3.12 -e `
  --accept-package-agreements --accept-source-agreements
```

If winget installed Python for the first time, **close every PowerShell window and reopen one now**. Continue only after both report x64/AMD64 Python 3.12:

```powershell
py -3.12 --version
py -3.12 -c "import platform; print(platform.machine())"

py -3.12 -m venv .venv-winml
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv-winml\Scripts\Activate.ps1

python -m pip install --index-url https://pypi.org/simple "pip==26.1.2"
python -m pip install --index-url https://pypi.org/simple `
  "numpy==1.26.4" `
  "wasdk-Microsoft.Windows.AI.MachineLearning[all]==2.1.3" `
  "wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap==2.1.3" `
  "onnxruntime-windowsml==1.24.6.202605042033"

winget install --id "Microsoft.VCRedist.2015+.x64" -e `
  --accept-package-agreements --accept-source-agreements

$runtimeInstaller = "$env:TEMP\windowsappruntimeinstall-2.1.3-x64.exe"
Invoke-WebRequest `
  https://aka.ms/windowsappsdk/2.1/2.1.3/windowsappruntimeinstall-x64.exe `
  -OutFile $runtimeInstaller

$signature = Get-AuthenticodeSignature -LiteralPath $runtimeInstaller
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
  Remove-Item -LiteralPath $runtimeInstaller -Force -ErrorAction SilentlyContinue
  throw "Windows App Runtime installer signature is not a valid Microsoft signature."
}

try {
  $process = Start-Process $runtimeInstaller -ArgumentList "--quiet" -Wait -PassThru
  if ($process.ExitCode -ne 0) {
    throw "Windows App Runtime installer failed: 0x$('{0:X8}' -f $process.ExitCode)"
  }
} finally {
  Remove-Item -LiteralPath $runtimeInstaller -Force -ErrorAction SilentlyContinue
}
```

Verify before running (both `wasdk-*` at `2.1.3`, ORT at `1.24.6.202605042033`; stop and recreate the venv on any mismatch):

```powershell
python -m pip list | findstr /i "wasdk onnxruntime-windowsml winrt-runtime"
```

Then run the verifier from the repository root:

```powershell
python AMD/provider_test.py `
  --target migraphx --windows-ml --strict-all
```

The script keeps the Windows App Runtime bootstrap context alive, calls `ensure_ready_async().get()`, registers the downloaded plugin with `ort.register_execution_provider_library()`, selects its `OrtEpDevice`, and creates the session in the **same Python process**.

### 10.1 How Python acquires the EP

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","actorBkg":"#e0f2fe","actorBorder":"#0ea5e9","actorTextColor":"#0c2a3d","actorLineColor":"#94a3b8","signalColor":"#475569","signalTextColor":"#1e293b","noteBkgColor":"#fef3c7","noteTextColor":"#713f12","noteBorderColor":"#f59e0b"}}}%%
sequenceDiagram
    participant App as Python app
    participant Boot as Windows App Runtime
    participant Cat as EP Catalog
    participant Env as ORT environment
    participant Sess as Session

    App->>Boot: Initialize and keep alive
    App->>Cat: find_all_providers()
    Cat-->>App: Candidate EPs (MIGraphX, VitisAI, DML...)
    App->>Cat: ensure_ready_async().get()
    Cat-->>App: Downloaded plugin + library_path
    App->>Env: register_execution_provider_library(library_path)
    App->>Env: get_ep_devices()
    Env-->>App: OrtEpDevice list
    App->>Sess: add_provider_for_devices(selected device)
    App->>Sess: Create session in this same process
```

> [!WARNING]
> Do **not** call `EnsureAndRegisterCertifiedAsync()` and assume it registers providers into Python's ORT environment — that pattern skips the steps above. Do not copy a fixed plugin DLL path either; Windows ML owns and updates it.

| AMD device | EP name |
|---|---|
| AMD GPU | `MIGraphXExecutionProvider` |
| AMD Ryzen AI NPU | `VitisAIExecutionProvider` |
| Generic DX12 GPU fallback | `DmlExecutionProvider` |

| Plugin | Current catalog release | Driver gate |
|---|---|---|
| MIGraphX | MSIX 1.8.57.0 / GPU EP 7.2.2606.20 | AMD GPU driver **25.10.13.09 exactly**; not currently supported for GenAI scenarios |
| VitisAI | MSIX 1.8.63.0 / EP 2858 | Min Adrenalin 25.6.3 + NPU 32.00.0203.280; max Adrenalin 25.9.1 + NPU 32.00.0203.297 |

> [!WARNING]
> These catalog values change through Windows Update D-week releases — recheck the live table before install or image freeze. A newer driver number is **not automatically compatible**.
>
> **Do not mix the two NPU tracks.** AMD's direct Ryzen AI 1.7.1 page links NPU drivers `32.0.203.280` and `32.0.203.314`, but the Windows ML VitisAI catalog caps at `32.00.0203.297`. Driver `.314` is valid for the direct 1.7.1 SDK but outside the Windows ML VitisAI gate. The NPU commands in this guide use the direct Ryzen AI environment, not `--windows-ml`.

### 10.2 Why native Windows ROCm is not this path

ROCm 7.14 substantially expands Windows Core SDK support, but AMD's MIGraphX 2.16 and ONNX Runtime 1.23.2 AI Ecosystem pages still validate only Linux x86-64 on `gfx950/gfx942`, and the AMD ORT wheel is a manylinux artifact. Native Windows ROCm therefore does not make `MIGraphXExecutionProvider` appear in a normal Windows ORT Python wheel. Current Windows choices remain: DirectML, Windows ML (for the MIGraphX plugin), or native Ubuntu ROCm/MIGraphX.

> [!WARNING]
> **WSL2 is not a MIGraphX route.** AMD's current ROCDXG WSL guide (Adrenalin 26.2.2 + ROCm 7.2.1) explicitly states MIGraphX is **not supported** on WSL. An older, now-legacy 7.2 compatibility page listed ONNX Runtime 1.23.2, but it does not override this limitation — the verifier rejects MIGraphX on a WSL kernel. Use native Ubuntu, native Windows DirectML, or Windows ML MIGraphX instead. Ryzen AI 1.7.1 NPU docs likewise cover native Windows and native Ubuntu 24.04 STX/KRK only, not WSL passthrough.

---

## Part C — Windows Ryzen AI NPU: Vitis AI

## 11. Supported scope

Ryzen AI Software 1.7 supports Phoenix (PHX), Hawk Point (HPT), Strix/Strix Halo (STX), and Krackan Point (KRK).

| Model type | PHX/HPT | STX/KRK |
|---|---:|---:|
| CNN INT8 | Yes | Yes |
| CNN BF16 | No | Yes |
| NLP/encoder BF16 | No | Yes |
| LLM through ONNX Runtime GenAI | No | Yes |

Recommended opset: **17**. Unsupported nodes auto-partition to CPU unless strict placement is requested and verified.

## 12. Install Ryzen AI Software 1.7.1

| Dependency | Requirement |
|---|---|
| Windows | Build >= 22621.3527 for the direct 1.7.1 stack |
| NPU driver | 32.0.203.280+; still verify against the exact EP release |
| Visual Studio | VS 2022 + Desktop Development with C++ for builds/custom ops; optional for the basic quicktest |
| CMake | >= 3.26 |
| Environment manager | Miniforge preferred |
| Supported NPU | Confirm in release notes, not by processor marketing name alone |

0. If Miniforge is missing, install the official `Miniforge3-Windows-x86_64.exe` to a path without spaces/special characters, create the **Miniforge Prompt** shortcut, add only that install's `condabin` to the **System** `PATH`, then confirm `where.exe conda` and `conda --version` in a fresh terminal.
1. Install CMake and verify:

```powershell
winget install --id Kitware.CMake -e --accept-package-agreements --accept-source-agreements
```

```powershell
cmake --version   # expect >= 3.26 after reopening Miniforge Prompt
```

2. Download the production NPU driver from the official Ryzen AI page, extract it, then from an **Administrator** terminal:

```powershell
.\npu_sw_installer.exe
```

3. Reboot if requested; confirm **Task Manager → Performance → NPU 0**. Use a linked production driver (`32.0.203.280` or `32.0.203.314`) — do not combine the Ryzen AI 1.8 beta driver with this 1.7.1 stack.
4. Download and run `ryzen-ai-lt-1.7.1.exe`, keep the default path, and let it create the Conda environment `ryzen-ai-1.7.1`.

### 12.1 Vendor quicktest (STX/KRK)

Open the **Miniforge Prompt** (Command Prompt shortcut, not PowerShell):

```bat
conda activate ryzen-ai-1.7.1
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
cd /d "%RYZEN_AI_INSTALLATION_PATH%\quicktest"
python quicktest.py
```

Expected final line: `Test Finished`. Stop if `VitisAIExecutionProvider` is absent — never repair the vendor environment with pip.

> [!NOTE]
> **PHX/HPT:** do not use an unmodified `quicktest.py`. AMD requires `target=X1`, `xlnx_enable_py3_round=0`, and the Phoenix `4x4.xclbin`. Skip to [§12.2](#122-one-command-proof-with-profiling) — the repository verifier applies those options without touching the vendor file.

### 12.2 One-command proof with profiling

```powershell
python AMD/provider_test.py --target npu --strict-all
```

The script locates the vendor `quicktest/test_model.onnx`, detects PHX/HPT vs STX/KRK, creates the correct Vitis AI options, runs inference, and rejects a zero-NPU-node result.

> [!WARNING]
> Never run `pip install onnxruntime` inside the Ryzen AI environment — a generic CPU wheel can overwrite the vendor ORT files and remove `VitisAIExecutionProvider`.

## 13. Vitis AI provider options by generation

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart TD
    A["Ryzen AI NPU model"] --> B{"Generation?"}
    B -->|STX or KRK| C["target = X2 default<br/>no xclbin option"]
    B -->|PHX or HPT| D["target = X1 required<br/>xclbin = phoenix 4x4.xclbin"]
    C --> E["Current INT8 compiler + BF16 available"]
    D --> F["CNN INT8 only"]

    classDef step fill:#e0f2fe,stroke:#0ea5e9,color:#0c2a3d;
    classDef dec fill:#fef3c7,stroke:#f59e0b,color:#713f12;
    classDef good fill:#dcfce7,stroke:#22c55e,color:#14532d;
    class A step;
    class B dec;
    class C,D,E,F good;
```

| Device | INT8 `target` | `xclbin` | Notes |
|---|---|---|---|
| STX/KRK and newer | `X2` by default; `X1` testable for specific models | Must **not** be set for the normal X2 flow | Current INT8 compiler; BF16 available |
| PHX/HPT | `X1` required | `...\xclbins\phoenix\4x4.xclbin` required | CNN INT8 only |

All keys below are read either directly by the open-source `VitisAIExecutionProvider` (the three
`ep_context_*` rows), or forwarded as-is to AMD's closed-source Vitis AI compiler (`vaip`) that the EP loads
at runtime — see [`vitisai_provider_factory.cc`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/vitisai/vitisai_provider_factory.cc) and [`vitisai_execution_provider.cc`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/vitisai/vitisai_execution_provider.cc) in the ONNX Runtime source.

| Option | Values | Meaning |
|---|---|---|
| `target` | `X1` / `X2` | Compiler target generation: `X2` for STX/KRK (default there), `X1` required for PHX/HPT |
| `xclbin` | path | PHX/HPT only: absolute path to the generation's `.xclbin` overlay; must be omitted for the X2 flow |
| `cache_dir` | path | Directory for the Vitis AI compiler cache; reusing it across runs skips recompilation |
| `cache_key` | string | Cache namespace/id; change it whenever the model or these options change |
| `enable_cache_file_io_in_mem` | `"0"` / `"1"` | `"0"` (recommended) writes the cache to disk under `cache_dir` so it can be inspected; `"1"` keeps it in memory only |
| `config_file` | path | JSON file controlling BF16 `optimize_level` and preferred data layout (STX/KRK BF16 flow) |
| `ep_context_enable` | `"0"` / `"1"` | Emit an ONNX Runtime **EPContext** model that embeds the pre-compiled Vitis AI graph, so the next load skips recompilation |
| `ep_context_embed_mode` | `"0"` / `"1"` | With EPContext enabled: `"1"` embeds the compiled binary directly inside the generated `.onnx`; omitted or `"0"` (default) keeps it as a separate sibling file |
| `ep_context_file_path` | path | Custom output path for the generated EPContext model; defaults to alongside the source model |
| `external_ep_library` | path | Advanced/internal: delegates VitisAI EP construction to another EP's factory library; not needed for normal model deployment |

> [!NOTE]
> `ep_context_*` are the same generic EPContext cache keys shared by several ORT execution providers
> (TensorRT, OpenVINO, QNN...). Every other key in this options dict is passed through unchanged to AMD's
> `vaip` compiler, so a newer Ryzen AI release may document additional model- or generation-specific tuning
> keys beyond this table — check the release notes for the Ryzen AI Software version you installed.

```python
import onnxruntime as ort

options = {
    # --- Compiler target: see the table above for STX/KRK vs PHX/HPT ---
    "target": "X2",                        # "X2" (STX/KRK, default) or "X1" (PHX/HPT, required there)
    # "xclbin": r"C:\...\xclbins\phoenix\4x4.xclbin",  # PHX/HPT only; omit entirely on the X2 flow

    # --- Compiler cache: skip recompilation on repeat runs of the same model ---
    "cache_dir": r"C:\temp\my-vitis-cache",
    "cache_key": "my-model-v1",            # bump this whenever the model or these options change
    "enable_cache_file_io_in_mem": "0",    # "0" = cache written to disk (inspectable); "1" = memory-only

    # --- Optional: BF16 compile tuning (STX/KRK only) ---
    # "config_file": r"C:\path\to\bf16_config.json",

    # --- Optional: EPContext cache (compile once, reload fast next time) ---
    # "ep_context_enable": "1",
    # "ep_context_embed_mode": "0",        # "1" embeds the compiled blob in the .onnx; "0" keeps a sibling file
    # "ep_context_file_path": r"C:\path\to\model_ctx.onnx",
}

session = ort.InferenceSession(
    "model_int8.onnx",
    providers=[
        ("VitisAIExecutionProvider", options),
        "CPUExecutionProvider",
    ],
)
```

| BF16 and production notes | Detail |
|---|---|
| Entry path | FP32 CNN/Transformer models can enter BF16 compilation on supported STX/KRK devices |
| Deployment | AMD recommends precompiled BF16 models for C++; not every on-the-fly BF16 scenario is supported |
| First compile | Can take minutes — use the Vitis AI cache in development, ORT EP Context for packaging |
| Cache hygiene | Delete or re-key caches after changing the Vitis AI EP or NPU driver; caches are not portable across versions |

Generate an assignment report:

```powershell
$env:XLNX_ONNX_EP_REPORT_FILE = "vitisai_ep_report.json"
python your_inference.py
```

The report's `deviceStat` section shows `CPU`/`NPU` node counts. Set `enable_cache_file_io_in_mem=0` and inspect the configured cache directory.

---

## Part D — Ubuntu Ryzen AI NPU: Vitis AI

## 14. Current Linux support gate

Ryzen AI 1.7.1 is the first product documentation in this guide to explicitly support Ryzen NPU inference on Linux.

| Requirement | Current 1.7.1 value |
|---|---|
| Supported NPU families | STX and KRK |
| Distribution | Ubuntu 24.04 LTS |
| Kernel | >= 6.10 |
| Python | 3.12.x |
| RAM | 64 GB recommended |
| Models | CNN INT8/BF16, encoder NLP BF16, NPU-only LLM flow |
| EP | `VitisAIExecutionProvider` |

> [!NOTE]
> PHX/HPT are **not** listed in the current Linux support statement. Do not infer Linux support from the Windows matrix.

## 15. Install the Ubuntu NPU driver and Ryzen AI

### 15.1 Base packages

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y python3.12 python3.12-venv libboost-filesystem1.74.0 pciutils
uname -r
```

`libboost-filesystem1.74.0` ships in Ubuntu 24.04's `universe` component, enabled above. If the GA kernel is below 6.10, move to a supported HWE/OEM kernel and reboot before installing XRT:

```bash
sudo apt-get update
sudo apt-get install --install-recommends linux-generic-hwe-24.04
sudo reboot
```

After reboot, `uname -r` must show >= 6.10. If `ubuntu-drivers list-oem` shows an OEM kernel track, keep that cadence and follow vendor/Ubuntu documentation instead of switching tracks.

### 15.2 Download and install XRT/NPU packages

Get `RAI_1.7.1_Linux_NPU_XRT.zip` from the official AMD Ryzen AI download page, extract it, then from that directory:

```bash
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-base.deb
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-base-dev.deb
sudo apt install --fix-broken -y ./xrt_202610.2.21.75_24.04-amd64-npu.deb
sudo apt install --fix-broken -y ./xrt_plugin.2.21.260102.53.release_24.04-amd64-amdxdna.deb

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

Expected device name resembles `NPU Strix` (exact BDF/name vary by machine).

### 15.3 Install the Ryzen AI 1.7.1 package

```bash
mkdir -p ryzen_ai-1.7.1
cp ryzen_ai-1.7.1.tgz ryzen_ai-1.7.1/
cd ryzen_ai-1.7.1
tar -xvzf ryzen_ai-1.7.1.tgz

./install_ryzen_ai.sh -a yes -p "$HOME/ryzen-ai-1.7.1/venv"
source "$HOME/ryzen-ai-1.7.1/venv/bin/activate"
echo "$RYZEN_AI_INSTALLATION_PATH"
python -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Linux uses the installer-created venv — ignore Windows-only Conda steps. Stop if `VitisAIExecutionProvider` is absent; never install a generic ORT wheel here.

### 15.4 Quicktest and one-command proof

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
source /opt/xilinx/xrt/setup.sh
source "$HOME/ryzen-ai-1.7.1/venv/bin/activate"
cd "$HOME/ryzen-ai-1.7.1/venv/quicktest"
python quicktest.py

# Replace with the absolute path to this repository.
REPO_ROOT="/absolute/path/to/Tutorial-ONNX-Runtime-Execution-Providers-main"
cd "$REPO_ROOT"
python AMD/provider_test.py --target npu --strict-all
```

If the installation path differs, activate that environment and pass the model explicitly:

```bash
source /opt/xilinx/xrt/setup.sh
python AMD/provider_test.py \
  --target npu \
  --model /your/ryzen-ai/venv/quicktest/test_model.onnx \
  --strict-all
```

---

## Part E — Vitis AI on AMD Adaptive SoCs

## 16. Embedded Linux targets

| Host ISA | Vitis AI target | Example boards | OS |
|---|---|---|---|
| Arm Cortex-A53 | Zynq UltraScale+ MPSoC | ZCU102, ZCU104, KV260 | Linux |
| Arm Cortex-A72 | Versal AI Core/Premium | VCK190 | Linux |
| Arm Cortex-A72 | Versal AI Edge | VEK280 | Linux |

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart LR
    A["1 Board target image / BSP"] --> B["2 Vitis AI Target Setup:<br/>firmware, XRT, DPU overlay, runtime"]
    B --> C["3 Vitis AI ONNX Runtime EP"]
    C --> D["4 Quantize with AMD Quark<br/>or Vitis AI Quantizer"]
    D --> E["5 Session with VitisAIExecutionProvider<br/>+ CPU fallback"]
    E --> F["6 Verify with Vitis AI logs,<br/>reports, device tools"]

    classDef step fill:#e0f2fe,stroke:#0ea5e9,color:#0c2a3d;
    class A,B,C,D,E,F step;
```

> [!WARNING]
> Do not use the x86-64 Ryzen AI installer or the ROCm MIGraphX wheel on these Arm targets. The generic ONNX Runtime build page confirms Linux `--use_vitisai` support for AMD Adaptive SoCs through this target workflow.

---

## Part F — One-click Python demo

## 17. Demo behavior

File: [provider_test.py](provider_test.py)

| Feature | Behavior |
|---|---|
| `--target auto` | Priority: Vitis AI NPU → MIGraphX GPU → DirectML GPU |
| `--target gpu` | Linux selects MIGraphX; a normal Windows pip environment selects DirectML |
| `--windows-ml` | Windows only: bootstraps the runtime, verifies pinned Python distributions and current MIGraphX MSIX 1.8.57.0, acquires/registers the plugin, selects its AMD `OrtEpDevice` in the same process |
| `--target npu` | Requires the vendor-installed Vitis AI EP; never replaces it with a public wheel |
| `--bootstrap` | Requires a clean isolated environment; pins and hash-verifies the DirectML wheel or the exact AMD ROCm 7.2.1/7.2.4/7.14.0 wheel; enforces the 7.14 `gfx942/gfx950` gate; never installs drivers or uninstalls an existing ORT |
| Runtime provenance | Rechecks the installed distribution version and provider DLL/SO hash; Windows ML rechecks all three pinned versions |
| Default GPU model | Integrity-checked embedded opset-17 Conv → Relu → GlobalAveragePool model; no separate `onnx` package needed |
| Default NPU model | The Ryzen AI vendor quicktest model, known NPU-compatible |
| Output sanity | Every result must have nonempty tensors; floats/complex must be finite; object/sequence/map outputs fail closed |
| Numerical check | Always compares the built-in GPU model with CPU EP; `--compare-cpu` enables the same for a user model (`--rtol`/`--atol` configurable) |
| Verification | Counts current-run ORT `*_kernel_time` `Node` events with provider attribution; Vitis may use a fresh assignment report + successful inference when attribution is absent |
| Failure policy | EP loaded but no target profile event (and no fresh Vitis evidence) → nonzero exit |
| `--strict-all` | Sets `session.disable_cpu_ep_fallback=1` before session creation, then independently rejects CPU events/nodes |
| Evidence isolation | Every invocation uses a new artifact/cache directory, so stale reports or caches cannot produce a false pass |
| `--unit-tests` | Runs the built-in deterministic safety/unit suite without AMD hardware, then exits |
| WSL | Rejects MIGraphX — AMD's current WSL guide marks it unsupported |
| Scope limit | Ryzen AI PC NPU only; rejects Arm Zynq/Versal Adaptive SoCs, which need board-specific models/options |

### 17.1 Command table

| Platform | Command |
|---|---|
| Windows AMD GPU, DirectML | `python AMD/provider_test.py --target dml --bootstrap --strict-all` |
| Windows AMD GPU, Windows ML MIGraphX | `python AMD/provider_test.py --target migraphx --windows-ml --strict-all` |
| Ubuntu AMD GPU | `python AMD/provider_test.py --target migraphx --bootstrap --strict-all` (auto-detects 7.2.1, 7.2.4, or gated 7.14.0) |
| Windows Ryzen AI NPU | `python AMD/provider_test.py --target npu --strict-all` |
| Ubuntu Ryzen AI NPU | `python AMD/provider_test.py --target npu --strict-all` |
| Existing custom model | Add `--model path/to/model.onnx` |
| Custom model + CPU parity | Add `--compare-cpu`; set `--rtol`/`--atol` if reduced precision is expected |
| Dynamic input | Add `--shape input_name=1,3,224,224` |
| Select the second GPU | Add `--device-id 1` — DirectML indexes DXGI adapters, Windows ML indexes AMD `OrtEpDevice` objects, Linux MIGraphX follows `rocminfo` GPU-agent order |
| Allow partial CPU fallback | Omit `--strict-all`; at least one accelerator node is still required |
| Script-only CPU self-test | `python AMD/provider_test.py --target cpu` |
| Built-in unit tests | `python AMD/provider_test.py --unit-tests` |

A profile-backed accelerator pass ends with:

```text
[PASS/通过] Runtime profile verified ... executed node event(s) on ...
```

> [!NOTE]
> On a Vitis build without provider attribution, success is reported as "inference succeeded + fresh assignment report shows NPU nodes." Assignment counts are unique graph nodes; profile counts are repeated execution events — never compare them as percentages. A provider that only appears in `get_available_providers()`, with neither a profiled event nor fresh Vitis evidence, is treated as a failure by design.
>
> `--strict-all` first asks ORT to reject CPU placement at session creation, then independently rejects every CPU event/node the profile or Vitis report exposes. It cannot prove facts no evidence channel reports. A hardware-placement PASS is not an accuracy certification — use trusted test vectors or `--compare-cpu` before production.

Run evidence is stored under `~/.cache/amd-ort-oneclick/runs/` (Linux) or the equivalent Windows home directory, or under `AMD_ORT_DEMO_CACHE` when set. The Vitis cache is intentionally fresh per invocation, so an NPU check can take minutes every time — this favors trustworthy proof over benchmark convenience.

```bash
# Inspect, then remove Linux run directories older than 7 days:
find ~/.cache/amd-ort-oneclick/runs -mindepth 1 -maxdepth 1 -type d -mtime +7 -print
# After reviewing the printed paths, repeat with -exec rm -rf -- {} +
```

### 17.2 Run your own model

```bash
python AMD/provider_test.py \
  --target migraphx \
  --model /absolute/path/model.onnx \
  --shape images=1,3,224,224 \
  --compare-cpu
```

The generic input generator supports common numeric/Boolean tensors. Every dynamic input needs an explicit, rank-correct `--shape`; unknown input names and fixed-dimension changes are rejected before inference. Floating inputs use deterministic values; integer/Boolean inputs use zeros. Models needing token semantics, nonzero lengths, correlated inputs, strings, custom operators, calibration data, or domain accuracy metrics need a model-specific runner — the EP verification logic can still be reused.

---

## 18. Minimal provider code

### 18.1 Linux MIGraphX GPU

All keys below come straight from ONNX Runtime's [`migraphx_execution_provider_info.h`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/migraphx/migraphx_execution_provider_info.h) — every value in the options dict is a **string** (booleans are `"0"`/`"1"`), matching how the C++ parser reads `ProviderOptions`.

```python
import onnxruntime as ort

options = {
    # --- Device selection ---
    "device_id": "0",                       # ROCm GPU index (default "0"); validated against hipGetDeviceCount()

    # --- Reduced-precision compute: measure accuracy against CPU before shipping any of these ---
    "migraphx_fp16_enable": "0",             # "1" = convert eligible ops to FP16 (default "0")
    "migraphx_bf16_enable": "0",             # "1" = convert eligible ops to BF16 (default "0")
    "migraphx_fp8_enable": "0",              # "1" = convert eligible ops to FP8; hardware/model dependent (default "0")
    "migraphx_int8_enable": "0",             # "1" = enable INT8 (default "0"); pair with the two keys below unless the model is pre-quantized
    "migraphx_int8_calibration_table_name": "",          # Path/name of an INT8 calibration table file (used only when int8_enable="1")
    "migraphx_int8_use_native_calibration_table": "0",   # "1" = use MIGraphX's own calibration table format instead of ORT's (default "0")

    # --- Compile behavior and caching ---
    "migraphx_exhaustive_tune": "0",         # "1" = try more kernel configs at compile time for a possible speed gain; slower first compile (default "0")
    "migraphx_model_cache_dir": "",          # Directory that caches the compiled MIGraphX program across process runs (empty = no cache dir)

    # --- GPU memory arena ---
    "migraphx_mem_limit": str(2 * 1024**3),  # EP arena byte limit as a string; example = 2 GiB (default if unset: unbounded / SIZE_MAX)
    "migraphx_arena_extend_strategy": "kNextPowerOfTwo",  # "kNextPowerOfTwo" (default) or "kSameAsRequested"
}

session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("MIGraphXExecutionProvider", options),
        "CPUExecutionProvider",
    ],
)
```

| Option | Type | Meaning |
|---|---|---|
| `device_id` | int string | ROCm GPU index, default `"0"` |
| `migraphx_fp16_enable` | `"0"`/`"1"` | Enable FP16 conversion where supported (default `"0"`) |
| `migraphx_bf16_enable` | `"0"`/`"1"` | Enable BF16 conversion where supported (default `"0"`) |
| `migraphx_fp8_enable` | `"0"`/`"1"` | Enable FP8 conversion; hardware/model dependent (default `"0"`) |
| `migraphx_int8_enable` | `"0"`/`"1"` | Enable INT8; requires calibration unless the model is pre-quantized (default `"0"`) |
| `migraphx_int8_calibration_table_name` | path | INT8 calibration table file name/path (used only when INT8 is enabled) |
| `migraphx_int8_use_native_calibration_table` | `"0"`/`"1"` | Use MIGraphX's own calibration table format instead of ORT's (default `"0"`) |
| `migraphx_exhaustive_tune` | `"0"`/`"1"` | Try more kernel configs at compile time for a possible speed gain; slower first compile (default `"0"`) |
| `migraphx_model_cache_dir` | path | Directory that caches the compiled MIGraphX program across runs |
| `migraphx_mem_limit` | bytes (string) | EP GPU arena limit; default is unbounded (`SIZE_MAX`) |
| `migraphx_arena_extend_strategy` | `kNextPowerOfTwo` / `kSameAsRequested` | Arena growth policy (default `kNextPowerOfTwo`) |
| `migraphx_external_alloc`, `migraphx_external_free`, `migraphx_external_empty_cache` | pointer address (string) | Advanced: share an existing GPU allocator (e.g. from PyTorch) with MIGraphX via raw function-pointer addresses; leave unset for normal use |

> [!NOTE]
> Do not enable reduced precision until accuracy has been measured against a trusted CPU/reference run.

### 18.2 Windows DirectML GPU

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.enable_mem_pattern = False
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    "model.onnx",
    sess_options=options,
    providers=[
        ("DmlExecutionProvider", {"device_id": "0"}),
        "CPUExecutionProvider",
    ],
)
```

### 18.3 Ryzen AI Vitis AI NPU

See [§13](#13-vitis-ai-provider-options-by-generation) for the full `VitisAIExecutionProvider` option reference (target/xclbin by generation, caching, EPContext). Minimal example:

```python
import onnxruntime as ort

options = {
    "target": "X2",                      # "X2" (STX/KRK, default) or "X1" (PHX/HPT, required there — plus "xclbin")
    "cache_dir": "./vitis-cache",         # compiler cache directory, reused across runs
    "cache_key": "model-v1",              # bump when the model or options change
    "enable_cache_file_io_in_mem": "0",   # "0" = cache on disk (inspectable); "1" = memory-only
}

session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("VitisAIExecutionProvider", options),
        "CPUExecutionProvider",
    ],
)
```

### 18.4 Advanced: build ONNX Runtime with an AMD EP

Use a source build only when no released package fits, or a custom ORT feature is genuinely required — it increases the compatibility surface and does not replace the device driver/runtime.

> [!WARNING]
> The one-click verifier accepts only the release binary hashes audited by this guide, so a custom source build fails its provenance gate even if legitimate. Validate a source build with its own ORT provider tests plus the same profile-placement methodology — do not present it as the audited prebuilt stack.

**Linux MIGraphX wheel** (matching ROCm/MIGraphX, supported compiler/CMake/Python, sufficient RAM/disk):

```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout v1.23.2
git submodule update --init --recursive

./build.sh \
  --config Release \
  --parallel \
  --build_wheel \
  --use_migraphx \
  --migraphx_home /opt/rocm

python -m pip install build/Linux/Release/dist/*.whl
```

Add `--build_shared_lib` for a reusable C/C++ library. Run applicable tests before packaging — don't use `--skip_tests` to hide an incompatibility.

**Windows DirectML wheel** (Visual Studio Developer PowerShell, supported Windows SDK):

```powershell
git clone --recursive --branch v1.24.4 `
  https://github.com/microsoft/onnxruntime.git onnxruntime-dml-1.24.4
cd onnxruntime-dml-1.24.4
.\build.bat --config Release --parallel --use_dml --build_wheel
```

**Windows Vitis AI build** — not a replacement for the Ryzen AI installer and not a rookie path. Only for developers who already have the matching Ryzen AI/Vitis AI dependencies and the exact ORT source revision that SDK requires:

```powershell
.\build.bat --use_vitisai --build_shared_lib --parallel --config Release --build_wheel
```

Do not run this in the DirectML checkout above or on an arbitrary `main` branch — get the supported ORT revision from the Ryzen AI release package/support channel first. For Linux Adaptive SoCs, follow the board's Vitis AI target setup instead of treating the x86 ROCm build as interchangeable. Provider `.so`/`.dll` files must stay co-located with the matching ORT runtime — never combine binaries from different builds.

---

## 19. Verification flow

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart LR
    A["Driver sees the device"] --> B["ORT wheel or plugin loads the target EP"]
    B --> C["Session registers the target EP first"]
    C --> D["Model compiles and partitions"]
    D --> E["Warm-up inference"]
    E --> F["Timed inference"]
    F --> G["Parse the ORT profile provider field"]
    G --> H{"Target-EP profile events > 0?"}
    H -->|No, Vitis path| N{"Fresh report has NPU nodes?"}
    H -->|No, other EP| I["Fail: CPU fallback or unsupported graph"]
    N -->|No| I
    H -->|Yes| J{"Was full offload requested?"}
    N -->|Yes| J
    J -->|No| K["Pass, evidence type labeled"]
    J -->|"Yes, CPU count = 0"| L["Strict pass"]
    J -->|"Yes, CPU count > 0"| M["Fail: partial offload"]

    classDef step fill:#e0f2fe,stroke:#0ea5e9,color:#0c2a3d;
    classDef dec fill:#fef3c7,stroke:#f59e0b,color:#713f12;
    classDef good fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef bad fill:#fee2e2,stroke:#ef4444,color:#7f1d1d;
    class A,B,C,D,E,F,G step;
    class H,J,N dec;
    class K,L good;
    class I,M bad;
```

| Platform | Command or UI | What to look for |
|---|---|---|
| Linux GPU | `/opt/rocm/bin/amd-smi monitor` or `metric` | GPU activity and VRAM during repeated inference |
| Linux GPU | `rocminfo` | Correct `gfx` target and device count |
| Windows GPU | Task Manager → GPU → Compute | Activity on the intended adapter |
| Windows/Linux NPU | Task Manager NPU / `xrt-smi examine` | NPU visibility and activity |
| Vitis AI | Assignment report | Nonzero NPU node count |

A tiny model can finish too fast for a utilization graph to register — repeat inference or use a real model, but keep profile-based node assignment as the primary correctness gate.

---

## 20. Performance guidance

| Recommendation | Why |
|---|---|
| Warm up before timing | First session/run can compile kernels, allocate memory, populate caches |
| Measure session creation separately | MIGraphX/Vitis AI compile time is not steady-state latency |
| Use fixed shapes when practical | Better folding, memory planning, and DirectML/MIGraphX compilation |
| Reuse one session | Avoid repeated compilation and allocator setup |
| Version cache keys | Prevent stale artifacts after model/driver/EP changes |
| Measure end-to-end and device-only time separately | NumPy CPU inputs/outputs include host-device transfer cost |
| Inspect CPU fallback | One unsupported operator can create expensive device boundaries |
| Establish an FP32 baseline first | Reduced precision can change accuracy and supported partitioning |
| Use I/O Binding only after correctness | Eliminates copies, but memory management is more complex |
| Pin a validated production stack | Driver + ROCm/XRT + EP + ORT + Python ABI must remain compatible |

---

## 21. Troubleshooting

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontSize":"14px","lineColor":"#94a3b8","edgeLabelBackground":"#e2e8f0","primaryTextColor":"#1e293b"}}}%%
flowchart TD
    A{"What do you see?"}
    A -->|Only CPUExecutionProvider| B["Wrong ORT distribution<br/>or vendor env not active"]
    A -->|"EP loads, exit code 5"| C["No target-EP profile event<br/>and no fresh Vitis NPU report"]
    A -->|Wrong GPU used| D["Adapter / device-id mismatch"]
    A -->|Vitis nodes all on CPU| E["Unsupported ops, shape,<br/>or wrong target/xclbin"]
    A -->|First NPU run is slow| F["Expected compilation,<br/>not a failure"]
    A -->|"--bootstrap refuses to run"| G["Not an isolated env,<br/>or unverifiable ROCm/hash"]

    classDef dec fill:#fef3c7,stroke:#f59e0b,color:#713f12;
    classDef step fill:#e0f2fe,stroke:#0ea5e9,color:#0c2a3d;
    class A dec;
    class B,C,D,E,F,G step;
```

| Symptom or error | Likely cause | Fix |
|---|---|---|
| Only `CPUExecutionProvider` appears | Wrong ORT distribution or inactive vendor environment | Create a clean venv; install the exact DML/MIGraphX wheel or activate the Ryzen AI environment |
| Multiple `onnxruntime-*` distributions reported | Overlapping wheels share the same module files | Delete and recreate the environment with exactly one runtime package |
| `--bootstrap` refuses the environment | Base/system Python, a vendor env, an existing ORT, or unverifiable/mismatched ROCm | Delete and recreate the dedicated disposable venv; bootstrap never repairs/uninstalls ORT in place |
| Reports an unaudited distribution or hash | Same-named PyPI wheel, modified binary, different release, or custom source build | Recreate from the direct vendor URL or `--bootstrap`; validate intentional source builds separately |
| `ROCMExecutionProvider` missing on ORT 1.23+ | Expected removal | Migrate to `MIGraphXExecutionProvider` |
| MIGraphX provider library cannot load | ROCm/MIGraphX version mismatch or missing runtime library | `sudo apt install migraphx`; inspect the provider `.so` with `ldd`; align the wheel repository |
| `Permission denied` for `/dev/kfd` | User not in `render,video` | `sudo usermod -a -G render,video $LOGNAME`, then log out or reboot |
| `hipErrorNoBinaryForGpu` / invalid device function | GPU architecture absent or unsupported | Check the official GPU matrix; don't rely only on `rocminfo` visibility |
| Import fails after a NumPy upgrade | AMD wheel ABI mismatch | Clean venv + `numpy==1.26.4` for the current wheel |
| DirectML uses the wrong GPU | `device_id=0` maps to another DXGI adapter | Check Task Manager; try `--device-id 1`; benchmark both |
| DirectML test rejects PCI vendor other than `0x1002` | Selected DXGI index is Intel/NVIDIA/Microsoft, not AMD | Use the printed adapter list; pass the AMD index with `--device-id` |
| DirectML session rejects options | Parallel mode or memory pattern enabled | Set sequential mode; disable memory pattern |
| Windows ML pip install fails on Python 3.10 | Pinned `onnxruntime-windowsml` declares Python >= 3.11 | Use the guide's Python 3.12 environment |
| Windows ML bootstrap fails / no MIGraphX catalog entry | `wasdk-*`/runtime mismatch, Store Python, OS below 24H2, or incompatible driver | Use the exact 2.1.3/1.24.6.202605042033 recipe, python.org/winget Python, build >=26100, exact live driver |
| Vitis AI EP present but all nodes on CPU | Unsupported ops/shapes/precision or wrong model generation | Use opset 17; check the supported-op table and assignment report; quantize/compile correctly |
| PHX/HPT Vitis session fails | Missing `target=X1` or `4x4.xclbin` | Use generation-specific options and the vendor install path |
| STX/KRK error mentions xclbin | Legacy option carried forward | Remove `xclbin` for the current X2 flow |
| First NPU load takes minutes | Expected compilation | Enable caching; separate compile time from inference time |
| NPU cache fails after an update | Cache/driver/EP incompatibility | Delete or version the cache; regenerate EP Context |
| Ubuntu cannot see the NPU | Kernel < 6.10, XRT/amdxdna absent, or unsupported PHX/HPT | Meet the exact 1.7.1 Linux gate; run `xrt-smi examine` |
| Docker cannot see the GPU | Device passthrough missing | Add `--device /dev/kfd --device /dev/dri`; verify the host driver |
| EP registered but demo exits with code 5 | No target-provider profile events and no fresh Vitis NPU evidence | Intentional fail-closed behavior — inspect unsupported nodes, the current-run report, and logs |

**Advanced Linux library check** — locate and inspect the MIGraphX provider library without copying it to a global system directory:

```bash
provider_so="$(find "$VIRTUAL_ENV" -name 'libonnxruntime_providers_migraphx.so' -print -quit)"
if [[ -z "$provider_so" ]]; then
  echo "MIGraphX provider library was not found in $VIRTUAL_ENV" >&2
else
  echo "$provider_so"
  ldd "$provider_so" | grep 'not found' || true
fi
```

ORT recommends keeping provider shared libraries beside the matching ORT library — do not globally mix `.so`/`.dll` files from different ORT installations.

---

## 22. Production checklist

- [ ] The hardware SKU is explicitly listed in the matching AMD support matrix.
- [ ] The OS build/kernel is exactly supported.
- [ ] Driver, ROCm/XRT, MIGraphX/Vitis AI, ORT, and Python ABI are pinned as one tested set.
- [ ] Only one `onnxruntime-*` distribution is installed in the environment.
- [ ] The target EP is first, and CPU fallback policy is intentional.
- [ ] A profile/assignment report proves target-device node execution.
- [ ] Accuracy is compared with CPU/reference data before reduced precision is enabled.
- [ ] First-run compilation and steady-state latency are measured separately.
- [ ] Cache invalidation policy covers model, EP, and driver version changes.
- [ ] Unsupported operators and CPU/device boundaries are documented.
- [ ] Deployment licenses for AMD/Windows ML/Vitis AI packages are reviewed.
- [ ] CI runs at least one real target-device smoke test; provider-list-only tests are rejected.

---

## 23. References

| Topic | Official source |
|---|---|
| ORT EP build page | <https://onnxruntime.ai/docs/build/eps.html#amd-migraphx> |
| MIGraphX EP | <https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html> |
| MIGraphX EP provider-options source | <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/migraphx/migraphx_execution_provider_info.h> |
| Removed ROCm EP notice | <https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html> |
| Vitis AI EP | <https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html> |
| VitisAI EP provider-options source | <https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/vitisai/vitisai_execution_provider.cc> |
| VSINPU EP source (not AMD — see appendix) | <https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/vsinpu> |
| DirectML EP | <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html> |
| PyPI DirectML package | <https://pypi.org/project/onnxruntime-directml/> |
| ORT Python API | <https://onnxruntime.ai/docs/api/python/api_summary.html> |
| ROCm documentation | <https://rocm.docs.amd.com/en/latest/> |
| ROCm Linux installation | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/> |
| ROCm Linux system requirements | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html> |
| ROCm Docker | <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html> |
| AMD ORT Docker tags | <https://hub.docker.com/r/rocm/onnxruntime/tags> |
| AMD ROCm wheel repository | <https://repo.radeon.com/rocm/manylinux/> |
| MIGraphX installation | <https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/install/install-migraphx.html> |
| Radeon native-Linux support and ONNX matrix | <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html> |
| Radeon 7.2.1 driver/ROCm installation | <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html> |
| Radeon MIGraphX + ONNX installation | <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html> |
| ROCm 7.14 release notes and compatibility | <https://rocm.docs.amd.com/en/latest/about/release-notes.html> · <https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html> |
| ROCm 7.14 ONNX Runtime / MIGraphX install | <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html> · <https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/migraphx.html> |
| Ryzen AI 1.7.1 documentation | <https://ryzenai.docs.amd.com/en/latest/> |
| Ryzen AI Windows installation | <https://ryzenai.docs.amd.com/en/latest/inst.html> |
| Ryzen AI Linux installation | <https://ryzenai.docs.amd.com/en/latest/linux.html> |
| Ryzen AI model deployment and options | <https://ryzenai.docs.amd.com/en/latest/modelrun.html> |
| Ryzen AI release notes | <https://ryzenai.docs.amd.com/en/latest/relnotes.html> |
| Ryzen AI supported operators | <https://ryzenai.docs.amd.com/en/latest/ops_support.html> |
| Windows ML overview | <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview> |
| Windows ML installation | <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app?tabs=python> |
| Windows ML available EPs | <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers> |
| Windows ML EP acquisition | <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers?tabs=python> |
| Windows ML EP registration | <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers?tabs=python> |
| Windows App SDK runtime downloads | <https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads> |
| Windows App Runtime installer options | <https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/deploy-unpackaged-apps> |
| PowerShell Authenticode verification | <https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/get-authenticodesignature> |
| PyWinRT Windows App Runtime bootstrap | <https://pywinrt.readthedocs.io/en/latest/api/winui3/index.html> |
| Windows ML Python package metadata | <https://pypi.org/project/wasdk-Microsoft.Windows.AI.MachineLearning/> |
| Official Miniforge installer | <https://github.com/conda-forge/miniforge#install> |
| Current Radeon WSL / ROCDXG guide and MIGraphX limitation | <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html> |
| Ubuntu HWE kernel | <https://ubuntu.com/kernel/lifecycle> |

> URLs and version matrices evolve quickly. Recheck the live compatibility page before upgrading a production image; never infer support solely from a higher version number.

---

<a id="appendix-vsinpu"></a>
## Appendix: VSINPU is not an AMD provider

> [!WARNING]
> `VSINPUExecutionProvider` has **no connection to AMD**. It targets **VeriSilicon/Vivante** NPU IP (the
> "VSI" in the name), found in SoCs such as NXP i.MX 8/9 and Amlogic — never an AMD GPU or Ryzen AI NPU. It
> is documented here only because it was requested alongside MIGraphX and VitisAI, and "VSINPU" vs
> "VitisAI" is an easy name mix-up. Do not install or configure it expecting it to touch AMD hardware, and
> do not confuse its `vsinpu` source folder with the AMD `vitisai` folder covered in
> [Part C](#part-c--windows-ryzen-ai-npuvitis-ai)/[Part D](#part-d--ubuntu-ryzen-ai-npuvitis-ai) above.

| Item | Detail |
|---|---|
| Source | [`onnxruntime/core/providers/vsinpu`](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/vsinpu) |
| Vendor | Vivante Corporation (VeriSilicon) — an unrelated third company, not AMD or Xilinx |
| Typical hardware | NXP i.MX 8/9-series SoCs, Amlogic SoCs, and other boards with a Vivante VIP9000-family NPU |
| ORT provider name | `VSINPUExecutionProvider` |
| Build flag | `--use_vsinpu` (source build only; ONNX Runtime does not publish a PyPI wheel for it) |

### Provider options

The public factory function, `CreateExecutionProviderFactory_VSINPU()`, takes **no arguments** — it always
constructs a default `VSINPUExecutionProviderInfo`. That struct declares exactly one field, and nothing in
the factory ever reads it from a provider-options map:

| Field | Type | Meaning |
|---|---|---|
| `device_id` | int | Declared in the C++ struct with default `0`, but not wired to any provider-options string key, C API parameter, or Python argument — it cannot currently be configured from outside the EP's own source code |

```python
import onnxruntime as ort

# No provider-options dict exists for this EP today — pass the bare provider name.
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        "VSINPUExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

> [!NOTE]
> If you actually want an AMD Ryzen AI NPU, use `VitisAIExecutionProvider` from
> [Part C](#part-c--windows-ryzen-ai-npuvitis-ai) (Windows) or [Part D](#part-d--ubuntu-ryzen-ai-npuvitis-ai)
> (Ubuntu) instead — see [§13](#13-vitis-ai-provider-options-by-generation) for its full option reference.
