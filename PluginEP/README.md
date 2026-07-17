# ONNX Runtime Plugin EP: Source-Audited Guide

**Start with one idea: a Plugin EP is a *doorway*, not a *destination*.** It is ONNX Runtime's public C ABI and runtime model for **loading, discovering, selecting, and packaging** execution providers (EPs). It is **not** a GPU/NPU, and there is no generic compute backend called `PluginExecutionProvider`.

> **The one-line answer:** Plugin EP lets you expose a brand-new out-of-tree EP *or* modernize how an existing EP ships. It is the delivery mechanism — never the compute provider itself.

### How to read this guide

The guide runs in three phases. Follow them in order the first time, then jump back to any box as a reference.

```mermaid
flowchart LR
    U["Phase 1<br/>Understand the model"] --> B["Phase 2<br/>Build your EP"] --> P["Phase 3<br/>Use and prove it"]
    U -.-> Ux["Start here<br/>Integration path<br/>Core objects"]
    B -.-> Bx["Lifecycle<br/>Execution model<br/>ABI versions"]
    P -.-> Px["Use it<br/>Prove execution<br/>Test and package"]

    style U fill:#1565c0,stroke:#90caf9,color:#ffffff
    style B fill:#6a1b9a,stroke:#ce93d8,color:#ffffff
    style P fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class Ux,Bx,Px leaf
```

**Audit baseline:** ONNX Runtime `main` at commit [`bf6aa00`](https://github.com/microsoft/onnxruntime/commit/bf6aa0063d1c178c4a4d33ed6770425834147e2a), checked on 2026-07-17. That development tree reports `ORT_VERSION=1.29.0` and `ORT_API_VERSION=29`; it is not a released-package contract. Runnable guides in this repository remain pinned to their tested package versions.

Throughout, every claim is tagged so you know how much to trust it:

| Tag | What it means for you |
|---|---|
| **Contract** | Guaranteed by a public header or the official Plugin EP docs — safe to rely on |
| **Source snapshot** | True at the pinned commit, but an implementation detail that can change |
| **Repository route** | Package-specific behavior this tutorial actually exercised and tested |

[简体中文](README.zh-CN.md) · [Official Plugin EP documentation](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/)

---

## Contents

**Phase 1 · Understand the model**
- [Start here](#start-here) — the three questions that untangle every confusion
- [Choose an integration path](#choose-an-integration-path) — internal, provider bridge, or pure plugin
- [Know the core objects](#know-the-core-objects) — who creates what, and who owns it

**Phase 2 · Build your EP**
- [Follow the lifecycle](#follow-the-lifecycle) — registration through execution and safe teardown
- [Choose an execution model](#choose-an-execution-model) — compile subgraphs, register kernels, or both
- [Handle ABI versions](#handle-abi-versions) — the two directions of compatibility

**Phase 3 · Use and prove it**
- [Use from an application](#use-from-an-application) — the register, select, run pattern
- [Prove execution](#prove-execution) — five levels of evidence
- [Build, test, and package](#build-test-and-package) — ship it without surprises
- [Repository routes and evidence](#repository-routes-and-evidence) — what this repo already verified

---

## Start here

Almost every Plugin EP question becomes easy once you separate **three independent choices**. Mix them up and everything feels confusing; keep them apart and the system clicks into place. Here is the whole guide on one screen, followed by those three dimensions.

### The whole guide on one screen

```mermaid
flowchart LR
    P["Plugin EP"]
    P --> A["Public C ABI"]
    P --> L["Loading"]
    P --> D["Device model"]
    P --> S["Shipping boundary"]

    A --> A1["OrtEpFactory + OrtEp callbacks"]
    A --> A2["Append-only versioned fields"]
    L --> L1["Register shared library at runtime"]
    L --> L2["Pure plugin or legacy bridge"]
    D --> D1["Discover hardware"]
    D --> D2["Select OrtEpDevice"]
    S --> S1["EP can ship outside ORT core"]
    S --> S2["Package and ORT version separately"]

    style P fill:#455a64,stroke:#cfd8dc,color:#ffffff
    style A fill:#1565c0,stroke:#90caf9,color:#ffffff
    style L fill:#c62828,stroke:#ef9a9a,color:#ffffff
    style D fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    style S fill:#e65100,stroke:#ffcc80,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class A1,A2,L1,L2,D1,D2,S1,S2 leaf
```

### Three dimensions, three answers

Ask which dimension a question belongs to first. A change in one rarely forces a change in the others.

```mermaid
flowchart TD
    Q["Any Plugin EP question"]
    Q --> ID["1 · EP identity<br/>Who claims and executes the nodes?"]
    Q --> LO["2 · Loading model<br/>How does EP code enter the process?"]
    Q --> EX["3 · Execution model<br/>How does ORT hand work over?"]

    style Q fill:#455a64,stroke:#cfd8dc,color:#ffffff
    style ID fill:#1565c0,stroke:#90caf9,color:#ffffff
    style LO fill:#c62828,stroke:#ef9a9a,color:#ffffff
    style EX fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
```

| Dimension | The question it answers | Examples |
|---|---|---|
| **1 · EP identity** | Who claims and executes graph nodes? | CUDA EP, QNN EP, WebGPU EP, a new vendor EP |
| **2 · Loading model** | How does EP code enter the process? | Built in, provider bridge library, pure Plugin EP library |
| **3 · Execution model** | How does ORT hand work to the EP? | Compile fused subgraphs, register operator kernels, or both |

> **Why this matters:** moving CUDA EP from an ORT wheel into a Plugin EP package changes loading, discovery, selection, ABI, and distribution. It does **not** automatically change CUDA operator coverage or model semantics.

---

## Choose an integration path

The shared `EpLibrary` abstraction has three source-level paths.

```mermaid
flowchart TD
    Q{"Is the EP compiled into ORT?"}
    Q -->|Yes| I["EpLibraryInternal"]
    Q -->|No| G{"Does the library export GetProvider?"}
    G -->|Yes| B["EpLibraryProviderBridge"]
    G -->|No| P["EpLibraryPlugin"]

    I --> I1["Factory/device model around an internal EP"]
    B --> B1["Public discovery + legacy IExecutionProvider creation"]
    P --> P1["Public ABI from factory through execution"]

    style I fill:#1565c0,stroke:#90caf9,color:#ffffff
    style B fill:#e65100,stroke:#ffcc80,color:#ffffff
    style P fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef note fill:#eceff1,stroke:#90a4ae,color:#20242b
    class I1,B1,P1 note
```

| Path | How ORT recognizes it | Session EP creation | Pinned-source examples | Best fit |
|---|---|---|---|---|
| **Internal** | Registered by ORT itself | Direct internal factory | CPU; DML when `USE_DML`; WebGPU when `USE_WEBGPU && !ORT_USE_EP_API_ADAPTERS` | ORT core builds |
| **Provider bridge** | Dynamic library has `GetProvider` **and** the two factory exports | Calls legacy `Provider::CreateIExecutionProvider()` | CUDA, OpenVINO, QNN, MIGraphX, Vitis AI, TensorRT RTX | Staged modernization of an existing EP |
| **Pure plugin** | Dynamic library has the two factory exports and no `GetProvider` | Calls `OrtEpFactory::CreateEp()` and wraps `OrtEp` | Native WebGPU, standalone CUDA plugin, sample plugins | New or fully decoupled EP |

The loader probes `GetProvider`; the application does not choose a path flag. CUDA appears in both bridge and pure-plugin source because those are distinct delivery routes.

### Required pure-plugin exports

```cpp
OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

| Contract | Practical meaning |
|---|---|
| Exact C symbols are required | ORT resolves `CreateEpFactories` and `ReleaseEpFactory` |
| Public host boundary | The library obtains versioned API tables from `OrtApiBase` |
| No C++ exception crosses the boundary | Catch exceptions and return `OrtStatus*` |
| Internal code may be reused at build time | The **runtime ABI boundary**, not every implementation file, must stay public-C-compatible |

---

## Know the core objects

### Object and ownership map

```mermaid
flowchart LR
    LIB["Plugin library"] -->|creates| F["OrtEpFactory"]
    ORT["ORT Environment"] -->|discovers| H["OrtHardwareDevice"]
    F -->|accepts hardware| D["OrtEpDevice"]
    H --> D
    APP["Application"] -->|selects| D
    D -->|session creation| E["OrtEp"]
    F -->|creates + releases| E
    E --> C["Compiled subgraphs"]
    E --> K["Registered kernels"]

    style LIB fill:#455a64,stroke:#cfd8dc,color:#ffffff
    style ORT fill:#1565c0,stroke:#90caf9,color:#ffffff
    style APP fill:#e65100,stroke:#ffcc80,color:#ffffff
    style F fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class H,D,E,C,K leaf
```

| Object | Created by | Lifetime / owner | Job |
|---|---|---|---|
| `OrtHardwareDevice` | ORT discovery, or a permitted virtual-device factory | Environment | Describes physical or virtual CPU/GPU/NPU hardware |
| `OrtEpFactory` | Plugin's `CreateEpFactories()` | Library registration; released through `ReleaseEpFactory()` | Names the EP, accepts devices, provides shared resources, creates `OrtEp` |
| `OrtEpDevice` | Factory calls `OrtEpApi::CreateEpDevice()`; ORT takes ownership | Environment registration | Pairs **one factory** with **one hardware device**; carries metadata and default options |
| `OrtEp` | `OrtEpFactory::CreateEp()` | Session; released by `OrtEpFactory::ReleaseEp()` | Claims nodes and executes them for that session |
| `OrtNodeComputeInfo` | A compiling `OrtEp` | Held by ORT for the session; batch-released by the EP | Defines create-state, compute, and release-state callbacks for one compiled graph |
| `OrtKernelRegistry` | Kernel-based EP | Original remains EP-owned; this source snapshot copies registrations into ORT | Defines per-operator kernel creation and implementation callbacks |

An `OrtEpDevice` is a selectable **EP + hardware pairing**, not device memory and not an allocator.

### Three names, three jobs

Newcomers often collapse these three names into one. They are chosen by different people for different purposes, and only one pair is ever required to match.

```mermaid
flowchart TD
    A["Registration name<br/>you choose · unique per OrtEnv"]
    B["Package name<br/>distributor chooses · for install"]
    C["EP name<br/>factory and EP choose"]
    C --> C1["OrtEpFactory::GetName()"]
    C --> C2["OrtEp::GetName()"]
    C1 <-->|MUST be identical| C2

    style A fill:#1565c0,stroke:#90caf9,color:#ffffff
    style B fill:#e65100,stroke:#ffcc80,color:#ffffff
    style C fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class C1,C2 leaf
```

| Name | Chosen by | Used for | Equality rule |
|---|---|---|---|
| Registration name | Application | Key inside one `OrtEnv`; unregistering the library | Must be unique in that environment |
| EP name | Factory / EP implementation | Device filtering, session provider identity, node assignment | `OrtEpFactory::GetName()` and `OrtEp::GetName()` **must match** |
| Package name | Distributor | Installing and locating the shared library | No required equality with either name |

`onnxruntime-ep-webgpu` may be a package name; it is not automatically the EP name or registration name.

### What the runtime checks

| Point | Pinned-source check | Do not assume |
|---|---|---|
| Registration | Duplicate registration name is rejected | A package name is a valid EP name |
| Explicit device selection | Every device has the same EP name **and the same factory pointer** | Same text name from different factories is enough |
| Initial pure `OrtEp` sanity check | `ort_version_supported >= 22`; `GetName` pointer and returned string are non-null | The complete callback table or factory/EP name equality is validated up front |
| Callback use | Required callbacks are checked when that path is used | Session construction catches every bad optional callback |

The official contract still requires matching factory and EP names. Do not rely on a late runtime failure to catch a mismatch.

---

## Follow the lifecycle

### Registration through execution

```mermaid
sequenceDiagram
    participant App
    participant Loader as ORT loader
    participant Lib as EP library
    participant Env as ORT Environment
    participant Factory as OrtEpFactory
    participant EP as OrtEp

    App->>Loader: RegisterExecutionProviderLibrary(name, path)
    Loader->>Loader: Resolve path and probe GetProvider
    Loader->>Lib: Load and resolve two factory symbols
    Loader->>Lib: CreateEpFactories(OrtApiBase, logger, ...)
    Lib-->>Loader: OrtEpFactory values
    Env->>Factory: GetSupportedDevices(all discovered hardware)
    Factory->>Env: OrtEpDevice values
    Env->>Env: Register devices, allocators, data transfer
    App->>Env: GetEpDevices()
    Env-->>App: Selectable EP + hardware pairings

    App->>App: Add selected device(s) to SessionOptions
    App->>Env: Create Session
    Env->>Factory: CreateEp for a pure plugin
    Factory-->>Env: OrtEp
    Env->>EP: GetCapability, then compile and/or kernels
    App->>Env: Run model
    Env->>EP: Execute assigned work
```

Internal and provider-bridge factories diverge at session creation: they create an internal `IExecutionProvider` directly. A pure plugin goes through `CreateEp()` and ORT's internal `PluginExecutionProvider` adapter.

### Source-snapshot facts

| Topic | Pinned behavior | Stability |
|---|---|---|
| Relative library path | `GetRuntimePath() / relative_path`; this is the ORT runtime directory, not the process working directory | **Source snapshot**; package helpers should return an absolute path |
| Factory output capacity | Loader currently supplies 4 slots | **Source snapshot**, not a plugin constant |
| Device output capacity | Environment currently supplies 8 slots per factory | **Source snapshot**, not a plugin constant |
| Virtual devices | Registration names ending in `.virtual` temporarily set `allow_virtual_devices=1` | **Source snapshot**; useful for cross-compilation, not proof of local hardware |
| Minimal build | Registration, devices, V2 append, and `GetEpApi()` are unavailable / `ORT_NOT_IMPLEMENTED` | Build capability |
| Multi-device factory | One `OrtEp` receives selected devices and must coordinate them | **Contract** |
| ORT-managed cross-device partitioning | Expose separate factories, each supporting one device and producing a unique EP name | **Contract** |

### Destruction state machine

```mermaid
stateDiagram-v2
    [*] --> Registered: register library
    Registered --> Active: create Session
    Active --> Registered: destroy Session and session-bound objects
    Registered --> Unloaded: unregister library
    Active --> Unsafe: unregister too early
    Unsafe --> [*]: callbacks may point into unloaded code
    Unloaded --> [*]
```

The public API places the precondition on the caller: **all sessions using the library must be released before unregistering it**. The loader does not detect or reference-count live sessions for application safety.

| Order | Normal cleanup event |
|---:|---|
| 1 | Release `RunOptions`, `IOBinding`, outputs, and other session-bound objects |
| 2 | Destroy the session; compiled EPs release `OrtNodeComputeInfo`, then the factory releases `OrtEp` |
| 3 | Call `UnregisterExecutionProviderLibrary(registration_name)` |
| 4 | ORT unregisters data transfer, internal-factory lookup entries, devices, and shared allocators |
| 5 | ORT clears `OrtEpDevice` values, calls `ReleaseEpFactory`, then unloads the shared library |

At environment destruction, ORT clears shared allocators before unloading remaining EP libraries because allocator deleters may call plugin code.

---

## Choose an execution model

```mermaid
flowchart TD
    G["OrtEp::GetCapability"] --> Q{"How is a supported node group reported?"}
    Q -->|"AddNodesToFuse"| C["Compiling path"]
    Q -->|"AddSingleNode"| K["Kernel-registry path"]
    C --> C1["ORT creates fused graph"]
    C1 --> C2["OrtEp::Compile"]
    C2 --> C3["OrtNodeComputeInfo per graph"]
    K --> K1["Look up OrtKernelDef"]
    K1 --> K2["Create OrtKernelImpl"]
    K2 --> K3["Invoke kernel Compute"]

    style C fill:#1565c0,stroke:#90caf9,color:#ffffff
    style K fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class C1,C2,C3,K1,K2,K3 leaf
```

| Dimension | Compiling EP | Kernel-registry EP | Mixed EP |
|---|---|---|---|
| Stable execution surface | 1.23 | 1.24 | 1.24+ |
| Capability report | `EpGraphSupportInfo_AddNodesToFuse()` | `EpGraphSupportInfo_AddSingleNode()` after kernel lookup | Uses both by node/group |
| Runtime object | `OrtNodeComputeInfo` for each fused graph | `OrtKernelDef` + create function + `OrtKernelImpl` | Both |
| `Compile` | Required | May be null | Handles only fused groups |
| Typical fit | Backend compiler, graph accelerator, EPContext | Existing operator-kernel library | Gradual migration or specialized fused ops |

### Ownership and lifetime rules

| Item | Rule |
|---|---|
| `OrtGraph` passed to `GetCapability()` / `Compile()` | Temporary; copy every name or fact needed later |
| `OrtNodeComputeInfo` | EP allocates; ORT retains for the session; EP releases in `ReleaseNodeComputeInfos()` |
| EPContext node returned by `Compile()` | ORT takes ownership |
| Registry returned by `GetKernelRegistry()` | Must remain valid for the EP lifetime; pinned ORT source copies its registrations |
| If / Loop / Scan | Public 1.24 `OrtEpApi` helpers create control-flow kernels that can access ORT session internals |

The official TensorRT Plugin EP example exposes both `Compile` and `GetKernelRegistry`, confirming that the models can be combined.

---

## Handle ABI versions

### Two directions of compatibility

```mermaid
flowchart LR
    H["Plugin build headers"] --> S["Set callback struct version to compiled ORT_API_VERSION"]
    S --> O["ORT calls only fields known to that struct version"]

    R["Loaded ORT runtime"] --> A["Negotiate runtime API version"]
    A --> G["Gate every newer callback and API call"]
    G --> P["Plugin calls only functions present in the runtime"]

    style H fill:#1565c0,stroke:#90caf9,color:#ffffff
    style R fill:#c62828,stroke:#ef9a9a,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class S,O,A,G,P leaf
```

| Direction | Correct guard | Common mistake |
|---|---|---|
| ORT calls plugin callback structs | Set `ort_version_supported` / `version` to the **header version used to compile the plugin** | Lowering it to pretend the runtime is older |
| Plugin calls `OrtApi` / `OrtEpApi` | Detect the **loaded runtime version**, enforce a minimum, and gate newer calls | Calling `GetApi(ORT_API_VERSION)` and assuming an older runtime has that table |

The pure CUDA and WebGPU plugins use `ApiInit(ort_api_base, ORT_PLUGIN_EP_MIN_ORT_VERSION)`; CUDA also gates optional callbacks with the negotiated runtime version. `ort_version_supported` does not perform that negotiation.

### Version map at the pinned snapshot

Each ORT release *adds* surface; it does not rewrite what came before. Read this timeline as "what became available when."

```mermaid
flowchart LR
    V22["1.22<br/>Register + discover"] --> V23["1.23<br/>Compiling EP path"]
    V23 --> V24["1.24<br/>Kernel-registry path"]
    V24 --> V25["1.25<br/>Profiler + Sync"]
    V25 --> V26["1.26<br/>Graph capture/replay"]
    V26 --> V27["1.27<br/>Session-init hooks"]
    V27 --> V28["1.28<br/>SelectBestModelCandidate"]

    style V23 fill:#1565c0,stroke:#90caf9,color:#ffffff
    style V24 fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef leaf fill:#eceff1,stroke:#90a4ae,color:#20242b
    class V22,V25,V26,V27,V28 leaf
```

| ORT API | Public surface added or extended | Ground-truth note |
|---:|---|---|
| 1.22 | Library register/unregister; hardware/EP-device discovery and selection; base factory/EP fields | Foundation; graph execution callbacks stabilize in 1.23 |
| 1.23 | Graph inspection, `GetCapability`, `Compile`, `OrtNodeComputeInfo`, allocators, transfer, streams, layout, run hooks, compiled-model compatibility | Compiling EP path |
| 1.24 | Kernel registry, If/Loop/Scan helpers, virtual devices, external resources, custom-op domains, incompatibility details | Kernel path; `Compile` becomes optional for registry-only EPs |
| 1.25 | EP profiler and profiling events, operator-schema queries, `OrtEp::Sync`, graphics interop | Last append currently present in the `OrtEpApi` function table |
| 1.26 | Resource budgets and graph capture/replay callbacks | Added mainly to `OrtEp` and related types, not `OrtEpApi` |
| 1.27 | Session-initialization completion, default memory device, captured-graph release | Latest `OrtEp` callbacks in this snapshot |
| 1.28 | `OrtEpFactory::SelectBestModelCandidate`; core `OrtApi` adds `KernelContext_GetSyncStream` and the stable lookup for experimental functions | Latest `OrtEpFactory` callback in this snapshot |
| 1.29 development tree | Header reports API 29, but the pinned source has no finalized 1.29 `OrtApi`, `OrtEpApi`, `OrtEp`, or `OrtEpFactory` additions | Do not infer a released 1.29 contract from `main` |

`OrtEpApi` itself ends at a version-25 slot assertion in this source. Later Plugin EP capabilities are also carried by callback structs and the core `OrtApi`, so “Plugin EP API version” is not one single table.

> [!IMPORTANT]
> Treat the `main` header, a released ORT package, and a vendor plugin package as three separately versioned artifacts. Successful compilation does not prove runtime compatibility.

---

## Use from an application

The API family begins in ORT 1.22, but a real plugin may require a later ORT patch/minor version. Follow the package's declared floor and runtime check.

```python
import onnxruntime as ort
import vendor_plugin_ep

registration_name = "my_plugin_registration"
library_path = vendor_plugin_ep.get_library_path()
ep_names = vendor_plugin_ep.get_ep_names()
if not ep_names:
    raise RuntimeError("The plugin package did not report an EP name")
ep_name = ep_names[0]

ort.register_execution_provider_library(registration_name, library_path)
session = None
try:
    devices = [device for device in ort.get_ep_devices() if device.ep_name == ep_name]
    if not devices:
        raise RuntimeError(f"Plugin loaded, but no compatible {ep_name} device was found")

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices([devices[0]], {})

    session = ort.InferenceSession("model.onnx", sess_options=options)
    # Run fixed inputs, compare outputs, and collect assignment/profile evidence.
finally:
    del session
    ort.unregister_execution_provider_library(registration_name)
```

This follows the official Python example's API names and teardown order.

| Common trap | Do this instead |
|---|---|
| Assume registration name equals EP name | Read package helper names and filter `get_ep_devices()` |
| Use `get_available_providers()` as the dynamic device directory | Register first, then use `get_ep_devices()` |
| Pass every device with the same text name | Start with one device; multiple values must share the exact factory |
| Pass a relative path from the working directory | Prefer `get_library_path()` returning an absolute path |
| Treat session creation as execution proof | Disable CPU fallback, run, compare output, and inspect assignment/profile |
| Load an unknown plugin package | A plugin is native in-process code; load only trusted artifacts |

Automatic selection uses `SessionOptionsSetEpSelectionPolicy()` or a custom delegate. It selects devices; it does not prove that the chosen EP covers the model.

---

## Prove execution

```mermaid
flowchart TD
    A{"Library registered?"} -->|No| A0["Loader / symbol / ABI issue"]
    A -->|Yes| B{"Expected OrtEpDevice found?"}
    B -->|No| B0["Discovery / compatibility issue"]
    B -->|Yes| C{"Strict session and Run succeed?"}
    C -->|No| C0["Coverage / options / runtime issue"]
    C -->|Yes| D{"Output matches independent reference?"}
    D -->|No| D0["Kernel / compile correctness issue"]
    D -->|Yes| E{"Assignment and device evidence agree?"}
    E -->|No| E0["Attribution is still unproven"]
    E -->|Yes| PASS["Execution claim is supported"]

    style PASS fill:#2e7d32,stroke:#a5d6a7,color:#ffffff
    classDef fail fill:#ffebee,stroke:#c62828,color:#20242b
    class A0,B0,C0,D0,E0 fail
```

| Level | Evidence | What it proves |
|---:|---|---|
| 1 | Registration succeeds | Library loaded; required symbols and factory creation worked |
| 2 | Expected `OrtEpDevice` appears | Factory accepted a discovered or permitted virtual device |
| 3 | CPU fallback disabled; session and `Run` succeed | ORT did not silently assign unsupported graph work to CPU fallback |
| 4 | Output matches an independent CPU/NumPy reference | Numerical behavior is correct within a declared tolerance |
| 5 | Assignment/profile names the EP; vendor trace shows device work | ORT assignment and target-hardware activity support the execution claim |

For C/C++, source API 1.24 can record `session.record_ep_graph_assignment_info=1` and query `Session_GetEpGraphAssignmentInfo()`. ORT profiles can also attribute node events. From 1.25, an `OrtEpProfilerImpl` may merge plugin device events into the ORT timeline. Latency or utilization alone is only a supporting signal.

---

## Build, test, and package

```mermaid
flowchart LR
    U["Callback unit tests"] --> O["onnxruntime_provider_test"]
    O --> M["Strict model tests"]
    M --> N["Numerical comparison"]
    N --> A["Assignment + profile"]
    A --> V["Minimum + target ORT"]
    V --> L["Concurrency + leak checks"]

    classDef stage fill:#eceff1,stroke:#607d8b,color:#20242b
    class U,O,M,N,A,V,L stage
```

| Area | Required baseline | Official/source anchor |
|---|---|---|
| ABI entry | Public headers; two C exports; no escaping C++ exceptions | Development guide; WebGPU/CUDA entry points |
| Struct setup | Zero-initialize; set compiled `ORT_API_VERSION`; populate only supported callbacks | Public header and samples |
| Identity | Factory and EP names match; version string is SemVer | Development guide |
| Discovery | Return only genuinely compatible devices; return none when unsupported | `GetSupportedDevices()` contract |
| Graph data | Do not retain temporary `OrtGraph` / node data without copying | `Compile()` header notes |
| Resources | Design allocator, transfer, stream, custom domain, and unload lifetimes together | Factory API and environment cleanup |
| Plugin tests | Own callback, error, no-device, bad-version, repeat-load, and cleanup tests | Official testing guidance |
| ORT operator tests | Build `onnxruntime_provider_test`; configure `ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON` | Official testing guidance |
| Model tests | Relevant full models; strict no-fallback run; output and assignment checks | Official guidance favors high-level model tests |
| Version CI | Minimum supported runtime plus target runtime; gate newer APIs | CUDA/WebGPU `ApiInit` pattern |
| Package contents | Plugin library and its dependencies; do not bundle another ORT core library | Official packaging guidance |
| Package helpers | `get_library_path()`, `get_ep_names()`, optionally `get_ep_name()` | Official PyPI guidance |
| Package dependency | Generic guidance avoids hard-depending on one ORT flavor; always document and validate the compatible ORT range | Official packaging guidance; vendor packages may choose stricter metadata |

Official testing currently expects each EP to own integration/model coverage; it does not provide a complete ABI conformance suite.

---

## Repository routes and evidence

### Routes exercised here

| Repository route | Loading class in upstream source family | Relationship to the traditional EP | Strict test |
|---|---|---|---|
| AMD Windows ML MIGraphX | Provider bridge | Existing MIGraphX backend exposed through factory/device discovery | [AMD/provider_test.py](../AMD/provider_test.py) |
| Qualcomm QNN 2.x | Provider bridge | QNN CPU/GPU/HTP backends decoupled from a monolithic ORT package | [Qualcomm/one_click.py](../Qualcomm/one_click.py) |
| NVIDIA TensorRT RTX | Provider bridge | Distinct TensorRT RTX product from classic TensorRT EP; same Plugin EP loading model | [NVIDIA/provider_test.py](../NVIDIA/provider_test.py) |
| Native WebGPU | Pure plugin | Native ORT host and package; not the browser `onnxruntime-web` API | [native_webgpu_validator.py](../WebGPU/onnxruntime-web-demo/native_webgpu_validator.py) |

Upstream also contains a pure standalone CUDA Plugin EP. That does not mean this repository's current built-in `CUDAExecutionProvider` route has been replaced; keep package, dependency, and validation claims separate.

### Pinned source ledger

All source links below use the audited commit, not moving `main`.

| Source | Claim verified |
|---|---|
| [`onnxruntime_ep_c_api.h`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/include/onnxruntime/core/session/onnxruntime_ep_c_api.h) | Public structs, ownership notes, callback versions, 4/8 current capacities, 1.28 factory tail |
| [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/include/onnxruntime/core/session/onnxruntime_c_api.h) and [`onnxruntime_c_api.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/onnxruntime_c_api.cc) | Registration contract, core API version, append-only slot assertions, minimal-build stubs |
| [`utils.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/utils.cc) | Relative path base, `GetProvider` probe, same-name + same-factory selection |
| [`ep_library_plugin.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_library_plugin.cc) | Required symbols, factory creation/release, dynamic unload |
| [`environment.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/environment.cc) | Duplicate names, devices, virtual mode, allocators/transfers, unregister order |
| [`ep_library_internal.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_library_internal.cc) and [`ep_library_provider_bridge.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_library_provider_bridge.cc) | Internal provider list and legacy bridge adaptation |
| [`ep_plugin_provider_interfaces.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_plugin_provider_interfaces.cc) | Pure-plugin adapter, sanity checks, capability, compile, release ordering |
| [`ep_kernel_registration.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_kernel_registration.cc) and [`ep_api.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/session/plugin_ep/ep_api.cc) | Registry copy, control-flow helpers, `OrtEpApi` version slots |
| [`example_plugin_ep`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/test/autoep/library/example_plugin_ep) and [`example_plugin_ep_kernel_registry`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry) | Reference compile and kernel-registry implementations |
| [`cuda/plugin`](https://github.com/microsoft/onnxruntime/tree/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/cuda/plugin) and [`webgpu/ep/api.cc`](https://github.com/microsoft/onnxruntime/blob/bf6aa0063d1c178c4a4d33ed6770425834147e2a/onnxruntime/core/providers/webgpu/ep/api.cc) | Pure plugin entry points and runtime-version negotiation |

Official references: [Usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html) · [Development](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/development.html) · [Testing](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/testing.html) · [Packaging](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/packaging.html)