"use strict";

const ORT_VERSION = "1.27.0";
const MODEL_PATH = "./execution_provider_demo.onnx";
const MODEL_DIMS = Object.freeze([1, 4, 128, 128]);
const MATRIX_SIZE = 128;
const SMOKE_BIAS = 0.125;
const MODEL_INPUTS = Object.freeze([
  Object.freeze({ name: "left", type: "float32", shape: MODEL_DIMS }),
  Object.freeze({ name: "right", type: "float32", shape: MODEL_DIMS }),
]);
const MODEL_OUTPUTS = Object.freeze([
  Object.freeze({ name: "output", type: "float32", shape: MODEL_DIMS }),
]);
const VALID_PROVIDERS = new Set(["wasm", "webgpu", "webnn"]);
const VALID_WEBNN_DEVICES = new Set(["cpu", "gpu", "npu"]);

const ui = Object.freeze({
  provider: document.querySelector("#provider"),
  device: document.querySelector("#device"),
  iterations: document.querySelector("#iterations"),
  fallback: document.querySelector("#fallback"),
  profile: document.querySelector("#profile"),
  run: document.querySelector("#run"),
  log: document.querySelector("#log"),
  status: document.querySelector("#status"),
  dot: document.querySelector("#dot"),
  secure: document.querySelector("#secure"),
  isolated: document.querySelector("#isolated"),
  webgpuApi: document.querySelector("#webgpu-api"),
  webnnApi: document.querySelector("#webnn-api"),
});

function setStatus(text, kind = "busy") {
  ui.status.textContent = text;
  ui.dot.className = `dot${kind === "ok" ? " ok" : kind === "bad" ? " bad" : ""}`;
}

function log(line = "") {
  ui.log.textContent += `${line}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function resetLog() {
  ui.log.textContent = "";
}

function tensorSize(dims) {
  return dims.reduce((product, value) => product * value, 1);
}

function validateModelContract(session) {
  const validate = (actualMetadata, expectedMetadata, label) => {
    if (actualMetadata.length !== expectedMetadata.length) {
      throw new Error(`Unexpected ${label} count: expected ${expectedMetadata.length}, got ${actualMetadata.length}.`);
    }
    for (let i = 0; i < expectedMetadata.length; i++) {
      const actual = actualMetadata[i];
      const expected = expectedMetadata[i];
      const shape = actual.isTensor ? Array.from(actual.shape) : [];
      if (!actual.isTensor || actual.name !== expected.name || actual.type !== expected.type ||
          JSON.stringify(shape) !== JSON.stringify(expected.shape)) {
        throw new Error(
          `Unexpected ${label} ${i}: expected ${expected.name} ${expected.type} ` +
          `[${expected.shape.join("×")}], got ${actual.name} ${actual.type || "non-tensor"} [${shape.join("×")}].`
        );
      }
    }
  };

  validate(session.inputMetadata, MODEL_INPUTS, "input");
  validate(session.outputMetadata, MODEL_OUTPUTS, "output");
}

function logModelContract(session) {
  log("[Model contract]");
  for (const metadata of session.inputMetadata) {
    log(`  input  ${metadata.name}: ${metadata.type} [${metadata.shape.join("×")}]`);
  }
  for (const metadata of session.outputMetadata) {
    log(`  output ${metadata.name}: ${metadata.type} [${metadata.shape.join("×")}]`);
  }
}

function makeInputData() {
  const size = tensorSize(MODEL_DIMS);
  const left = new Float32Array(size);
  const right = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    left[i] = Math.sin(i * 0.01) * 0.25;
    right[i] = Math.cos(i * 0.013) * 0.25;
  }

  return { left, right };
}

function makeFeeds(inputData) {
  return {
    left: new ort.Tensor("float32", inputData.left, MODEL_DIMS),
    right: new ort.Tensor("float32", inputData.right, MODEL_DIMS),
  };
}

function makeSmokeReference(inputData) {
  const matrixElements = MATRIX_SIZE * MATRIX_SIZE;
  const matrixCount = inputData.left.length / matrixElements;
  const output = new Float32Array(inputData.left.length);

  for (let batch = 0; batch < matrixCount; batch++) {
    const base = batch * matrixElements;
    for (let row = 0; row < MATRIX_SIZE; row++) {
      for (let column = 0; column < MATRIX_SIZE; column++) {
        let sum = 0;
        for (let inner = 0; inner < MATRIX_SIZE; inner++) {
          sum += inputData.left[base + row * MATRIX_SIZE + inner] *
            inputData.right[base + inner * MATRIX_SIZE + column];
        }
        output[base + row * MATRIX_SIZE + column] = Math.fround(
          Math.max(0, sum + SMOKE_BIAS)
        );
      }
    }
  }
  return output;
}

async function disposeResults(results) {
  for (const tensor of Object.values(results)) {
    tensor?.dispose?.();
  }
}

async function synchronizeWebGpu(enabled) {
  if (!enabled) return;
  const device = await ort.env.webgpu.device;
  const queue = device?.queue;
  if (queue?.onSubmittedWorkDone) {
    await queue.onSubmittedWorkDone();
  }
}

async function createWebGpuDevice(enableProfiling) {
  if (!globalThis.isSecureContext) {
    throw new Error("WebGPU requires HTTPS or http://localhost; file:// and ordinary HTTP are not secure contexts.");
  }
  if (!navigator.gpu) {
    throw new Error("navigator.gpu is unavailable. Update/enable a WebGPU-capable browser and inspect chrome://gpu.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("navigator.gpu exists, but requestAdapter() returned null. Enable hardware acceleration and update the GPU driver.");
  }
  const info = adapter.info || {};
  const identity = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean)
    .join(" · ") || "adapter available (identity hidden by browser)";
  log(`[Preflight] WebGPU adapter: ${identity}`);
  log(`[Preflight] WebGPU features: ${[...adapter.features].sort().join(", ") || "core only"}`);
  const requiredFeatures = [];
  if (adapter.features.has("chromium-experimental-timestamp-query-inside-passes")) {
    requiredFeatures.push("chromium-experimental-timestamp-query-inside-passes");
  } else if (adapter.features.has("timestamp-query")) {
    requiredFeatures.push("timestamp-query");
  }
  for (const feature of ["shader-f16", "subgroups"]) {
    if (adapter.features.has(feature)) requiredFeatures.push(feature);
  }
  if (enableProfiling && !requiredFeatures.some(feature => feature.includes("timestamp-query"))) {
    log("[Preflight] timestamp-query is unavailable; WebGPU profiling events may be absent.");
  }
  // Match the descriptor ORT 1.27 uses when it creates its own device. This
  // keeps the adapter's supported limits and optional optimized features while
  // still guaranteeing that the inspected adapter is the one used by ORT.
  const requiredLimits = {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
  };
  return adapter.requestDevice({ requiredFeatures, requiredLimits });
}

async function createWebNnContext(deviceType) {
  if (!globalThis.isSecureContext) {
    throw new Error("WebNN requires a secure context. Use this localhost launcher or HTTPS.");
  }
  if (!navigator.ml) {
    throw new Error("navigator.ml is unavailable. Use current Chrome/Edge Canary and enable the 'Enables WebNN API' feature.");
  }
  const context = await navigator.ml.createContext({
    deviceType,
    powerPreference: "high-performance",
  });
  log(`[Preflight] WebNN MLContext created (requested deviceType=${deviceType}).`);
  log("[Preflight] WebNN has no standard API that reveals the final native backend; use WebNN Report / Chromium histograms as documented.");
  return context;
}

function configureOrtEnvironment(provider, profilingSummary) {
  ort.env.debug = false;
  ort.env.logLevel = "warning";
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated
    ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
    : 1;
  ort.env.wasm.wasmPaths = globalThis.ORT_ASSET_BASE ||
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.webgpu.profiling = provider === "webgpu" && ui.profile.checked
    ? {
        mode: "default",
        ondata: data => {
          profilingSummary.events++;
          profilingSummary.programs.add(data.programName || data.kernelName || `kernel-${data.kernelId}`);
        },
      }
    : { mode: "off" };
}

function percentile(sorted, fraction) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

async function summarizeResults(results) {
  log("[Outputs]");
  for (const [name, tensor] of Object.entries(results)) {
    const values = await tensor.getData();
    const sample = Array.from(values.slice(0, 5), value =>
      typeof value === "bigint" ? value.toString() : Number(value).toPrecision(6)
    );
    log(`  ${name}: type=${tensor.type}, dims=[${tensor.dims.join("×")}], location=${tensor.location || "cpu"}`);
    log(`    first values: ${sample.join(", ")}`);
  }
}

function sameDims(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

async function compareWithIndependentReference(results, inputData) {
  log("[Correctness] Requested EP versus independent JavaScript MatMul/Add/Relu reference");
  const tensor = results.output;
  if (!tensor || tensor.type !== "float32" || !sameDims(tensor.dims, MODEL_DIMS)) {
    throw new Error("The smoke-model output is missing or has an unexpected type/shape.");
  }

  const actual = await tensor.getData();
  const expected = makeSmokeReference(inputData);
  let maxAbs = 0;
  let mismatches = 0;
  for (let i = 0; i < actual.length; i++) {
    const difference = Math.abs(Number(actual[i]) - Number(expected[i]));
    maxAbs = Math.max(maxAbs, difference);
    if (!Number.isFinite(actual[i]) || difference > 1e-3 + 1e-4 * Math.abs(expected[i])) {
      mismatches++;
    }
  }

  const passed = mismatches === 0;
  log(`  ${passed ? "PASS" : "FAIL"} output: max_abs_diff=${maxAbs.toPrecision(6)}, mismatches=${mismatches}`);
  if (!passed) throw new Error("Requested EP output failed the independent smoke-model reference check.");
}

async function compareWithWasm(candidateResults, referenceResults) {
  log("[Correctness] Requested EP versus WASM CPU reference");
  let passed = true;

  for (const expected of MODEL_OUTPUTS) {
    const candidate = candidateResults[expected.name];
    const reference = referenceResults[expected.name];
    if (!candidate || !reference || candidate.type !== reference.type || !sameDims(candidate.dims, reference.dims)) {
      log(`  FAIL ${expected.name}: output is missing or its type/shape differs from WASM.`);
      passed = false;
      continue;
    }

    const candidateData = await candidate.getData();
    const referenceData = await reference.getData();
    if (candidateData.length !== referenceData.length) {
      log(`  FAIL ${expected.name}: value count differs from WASM.`);
      passed = false;
      continue;
    }

    let maxAbs = 0;
    let mismatches = 0;
    const integerOutput = expected.type.startsWith("int") || expected.type.startsWith("uint");
    for (let i = 0; i < candidateData.length; i++) {
      const actual = Number(candidateData[i]);
      const wanted = Number(referenceData[i]);
      const difference = Math.abs(actual - wanted);
      maxAbs = Math.max(maxAbs, difference);
      const close = integerOutput
        ? difference <= 1
        : Number.isFinite(actual) && Number.isFinite(wanted) && difference <= 1e-3 + 1e-4 * Math.abs(wanted);
      if (!close) mismatches++;
    }

    const ok = mismatches === 0;
    log(`  ${ok ? "PASS" : "FAIL"} ${expected.name}: max_abs_diff=${maxAbs.toPrecision(6)}, mismatches=${mismatches}`);
    passed &&= ok;
  }

  if (!passed) throw new Error("Requested EP output failed the WASM parity check.");
}

async function runDemo() {
  if (typeof ort === "undefined") {
    throw new Error("ONNX Runtime Web did not load. Check Internet/CDN access or use the documented offline deployment.");
  }
  const loadedVersion = ort.env?.versions?.web;
  if (loadedVersion !== ORT_VERSION) {
    throw new Error(`ONNX Runtime Web version mismatch: expected ${ORT_VERSION}, loaded ${loadedVersion || "unknown"}.`);
  }

  const provider = VALID_PROVIDERS.has(ui.provider.value) ? ui.provider.value : "wasm";
  const deviceType = VALID_WEBNN_DEVICES.has(ui.device.value) ? ui.device.value : "gpu";
  const iterations = Math.max(1, Math.min(1000, Number.parseInt(ui.iterations.value, 10) || 10));
  const allowFallback = ui.fallback.checked && provider !== "wasm";

  ui.run.disabled = true;
  resetLog();
  setStatus(`Running ${provider.toUpperCase()}…`);
  const profilingSummary = { events: 0, programs: new Set() };
  configureOrtEnvironment(provider, profilingSummary);

  let session;
  let referenceSession;
  let inputData;
  let feeds;
  let finalResults;
  let referenceResults;
  try {
    log(`ONNX Runtime Web: ${loadedVersion}`);
    log(`Runtime source: ${globalThis.ORT_RUNTIME_SOURCE || "unknown"}`);
    log(`Model: ${MODEL_PATH}`);
    log(`Requested EP: ${provider}${provider === "webnn" ? ` (${deviceType})` : ""}`);
    log(`Policy: ${allowFallback ? `${provider} → WASM fallback` : "strict EP-only"}`);
    log(`WASM threads: ${ort.env.wasm.numThreads} (crossOriginIsolated=${globalThis.crossOriginIsolated})`);
    log();

    let primaryProvider = provider;
    if (provider === "webgpu") {
      const device = await createWebGpuDevice(ui.profile.checked);
      // Passing the inspected device prevents ORT from selecting a second,
      // potentially different adapter on a multi-GPU computer.
      primaryProvider = { name: "webgpu", device, validationMode: "basic" };
    } else if (provider === "webnn") {
      const context = await createWebNnContext(deviceType);
      // ORT 1.27 requires deviceType even when a pre-created MLContext is
      // supplied, because it uses the value to select the preferred layout.
      primaryProvider = { name: "webnn", deviceType, context };
    }

    const executionProviders = [primaryProvider];
    if (allowFallback) executionProviders.push("wasm");

    const sessionOptions = {
      executionProviders,
      executionMode: "sequential",
      graphOptimizationLevel: "all",
      enableGraphCapture: false,
      preferredOutputLocation: "cpu",
    };
    if (!allowFallback && provider !== "wasm") {
      sessionOptions.extra = { session: { disable_cpu_ep_fallback: "1" } };
    }

    const loadStart = performance.now();
    session = await ort.InferenceSession.create(MODEL_PATH, sessionOptions);
    const loadMs = performance.now() - loadStart;
    validateModelContract(session);
    log(`[Session] Created in ${loadMs.toFixed(3)} ms.`);
    logModelContract(session);

    inputData = makeInputData();
    feeds = makeFeeds(inputData);
    log("[Warm-up] 2 runs (excluded from statistics)…");
    for (let i = 0; i < 2; i++) {
      const warmupResults = await session.run(feeds);
      await synchronizeWebGpu(provider === "webgpu");
      await disposeResults(warmupResults);
    }

    const latencies = [];
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      const results = await session.run(feeds);
      await synchronizeWebGpu(provider === "webgpu");
      latencies.push(performance.now() - start);
      if (i === iterations - 1) {
        finalResults = results;
      } else {
        await disposeResults(results);
      }
    }

    const sorted = [...latencies].sort((a, b) => a - b);
    const mean = latencies.reduce((sum, value) => sum + value, 0) / latencies.length;
    log();
    log(`[Benchmark] ${iterations} measured run(s)`);
    log(`  min=${sorted[0].toFixed(3)} ms`);
    log(`  mean=${mean.toFixed(3)} ms`);
    log(`  median=${percentile(sorted, 0.5).toFixed(3)} ms`);
    log(`  p95=${percentile(sorted, 0.95).toFixed(3)} ms`);
    log(`  max=${sorted.at(-1).toFixed(3)} ms`);
    log();
    await summarizeResults(finalResults);
    log();
    await compareWithIndependentReference(finalResults, inputData);
    log();
    if (provider === "wasm") {
      log("[Correctness] WASM passed the independent reference and is the browser CPU baseline for accelerator comparisons.");
    } else {
      log("[Correctness] Creating a separate WASM reference session…");
      referenceSession = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
        preferredOutputLocation: "cpu",
      });
      validateModelContract(referenceSession);
      referenceResults = await referenceSession.run(feeds);
      await compareWithWasm(finalResults, referenceResults);
      log();
    }

    if (provider === "webgpu" && ui.profile.checked) {
      // Profiling readback resolves asynchronously after submitted GPU work.
      await new Promise(resolve => setTimeout(resolve, 0));
      if (profilingSummary.events > 0) {
        log(`[Proof] ${profilingSummary.events} WebGPU kernel event(s), ${profilingSummary.programs.size} unique program(s), were profiled.`);
      } else {
        log("[Proof] No timestamp-query event was returned. Correctness passed, but use chrome://gpu to verify hardware acceleration.");
      }
    } else if (provider === "webnn") {
      log(`[Proof] The requested ${deviceType} MLContext, WebNN session, and WASM parity succeeded.`);
      log("[Proof] WebNN does not expose a standard API for confirming its final native CPU/GPU/NPU backend.");
    }

    log(`PASS: ${provider.toUpperCase()} local inference and output validation completed.`);
    if (allowFallback) {
      log("NOTE: fallback was enabled; enable verbose logs/profiling to measure how much of your own model stayed on the requested EP.");
    }
    setStatus(`${provider.toUpperCase()} PASS`, "ok");
  } finally {
    if (referenceResults) await disposeResults(referenceResults);
    if (finalResults) await disposeResults(finalResults);
    if (feeds) {
      for (const tensor of Object.values(feeds)) tensor?.dispose?.();
    }
    if (referenceSession) await referenceSession.release();
    if (session) await session.release();
    ui.run.disabled = false;
  }
}

function updateProviderControls() {
  ui.device.disabled = ui.provider.value !== "webnn";
  ui.profile.disabled = ui.provider.value !== "webgpu";
  ui.fallback.disabled = ui.provider.value === "wasm";
}

function initialize() {
  const params = new URLSearchParams(location.search);
  const provider = params.get("ep");
  const device = params.get("device");
  if (provider && VALID_PROVIDERS.has(provider)) ui.provider.value = provider;
  if (device && VALID_WEBNN_DEVICES.has(device)) ui.device.value = device;
  if (params.has("iterations")) ui.iterations.value = params.get("iterations");
  ui.fallback.checked = params.get("fallback") === "1";
  ui.profile.checked = params.get("profile") !== "0";

  ui.secure.textContent = globalThis.isSecureContext ? "yes" : "no";
  ui.isolated.textContent = globalThis.crossOriginIsolated ? "yes" : "no";
  ui.webgpuApi.textContent = navigator.gpu ? "available" : "unavailable";
  ui.webnnApi.textContent = navigator.ml ? "available" : "unavailable";
  updateProviderControls();

  ui.provider.addEventListener("change", updateProviderControls);
  ui.run.addEventListener("click", () => {
    runDemo().catch(error => {
      console.error(error);
      log();
      log(`FAIL: ${error.name || "Error"}: ${error.message || error}`);
      log("Open DevTools (F12), then follow the bilingual README troubleshooting flow.");
      setStatus("Failed — see log", "bad");
      ui.run.disabled = false;
    });
  });

  if (params.get("autorun") === "1") ui.run.click();
}

initialize();
