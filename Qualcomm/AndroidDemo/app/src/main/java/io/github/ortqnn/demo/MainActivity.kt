package io.github.ortqnn.demo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.qnnpluginep.getEpName as getQnnPluginEpName
import ai.onnxruntime.qnnpluginep.getLibraryPath as getQnnPluginEpLibraryPath
import android.app.Activity
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.system.Os
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs

private const val QNN_EP_REGISTRATION_NAME = "QNNExecutionProvider"
private const val BATCH_SIZE = 32
private const val INPUT_SIZE = 64
private const val WARMUP_RUNS = 3
private const val TIMED_RUNS = 20

class MainActivity : Activity() {
    private enum class Backend(
        val qnnName: String,
        val title: String,
        val modelAsset: String,
        val tolerance: Float,
    ) {
        CPU("cpu", "QNN CPU reference backend", "qnn_smoke_fp32.onnx", 3e-3f),
        GPU("gpu", "QNN GPU / Adreno backend", "qnn_smoke_fp32.onnx", 5e-3f),
        HTP("htp", "QNN HTP / NPU backend", "qnn_smoke_qdq.onnx", 8e-2f),
    }

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var environment: OrtEnvironment
    private lateinit var statusView: TextView
    private val buttonByBackend = mutableMapOf<Backend, Button>()
    private var pluginRegistered = false

    override fun onCreate(savedInstanceState: Bundle?) {
        // These variables must exist before OrtEnvironment and the plugin are
        // initialized. The Maven QNN runtime extracts backend/stub/skel files to
        // nativeLibraryDir; HTP's loader searches that directory through ADSP_LIBRARY_PATH.
        Os.setenv("ADSP_LIBRARY_PATH", applicationInfo.nativeLibraryDir, true)
        // Some Android builds expose QNN only through this CPU-class registration
        // device; backend_type still selects the actual CPU, GPU, or HTP backend.
        Os.setenv("ORT_QNN_ENABLE_CPU_BACKEND", "1", true)
        super.onCreate(savedInstanceState)

        buildInterface()
        initializeQnnPlugin()

        intent.getStringExtra("backend")?.lowercase(Locale.ROOT)?.let { requested ->
            Backend.entries.firstOrNull { it.qnnName == requested }?.let(::startBackend)
        }
    }

    private fun buildInterface() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(22), dp(22), dp(22))
            setBackgroundColor(Color.rgb(245, 247, 251))
        }

        root.addView(TextView(this).apply {
            text = "Qualcomm QNN · ONNX Runtime"
            textSize = 25f
            setTextColor(Color.rgb(17, 36, 64))
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "Strict backend proof · GPU availability depends on device and driver"
            textSize = 15f
            setTextColor(Color.rgb(70, 84, 105))
            setPadding(0, dp(5), 0, dp(18))
        })

        val buttonSpecs = listOf(
            Triple(Backend.CPU, "Run QNN CPU", Color.rgb(64, 92, 170)),
            Triple(Backend.GPU, "Try QNN GPU", Color.rgb(25, 135, 120)),
            Triple(Backend.HTP, "Run QNN NPU / HTP", Color.rgb(177, 82, 48)),
        )
        buttonSpecs.forEach { (backend, label, color) ->
            val button = Button(this).apply {
                text = label
                textSize = 15f
                isAllCaps = false
                setTextColor(Color.WHITE)
                setBackgroundColor(color)
                setOnClickListener { startBackend(backend) }
            }
            buttonByBackend[backend] = button
            root.addView(
                button,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    dp(54),
                ).apply { bottomMargin = dp(10) },
            )
        }

        root.addView(TextView(this).apply {
            text = "A PASS means session.disable_cpu_ep_fallback=1 was active, the " +
                "selected QNN backend accepted the complete static graph, inference ran, " +
                "and output matched an independent ORT CPU reference."
            textSize = 13f
            setTextColor(Color.rgb(68, 76, 91))
            setPadding(0, dp(8), 0, dp(12))
        })

        statusView = TextView(this).apply {
            text = "Initializing QNN plugin…"
            textSize = 13f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextColor(Color.rgb(20, 31, 48))
            setPadding(dp(16), dp(16), dp(16), dp(16))
            setBackgroundColor(Color.WHITE)
            gravity = Gravity.START
            setTextIsSelectable(true)
        }
        root.addView(
            statusView,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun initializeQnnPlugin() {
        try {
            environment = OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            )
            environment.registerExecutionProviderLibrary(
                QNN_EP_REGISTRATION_NAME,
                getQnnPluginEpLibraryPath(),
            )
            pluginRegistered = true
            val qnnDevices = environment.epDevices.filter {
                it.epName == getQnnPluginEpName()
            }
            if (qnnDevices.isEmpty()) {
                error("The plugin registered but exposed no QNN device.")
            }
            val cpuBackendAvailable = File(
                applicationInfo.nativeLibraryDir,
                "libQnnCpu.so",
            ).isFile
            if (!cpuBackendAvailable) {
                buttonByBackend[Backend.CPU]?.apply {
                    isEnabled = false
                    alpha = 0.55f
                    text = "QNN CPU needs QAIRT libQnnCpu.so"
                }
            }

            val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
            statusView.text = buildString {
                appendLine("READY")
                appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                appendLine("SoC: $soc")
                appendLine("ABI: ${Build.SUPPORTED_ABIS.joinToString()}")
                appendLine("Native library dir: ${applicationInfo.nativeLibraryDir}")
                appendLine("Plugin: ${getQnnPluginEpLibraryPath()}")
                appendLine("QNN CPU backend packaged: $cpuBackendAvailable")
                appendLine("QNN EP registration devices:")
                qnnDevices.forEach { appendLine("  • ep=${it.epName}, type=${it.device.type}") }
                append("Choose one backend above.")
            }
        } catch (error: Throwable) {
            setButtonsEnabled(false)
            statusView.text = failureMessage("QNN plugin initialization", error)
        }
    }

    private fun startBackend(backend: Backend) {
        if (!pluginRegistered) return
        setButtonsEnabled(false)
        statusView.text = "Preparing ${backend.title}…"
        executor.execute {
            val message = try {
                runBackend(backend)
            } catch (error: Throwable) {
                failureMessage(backend.title, error)
            }
            runOnUiThread {
                statusView.text = message
                setButtonsEnabled(true)
            }
        }
    }

    private fun runBackend(backend: Backend): String {
        val model = assets.open(backend.modelAsset).use { it.readBytes() }
        val input = deterministicInput()
        val qnnDevices = environment.epDevices.filter {
            it.epName == getQnnPluginEpName()
        }
        check(qnnDevices.isNotEmpty()) { "No QNN EP device remains available." }

        OnnxTensor.createTensor(environment, input).use { inputTensor ->
            val reference = OrtSession.SessionOptions().use { referenceOptions ->
                environment.createSession(model, referenceOptions).use { session ->
                    runOnce(session, inputTensor)
                }
            }

            val providerOptions = linkedMapOf(
                "backend_type" to backend.qnnName,
                "offload_graph_io_quantization" to "0",
            )
            if (backend == Backend.HTP) {
                providerOptions["htp_performance_mode"] = "burst"
                providerOptions["htp_graph_finalization_optimization_mode"] = "3"
            }

            return OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(1)
                options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                options.addExecutionProvider(qnnDevices, providerOptions)

                environment.createSession(model, options).use { session ->
                    repeat(WARMUP_RUNS) { runOnce(session, inputTensor) }
                    val samples = DoubleArray(TIMED_RUNS)
                    var actual = FloatArray(0)
                    repeat(TIMED_RUNS) { index ->
                        val start = System.nanoTime()
                        actual = runOnce(session, inputTensor)
                        samples[index] = (System.nanoTime() - start) / 1_000_000.0
                    }

                    check(actual.size == reference.size) {
                        "Output size ${actual.size} != reference size ${reference.size}"
                    }
                    var maxError = 0f
                    for (index in actual.indices) {
                        maxError = maxOf(maxError, abs(actual[index] - reference[index]))
                    }
                    check(maxError <= backend.tolerance) {
                        "Maximum error $maxError exceeds ${backend.tolerance}"
                    }
                    samples.sort()
                    val median = samples[samples.size / 2]
                    val mean = samples.average()
                    val min = actual.minOrNull() ?: Float.NaN
                    val max = actual.maxOrNull() ?: Float.NaN

                    buildString {
                        appendLine("PASS · ${backend.title}")
                        appendLine()
                        appendLine("backend_type=${backend.qnnName}")
                        appendLine("model=${backend.modelAsset}")
                        appendLine("shape=[$BATCH_SIZE,$INPUT_SIZE] (fully static)")
                        appendLine("session.disable_cpu_ep_fallback=1")
                        appendLine("offload_graph_io_quantization=0")
                        appendLine("QNN registration devices=${qnnDevices.size}")
                        appendLine("warmup=$WARMUP_RUNS, measured=$TIMED_RUNS")
                        appendLine("median=${"%.3f".format(Locale.US, median)} ms")
                        appendLine("mean=${"%.3f".format(Locale.US, mean)} ms")
                        appendLine("max |QNN−CPU|=${"%.7f".format(Locale.US, maxError)}")
                        appendLine("output range=[${"%.5f".format(Locale.US, min)}, ${"%.5f".format(Locale.US, max)}]")
                        appendLine()
                        append("Strict session creation and execution would fail if any graph node needed ORT CPU fallback.")
                    }
                }
            }
        }
    }

    private fun runOnce(session: OrtSession, inputTensor: OnnxTensor): FloatArray {
        val inputName = session.inputNames.first()
        session.run(mapOf(inputName to inputTensor)).use { result ->
            @Suppress("UNCHECKED_CAST")
            val output = result[0].value as Array<FloatArray>
            val flattened = FloatArray(output.sumOf { it.size })
            var offset = 0
            output.forEach { row ->
                row.copyInto(flattened, offset)
                offset += row.size
            }
            return flattened
        }
    }

    private fun deterministicInput(): Array<FloatArray> =
        Array(BATCH_SIZE) { batch ->
            FloatArray(INPUT_SIZE) { feature ->
                val index = batch * INPUT_SIZE + feature
                ((index % 257) - 128) / 128.0f
            }
        }

    private fun failureMessage(stage: String, error: Throwable): String = buildString {
        appendLine("FAIL · $stage")
        appendLine("${error::class.java.simpleName}: ${error.message}")
        appendLine()
        appendLine("Check Logcat and the guide's troubleshooting table.")
        if (stage.contains("GPU")) {
            appendLine("QNN GPU is device/driver-dependent. If Logcat reports PLATFORM_NOT_SUPPORTED, use HTP on this device.")
        }
        appendLine("Common causes: non-Snapdragon device, stale OEM driver, unsupported model node, missing libcdsprpc access, or mixed QNN package versions.")
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        val cpuBackendAvailable = File(
            applicationInfo.nativeLibraryDir,
            "libQnnCpu.so",
        ).isFile
        buttonByBackend.forEach { (backend, button) ->
            val backendEnabled = enabled && (backend != Backend.CPU || cpuBackendAvailable)
            button.isEnabled = backendEnabled
            button.alpha = if (backendEnabled) 1.0f else 0.55f
        }
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()

    override fun onDestroy() {
        executor.shutdown()
        val workerStopped = runCatching {
            executor.awaitTermination(5, TimeUnit.SECONDS)
        }.getOrDefault(false)
        // A plugin must never be unloaded while a session still references it.
        // If Android destroys the Activity during a run, leave final unloading
        // to process teardown rather than risking a native use-after-unload.
        if (pluginRegistered && workerStopped) {
            runCatching {
                environment.unregisterExecutionProviderLibrary(QNN_EP_REGISTRATION_NAME)
            }
            pluginRegistered = false
        }
        super.onDestroy()
    }
}
