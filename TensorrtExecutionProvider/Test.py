import torch
import torch.nn as nn
import onnxruntime as ort
import time
import numpy as np
from onnxslim import slim


# ================================
# Step 1: Define a Simple PyTorch Model
# ================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4096, 2048, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024, dtype=torch.float32)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ================================
# Step 2: Export the Model to ONNX Format
# ================================
model = SimpleNet()
model.eval()
input_tensor = torch.randn(32, 4096, dtype=torch.float32)
onnx_file_path_cpu = "simplenet_cpu.onnx"
onnx_file_path_gpu = "simplenet_gpu.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,
    (input_tensor,),
    onnx_file_path_cpu,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    opset_version=17
)
print(f"ONNX CPU model exported to {onnx_file_path_cpu}")

# Convert the model to float16 format
slim(
    model=onnx_file_path_cpu,
    output_model=onnx_file_path_gpu,
    no_shape_infer=False,
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False,
    dtype="fp16"
)
print(f"ONNX GPU model exported to {onnx_file_path_gpu}")


# ================================
# Step 3: Benchmark Function for ONNX Runtime
# ================================
def benchmark_onnx(model_path, input_data, provider, num_runs=100):
    print(f"\nRunning benchmark on {provider}...")

    # Create ONNX Runtime session with options
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 3
    session_opts.inter_op_num_threads = 0
    session_opts.intra_op_num_threads = 0
    session_opts.enable_cpu_mem_arena = True
    session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")

    # Prepare input data based on the execution provider
    if "CPUExecutionProvider" in provider:
        input_data = input_data.astype(np.float32)
        dtype_info = "(float32)"
        provider_options = None
    else:
        input_data = input_data.astype(np.float16)
        dtype_info = "(float16)"
        provider_options = [
            # TensorrtExecutionProvider
            {
                # Device and Compute Configuration
                'device_id': 0,
                # 'user_compute_stream': "",

                # Engine Caching and Compatibility
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './Cache',
                # 'customize engine cache prefix': "",
                'trt_engine_hw_compatible': True,

                # Precision and Performance
                'trt_max_workspace_size': 25769803776,
                'trt_fp16_enable': True,
                'trt_int8_enable': False,               # For fine-tune
                # 'trt_int8_calibration_table_name': "",
                'trt_int8_use_native_calibration_table': False,
                'trt_build_heuristics_enable': True,
                'trt_sparsity_enable': True,
                'trt_dla_enable': True,
                'trt_dla_core': 0,

                # Subgraph and Graph Optimization
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1,
                'trt_dump_subgraphs': False,
                'trt_force_sequential_engine_build': True,

                # Advanced Configuration and Profiling
                'trt_context_memory_sharing_enable': True,
                'trt_layer_norm_fp32_fallback': False,
                'trt_cuda_graph_enable': False,         # Set to '0' to avoid potential errors when enabled.
                'trt_builder_optimization_level': 5,    # 0 ~ 5
                'trt_auxiliary_streams': -1,            # Set 0 for optimal memory usage
                'trt_detailed_build_log': False,

                # Timing cache
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': './Cache',
                'trt_force_timing_cache': True,

                # Dynamic Shape Profiling, The format of the profile shapes is input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...
                # 'trt_profile_min_shapes': "",
                # 'trt_profile_max_shapes': "",
                # 'trt_profile_opt_shapes': "",
            },

            # CUDAExecutionProvider
            {
                'device_id': 0,
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2 GB
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
                'enable_cuda_graph': '0'                # Set to '0' to avoid potential errors when enabled.
            }
        ]

    session = ort.InferenceSession(model_path, providers=provider, sess_options=session_opts, provider_options=provider_options)
    ort_inputs = {session.get_inputs()[0].name: input_data}

    # Warm-up
    for _ in range(10):
        _ = session.run(None, ort_inputs)

    # Measure inference time
    if "CPUExecutionProvider" in provider:
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, ort_inputs)
        end_time = time.time()
    else:
        ort_inputs = ort.OrtValue.ortvalue_from_numpy(input_data, 'cuda', 0)
        io_binding = session.io_binding()
        io_binding.bind_input(
            name='input',
            device_type=ort_inputs.device_name(),
            device_id=0,
            element_type=input_data.dtype,
            shape=ort_inputs.shape(),
            buffer_ptr=ort_inputs.data_ptr()
        )
        io_binding.bind_output('output', 'cuda')
        start_time = time.time()
        for _ in range(num_runs):
            session.run_with_iobinding(io_binding)
            _ = ort.OrtValue.numpy(io_binding.get_outputs()[0])
        end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time on {provider} {dtype_info}: {avg_time:.6f} seconds per batch")


# ================================
# Step 4: Run Benchmarks
# ================================
input_tensor_numpy = input_tensor.numpy()

# Benchmark on CPU
benchmark_onnx(onnx_file_path_cpu, input_tensor_numpy, ["CPUExecutionProvider"])

# Benchmark on GPU (if available)
if ort.get_device() == "GPU":
    benchmark_onnx(onnx_file_path_gpu, input_tensor_numpy, ["TensorrtExecutionProvider", "CUDAExecutionProvider"])
else:
    print("CUDA is not available for ONNX Runtime.")
