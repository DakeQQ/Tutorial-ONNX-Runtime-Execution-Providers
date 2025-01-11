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
    session = ort.InferenceSession(model_path, providers=provider, sess_options=session_opts)

    # Prepare input data based on the execution provider
    if "CPUExecutionProvider" in provider:
        input_data = input_data.astype(np.float32)
        dtype_info = "(float32)"
    else:
        input_data = input_data.astype(np.float16)
        dtype_info = "(float16)"

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
    benchmark_onnx(onnx_file_path_gpu, input_tensor_numpy, ["CUDAExecutionProvider"])
else:
    print("CUDA is not available for ONNX Runtime.")
