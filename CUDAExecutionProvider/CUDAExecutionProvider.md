# CUDA and ONNX Runtime Installation Guide

This guide provides step-by-step instructions for installing the NVIDIA CUDA Toolkit and ONNX Runtime with GPU support. Follow the steps carefully to ensure a successful setup.

---

## Step 1: Remove Existing NVIDIA Software

To avoid conflicts between different versions, remove all existing NVIDIA software from your system.

```bash
# Remove NVIDIA drivers and libraries
sudo apt-get --purge remove cuda-* nvidia-* gds-tools-* libcublas-* libcufft-* libcufile-* libcurand-* libcusolver-* libcusparse-* libnpp-* libnvidia-* libnvjitlink-* libnvjpeg-* nsight* nvidia-* libnvidia-* libcudnn8*

# Remove older CUDA versions
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

# Cleanup uninstall
sudo apt-get autoremove
sudo apt-get autoclean

# Remove CUDA directories
sudo rm -rf /usr/local/cuda*

# Remove from dpkg
sudo dpkg -r cuda
sudo dpkg -r $(dpkg -l | grep '^ii  cudnn' | awk '{print $2}')
```

---

## Step 2: Download CUDA Toolkit

Visit the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and select the version you want to install.

**Example:** CUDA Toolkit 12.6

![Screenshot showing CUDA selection process](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-02-42.png)

---

## Step 3: Configure the CUDA Toolkit

Follow the instructions provided on the website to configure your system. Below is an example for CUDA Toolkit 12.6.
![Screenshot showing CUDA Toolkit Installer](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-17.png)
![Screenshot showing Installer Instructions](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-35.png)

```bash
# CUDA Toolkit Installer
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Driver Installer
sudo apt-get install -y nvidia-open
sudo apt-get install -y cuda-drivers
```

---

## Step 4: Set Environment Variables

Add the necessary environment paths to your `.bashrc` file.

```bash
vi ~/.bashrc
```

Copy the following lines and paste them at the bottom of the file:

```bash
if [ $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
else
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
fi

if [ $PATH ]; then
    export PATH=$PATH:/usr/local/cuda/bin
else
    export PATH=/usr/local/cuda/bin
fi

if [ $CUDA_HOME ]; then
    export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
else
    export CUDA_HOME=/usr/local/cuda
fi
```

Save and exit the editor by typing `:wq`, then activate the environment:

```bash
source ~/.bashrc
```

---

## Step 5: Verify CUDA Installation

Run the following commands to verify that your system recognizes the GPU:

```bash
nvidia-smi
```
If successful, you will see your GPU details.
![Screenshot showing GPU Details](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-47.png)


This should display the CUDA version.
```bash
nvcc -V
```
![Screenshot showing CUDA Version](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-26.png)

---

## Step 6: Remove Conflicting ONNX Runtime Packages

To avoid conflicts, remove any existing ONNX Runtime packages that are not CUDA-compatible.

```bash
pip uninstall onnxruntime-directml
pip uninstall onnxruntime-openvino
```

---

## Step 7: Install ONNX Runtime with GPU Support

Refer to the [ONNX Runtime CUDA Execution Provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for the compatible versions. Below is an example of installing the latest version:

```bash
pip install onnxruntime --upgrade
pip install onnxruntime-gpu --upgrade
```

---

## Step 8: Run a Test Script

Run the `Test.py` script to benchmark the CUDA Execution Provider against the CPU Execution Provider.

**Expected Output:**

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['CUDAExecutionProvider']...
Average inference time on ['CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```


