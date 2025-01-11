# CUDA Execution Provider ONNX Runtime Installation Guide

This guide provides step-by-step instructions for installing the NVIDIA CUDA Toolkit, GPU drivers, and ONNX Runtime with CUDA acceleration. Please follow the steps carefully to ensure a successful setup.

---

## Step 1: Remove Existing NVIDIA Software

To avoid conflicts between different versions, remove all existing NVIDIA software from your system.

```bash
# Remove NVIDIA drivers and libraries
sudo apt-get --purge remove cuda-* nvidia-* gds-tools-* libcublas-* libcufft-* libcufile-* libcurand-* libcusolver-* libcusparse-* libnpp-* libnvidia-* libnvjitlink-* libnvjpeg-* nsight* nvidia-* libnvidia-* libcudnn7* libcudnn8* libcudnn9*

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

# Install cuDNN libraries
sudo apt-get -y install cudnn9-cuda-12
# or others version:
sudo apt-get -y install cudnn8-cuda-11
```

---

## Step 4: Set Environment Variables
---
### Linux OS: ###
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
### Windows OS: ###
To achieve the same effect on Windows, where environment variables for CUDA need to be set, you can follow these steps:

1. **Open Environment Variables:**
   - Right-click on 'This PC' or 'My Computer' on your desktop or in File Explorer.
   - Select 'Properties'.
   - Click on 'Advanced system settings' on the left sidebar.
   - In the 'System Properties' window, go to the 'Advanced' tab and click on 'Environment Variables'.

2. **Set Environment Variables:**
   - In the 'Environment Variables' window, you have two sections: 'User variables' and 'System variables'. You can choose to set these for your user account only or for the entire system.
   
   - **Add/Edit CUDA Paths:**
     - **LD_LIBRARY_PATH**: This variable is not typically used on Windows, so you can skip it. Windows uses the PATH variable for similar purposes.
     
     - **PATH**: Find the 'Path' variable in the 'System variables' section, select it, and click 'Edit'. Add the following paths if they are not already present:
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin` (replace `vX.X` with your CUDA version)
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp`
     
     - **CUDA_HOME**: Click 'New' under 'System variables' and add a new variable:
       - Variable name: `CUDA_HOME`
       - Variable value: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`

3. **Apply Changes:**
   - Click 'OK' to close each dialog box.

4. **Verify Changes:**
   - Open a new Command Prompt and type `echo %PATH%` to check if the CUDA paths have been added successfully.
   - You can also type `echo %CUDA_HOME%` to check the CUDA_HOME variable.

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

## Step 7: Install ONNX Runtime with NVIDIA-GPU Support

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


