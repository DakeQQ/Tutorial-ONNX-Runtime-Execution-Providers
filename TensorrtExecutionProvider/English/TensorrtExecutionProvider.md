# TensorRT Execution Provider Setup Guide for ONNX Runtime

This guide will walk you through the steps to set up TensorRT Execution Provider on your ONNX Runtime. Follow these instructions carefully to ensure a successful installation.

## **Step 1: Prepare for CUDA Execution Provider**
Make sure your environment is ready to run on the CUDA Execution Provider by following the instructions in the `CUDAExecutionProvider.md` guide.

---

## **Step 2: Download TensorRT Packages**
1. Visit the [NVIDIA TensorRT Developer page](https://developer.nvidia.com/tensorrt).
2. Click **GET STARTED** and then **Download Now**.

![Screenshot](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-34-39.png)

---

## **Step 3: Select and Download the TensorRT Package**
1. Choose the version of TensorRT you want to install.
2. Agree to the license terms by selecting the check-box.
3. Click the package you want to download. The download will start automatically.

For example, let's use:
- **Ubuntu 24.04**
- **TensorRT 10.7**
- **CUDA 12.6**

![Screenshot](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-35-08.png)
![Screenshot](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-36-40.png)
![Screenshot](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-37-07.png)
---

## **Step 4: Install the TensorRT Package**
Open the Terminal (CMD or PowerShell) and enter the following commands:

```bash
os="ubuntu2404"
tag="10.7.0-cuda-12.6"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
```

---

## **Step 5: Execute the Installation Commands**
Run the following commands to install TensorRT and upgrade the Python package:

```bash
sudo apt-get install tensorrt
pip install tensorrt --upgrade
```

---

## **Step 6: Verify the TensorRT Installation**
To check if TensorRT is installed correctly, run:

```bash
dpkg-query -W tensorrt
```

---

## **Step 7: Run the Test Script**
Execute the `Test.py` script to ensure everything is working correctly. The output should look like this:

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['TensorrtExecutionProvider', 'CUDAExecutionProvider']...
Average inference time on ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```

If you see similar output, your setup is complete!


