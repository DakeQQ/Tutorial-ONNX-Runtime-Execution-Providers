# CUDA Execution Provider 在 ONNX Runtime 上的设置指南

本指南提供了安装 NVIDIA CUDA Toolkit 和带 GPU 支持的 ONNX Runtime 的分步说明。请仔细按照以下步骤操作，以确保成功完成设置。

---

## 第 1 步：删除现有的 NVIDIA 软件

为了避免不同版本之间的冲突，请从系统中删除所有现有的 NVIDIA 软件。

```bash
# 删除 NVIDIA 驱动和库文件
sudo apt-get --purge remove cuda-* nvidia-* gds-tools-* libcublas-* libcufft-* libcufile-* libcurand-* libcusolver-* libcusparse-* libnpp-* libnvidia-* libnvjitlink-* libnvjpeg-* nsight* nvidia-* libnvidia-* libcudnn7* libcudnn8* libcudnn9*

# 删除旧版本的 CUDA
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

# 清理卸载残留
sudo apt-get autoremove
sudo apt-get autoclean

# 删除 CUDA 目录
sudo rm -rf /usr/local/cuda*

# 从 dpkg 中移除
sudo dpkg -r cuda
sudo dpkg -r $(dpkg -l | grep '^ii  cudnn' | awk '{print $2}')
```

---

## 第 2 步：下载 CUDA Toolkit

访问 [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)，选择您要安装的版本。

**示例：** CUDA Toolkit 12.6

![显示 CUDA 选择过程的截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-02-42.png)

---

## 第 3 步：配置 CUDA Toolkit

按照网站上的说明配置您的系统。以下是 CUDA Toolkit 12.6 的示例。

![显示 CUDA Toolkit 安装程序的截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-17.png)
![显示安装程序说明的截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-35.png)

```bash
# CUDA Toolkit 安装程序
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# 驱动程序安装
sudo apt-get install -y nvidia-open
sudo apt-get install -y cuda-drivers

# 安装 cuDNN 库
sudo apt-get -y install cudnn9-cuda-12
# 或其他版本：
sudo apt-get -y install cudnn8-cuda-11
```

---

## 第 4 步：设置环境变量

将必要的环境路径添加到 `.bashrc` 文件中。

```bash
vi ~/.bashrc
```

将以下内容复制并粘贴到文件的底部：

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

输入`:wq`就可以保存并退出编辑器，然后激活环境：

```bash
source ~/.bashrc
```

---

## 第 5 步：验证 CUDA 安装

运行以下命令，验证系统是否识别 GPU：

```bash
nvidia-smi
```
如果成功，您将看到您的 GPU 详细信息。

![显示 GPU 详细信息的截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-47.png)

验证 CUDA 版本：

```bash
nvcc -V
```
![显示 CUDA 版本的截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-26.png)

---

## 第 6 步：删除冲突的 ONNX Runtime 包

为了避免冲突，请删除任何不支持 CUDA 的现有 ONNX Runtime 包。

```bash
pip uninstall onnxruntime-directml
pip uninstall onnxruntime-openvino
```

---

## 第 7 步：安装带 NVIDIA-GPU 支持的 ONNX Runtime

参考 [ONNX Runtime CUDA Execution Provider 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 了解兼容版本。以下是安装最新版本的示例：

```bash
pip install onnxruntime --upgrade
pip install onnxruntime-gpu --upgrade
```

---

## 第 8 步：运行测试脚本

运行 `Test.py` 脚本，基准测试 CUDA 执行提供程序与 CPU 执行提供程序的性能差异。

**预期输出：**

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['CUDAExecutionProvider']...
Average inference time on ['CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```


