# CUDA Execution Provider 在 ONNX Runtime 上的设置指南

本指南提供了安装 NVIDIA CUDA Toolkit, GPU 驱动程序和 ONNX Runtime-CUDA 加速的分步说明。请仔细按照以下步骤操作，以确保成功完成设置。

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
---
### Linux OS: ###

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

### Windows OS: ###

1. **打开环境变量：**
   - 在桌面或文件资源管理器中右键点击“此电脑”或“我的电脑”。
   - 选择“属性”。
   - 点击左侧栏中的“高级系统设置”。
   - 在“系统属性”窗口中，进入“高级”选项卡并点击“环境变量”。

2. **设置环境变量：**
   - 在“环境变量”窗口中，有两个部分：“用户变量”和“系统变量”。您可以选择仅为您的用户帐户设置这些变量，或者为整个系统设置。
   
   - **添加/编辑 CUDA 路径：**
     - **LD_LIBRARY_PATH**：这个变量通常不在 Windows 上使用，所以可以忽略。Windows 使用 PATH 变量实现类似功能。
     
     - **PATH**：在“系统变量”部分找到“Path”变量，选择它并点击“编辑”。如果路径尚未存在，添加以下路径：
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin`（将 `vX.X` 替换为您的 CUDA 版本）
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp`
     
     - **CUDA_HOME**：在“系统变量”下点击“新建”并添加一个新变量：
       - 变量名：`CUDA_HOME`
       - 变量值：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`

3. **应用更改：**
   - 点击“确定”关闭每个对话框。

4. **验证更改：**
   - 打开一个新的命令提示符并输入 `echo %PATH%` 检查 CUDA 路径是否已成功添加。
   - 您也可以输入 `echo %CUDA_HOME%` 检查 CUDA_HOME 变量。

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


