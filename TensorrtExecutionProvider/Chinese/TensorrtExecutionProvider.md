# TensorRT Execution Provider 在 ONNX Runtime 上的设置指南

本指南将引导您完成在 ONNX Runtime 上设置 TensorRT Execution Provider 的步骤。请仔细按照这些说明进行操作，以确保成功安装。

## **步骤 1: 准备 CUDA 执行提供程序**
请确保您的环境已准备好运行 CUDA 执行提供程序，具体步骤请参阅 `CUDAExecutionProvider.md` 指南。

---

## **步骤 2: 下载 TensorRT 包**
1. 访问 [NVIDIA TensorRT 开发者页面](https://developer.nvidia.com/tensorrt)。您需要先创建一个账户并登录。
2. 点击 **立即下载**。

![截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-34-39.png)

---

## **步骤 3: 选择并下载 TensorRT 包**
1. 选择您要安装的 TensorRT 版本。
2. 勾选同意许可条款。
3. 点击您要下载的包，下载将自动开始。

例如，我们使用：
- **Ubuntu 24.04**
- **TensorRT 10.7**
- **CUDA 12.6**

![截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-35-08.png)
![截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-36-40.png)
![截图](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-37-07.png)

---

## **步骤 4: 安装 TensorRT 包**
打开终端（CMD 或 PowerShell），输入以下命令：

```bash
os="ubuntu2404"
tag="10.7.0-cuda-12.6"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
```

---

## **步骤 5: 执行安装命令**
运行以下命令来安装 TensorRT 并升级 Python 包：

```bash
sudo apt-get install tensorrt
pip install tensorrt --upgrade
```

---

## **步骤 6: 验证 TensorRT 安装**
要检查 TensorRT 是否正确安装，请运行：

```bash
dpkg-query -W tensorrt
```
如果成功，您将在终端窗口中看到以下内容：
```bash
tensorrt	10.7.0.23-1+cuda12.6
```

---

## **步骤 7: 运行测试脚本**
执行 `Test.py` 脚本以确保一切正常工作。输出应如下所示：

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['TensorrtExecutionProvider', 'CUDAExecutionProvider']...
Average inference time on ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```

如果您看到类似的输出，您的设置就完成了！
