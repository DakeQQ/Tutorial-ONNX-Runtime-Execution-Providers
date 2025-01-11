Here is the translated document in Japanese:

# ONNX Runtime のための TensorRT Execution Provider セットアップガイド

このガイドでは、ONNX Runtime に TensorRT Execution Provider をセットアップする手順を説明します。成功するインストールのために、これらの指示に注意深く従ってください。

## **ステップ 1: CUDA Execution Provider の準備**
`CUDAExecutionProvider.md` ガイドの指示に従って、環境が CUDA Execution Provider で実行できるように準備されていることを確認してください。

---

## **ステップ 2: TensorRT パッケージのダウンロード**
1. [NVIDIA TensorRT 開発者ページ](https://developer.nvidia.com/tensorrt)を訪問します。まず、アカウントを作成してログインする必要があります。
2. **Download Now** をクリックします。

![スクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-34-39.png)

---

## **ステップ 3: TensorRT パッケージの選択とダウンロード**
1. インストールしたい TensorRT のバージョンを選択します。
2. チェックボックスを選択してライセンス条項に同意します。
3. ダウンロードしたいパッケージをクリックします。ダウンロードが自動的に開始されます。

例えば、以下を使用します：
- **Ubuntu 24.04**
- **TensorRT 10.7**
- **CUDA 12.6**

![スクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-35-08.png)
![スクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-36-40.png)
![スクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2013-37-07.png)
---

## **ステップ 4: TensorRT パッケージのインストール**
ターミナル（CMD または PowerShell）を開き、以下のコマンドを入力します：

```bash
os="ubuntu2404"
tag="10.7.0-cuda-12.6"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
```

---

## **ステップ 5: インストールコマンドの実行**
以下のコマンドを実行して TensorRT をインストールし、Python パッケージをアップグレードします：

```bash
sudo apt-get install tensorrt
pip install tensorrt --upgrade
```

---

## **ステップ 6: TensorRT インストールの確認**
TensorRT が正しくインストールされているか確認するには、以下を実行します：

```bash
dpkg-query -W tensorrt
```
成功すると、ターミナルウィンドウに次のように表示されます：
```bash
tensorrt	10.7.0.23-1+cuda12.6
```

---

## **ステップ 7: テストスクリプトの実行**
`Test.py` スクリプトを実行して、すべてが正常に動作していることを確認します。出力は次のようになります：

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['TensorrtExecutionProvider', 'CUDAExecutionProvider']...
Average inference time on ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```

同様の出力が表示される場合は、セットアップが完了です！
