# CUDA Execution Provider と ONNX Runtime インストールガイド

このガイドは、NVIDIA CUDA Toolkit と GPU サポート付きの ONNX Runtime をインストールするための手順を説明します。以下の手順に従って、正確にセットアップしてください。

---

## ステップ 1: 既存の NVIDIA ソフトウェアを削除

異なるバージョン間の競合を避けるため、システムから既存の NVIDIA ソフトウェアをすべて削除します。

```bash
# NVIDIA ドライバーとライブラリを削除
sudo apt-get --purge remove cuda-* nvidia-* gds-tools-* libcublas-* libcufft-* libcufile-* libcurand-* libcusolver-* libcusparse-* libnpp-* libnvidia-* libnvjitlink-* libnvjpeg-* nsight* nvidia-* libnvidia-* libcudnn7* libcudnn8* libcudnn9*

# 古い CUDA バージョンを削除
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"

# アンインストールのクリーンアップ
sudo apt-get autoremove
sudo apt-get autoclean

# CUDA ディレクトリを削除
sudo rm -rf /usr/local/cuda*

# dpkg から削除
sudo dpkg -r cuda
sudo dpkg -r $(dpkg -l | grep '^ii  cudnn' | awk '{print $2}')
```

---

## ステップ 2: CUDA Toolkit をダウンロード

[NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) にアクセスし、インストールしたいバージョンを選択します。

**例:** CUDA Toolkit 12.6

![CUDA 選択プロセスを示すスクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-02-42.png)

---

## ステップ 3: CUDA Toolkit の構成

ウェブサイトに記載されている指示に従って、システムを構成します。以下は CUDA Toolkit 12.6 の例です。

![CUDA Toolkit インストーラーを示すスクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-17.png)
![インストーラー手順を示すスクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-03-35.png)

```bash
# CUDA Toolkit インストーラー
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# ドライバーインストーラー
sudo apt-get install -y nvidia-open
sudo apt-get install -y cuda-drivers

# cuDNN ライブラリのインストール
sudo apt-get -y install cudnn9-cuda-12
# 他のバージョンの場合:
sudo apt-get -y install cudnn8-cuda-11
```

---

## ステップ 4: 環境変数を設定
---
### Linux OS: ###
必要な環境パスを `.bashrc` ファイルに追加します。

```bash
vi ~/.bashrc
```

以下の行をコピーし、ファイルの末尾に貼り付けます。

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

エディターを保存して終了するには、`:wq` と入力し、環境を有効化します。

```bash
source ~/.bashrc
```
---
### Windows OS: ###

1. **環境変数を開く:**
   - デスクトップまたはファイルエクスプローラーで「PC」または「マイコンピュータ」を右クリックします。
   - 「プロパティ」を選択します。
   - 左側のバーで「システムの詳細設定」をクリックします。
   - 「システムのプロパティ」ウィンドウで、「詳細設定」タブに移動し、「環境変数」をクリックします。

2. **環境変数を設定する:**
   - 「環境変数」ウィンドウには、「ユーザー変数」と「システム変数」の2つのセクションがあります。これらをユーザーアカウントのみに設定するか、システム全体に設定することができます。
   
   - **CUDA パスを追加/編集:**
     - **LD_LIBRARY_PATH**: この変数は通常Windowsで使用されないため、スキップできます。Windowsは類似の機能にPATH変数を使用します。
     
     - **PATH**: 「システム変数」セクションで「Path」変数を見つけて選択し、「編集」をクリックします。存在しない場合は、次のパスを追加します：
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin`（`vX.X`をCUDAのバージョンに置き換えます）
       - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp`
     
     - **CUDA_HOME**: 「システム変数」で「新規」をクリックし、新しい変数を追加します：
       - 変数名：`CUDA_HOME`
       - 変数値：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`

3. **変更を適用する:**
   - 「OK」をクリックして各ダイアログボックスを閉じます。

4. **変更を確認する:**
   - 新しいコマンドプロンプトを開き、`echo %PATH%`と入力してCUDAパスが正常に追加されたか確認します。
   - また、`echo %CUDA_HOME%`と入力してCUDA_HOME変数を確認することもできます。

---

## ステップ 5: CUDA インストールの確認

次のコマンドを実行して、システムが GPU を認識していることを確認します。

```bash
nvidia-smi
```

成功すると、GPU の詳細が表示されます。
![GPU 詳細を示すスクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-47.png)

CUDA バージョンを表示するには、次のコマンドを実行します。

```bash
nvcc -V
```

![CUDA バージョンを示すスクリーンショット](https://github.com/DakeQQ/Tutorial-ONNX-Runtime-Execution-Providers/blob/main/screenshots/Screenshot%20from%202025-01-11%2012-27-26.png)

---

## ステップ 6: 競合する ONNX Runtime パッケージを削除

競合を避けるため、CUDA 非対応の既存の ONNX Runtime パッケージを削除します。

```bash
pip uninstall onnxruntime-directml
pip uninstall onnxruntime-openvino
```

---

## ステップ 7: NVIDIA-GPU サポート付き ONNX Runtime のインストール

[ONNX Runtime CUDA Execution Provider のドキュメント](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)を参照して、互換性のあるバージョンを確認してください。以下は最新バージョンをインストールする例です。

```bash
pip install onnxruntime --upgrade
pip install onnxruntime-gpu --upgrade
```

---

## ステップ 8: テストスクリプトの実行

`Test.py` スクリプトを実行して、CUDA Execution Provider と CPU Execution Provider のベンチマークを行います。

**期待される出力:**

```python
Running benchmark on ['CPUExecutionProvider']...
Average inference time on ['CPUExecutionProvider'] (float32): 0.003056 seconds per batch

Running benchmark on ['CUDAExecutionProvider']...
Average inference time on ['CUDAExecutionProvider'] (float16): 0.000888 seconds per batch
```


