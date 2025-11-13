# Qualcomm NPU-HTPを起動するための3つのステップ

## 概要
このガイドでは、Qualcomm NPU-HTPをデバイスにセットアップする手順を説明します。このプロセスには、必要なライブラリをデバイスおよび公式のQNN SDKから取得し、それらを適切なディレクトリに配置することが含まれます。

---

## ステップ1: デバイスから必要なライブラリを取得する
Qualcomm NPU-HTPを実行するには、モバイルデバイスを接続し、以下のコマンドをターミナルで実行して必要なライブラリを取得します。

**注意:** これらのライブラリはSnapdragon 8Gen1、8Gen2、8Gen3に対して確認済みです。他のデバイスでは異なるライブラリが必要な場合があります。

### ターミナルでコマンドを実行:
```bash
cd ~/Downloads/YOLO_Depth_Drivable_Qualcomm_NPU

adb pull /vendor/lib64/libcdsprpc.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/libvmmem.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhidlbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhardware.so ./app/libs/arm64-v8a
adb pull /system/lib64/libutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libcutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdmabufheap.so ./app/libs/arm64-v8a
adb pull /system/lib64/libc++.so ./app/libs/arm64-v8a
adb pull /system/lib64/libbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libvndksupport.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdl_android.so ./app/libs/arm64-v8a
adb pull /system/lib64/ld-android.so ./app/libs/arm64-v8a

adb pull /vendor/lib64/libcdsprpc.so ./app/src/main/assets
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/src/main/assets
adb pull /vendor/lib64/libvmmem.so ./app/src/main/assets
adb pull /system/lib64/libhidlbase.so ./app/src/main/assets
adb pull /system/lib64/libhardware.so ./app/src/main/assets
adb pull /system/lib64/libutils.so ./app/src/main/assets
adb pull /system/lib64/libcutils.so ./app/src/main/assets
adb pull /system/lib64/libdmabufheap.so ./app/src/main/assets
adb pull /system/lib64/libc++.so ./app/src/main/assets
adb pull /system/lib64/libbase.so ./app/src/main/assets
adb pull /system/lib64/libvndksupport.so ./app/src/main/assets
adb pull /system/lib64/libdl_android.so ./app/src/main/assets
adb pull /system/lib64/ld-android.so ./app/src/main/assets
```

---

## ステップ2: 公式QNN SDKから必要なライブラリを取得する

[Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) `(QNN SDK, demo version: 2.32.*)`から追加のライブラリが必要です。公式ウェブサイトからSDKをダウンロードし、`2.32.*/lib/aarch64-android/` & `2.32.*/lib/hexagon-v7*/unsigned/` ディレクトリにあるライブラリを見つけてください。SDKバージョンが`libonnxruntime.so`ライブラリをコンパイルする際に使用したものと一致していることを確認してください。

**注意:** 最新のSDKバージョンを取得するには、`Qualcomm® Package Manager (QPM3)`を使用する必要があります。そうでなければ、直接ダウンロードリンクから別のバージョンを受け取ることになります。<br> 
**注意:** デバイスを[こちら](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/QNN_general_overview.html)から見つけてください。

### 8Gen1 必要なライブラリ:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV69Skel.so`
5. `libQnnHtpV69Stub.so`
6. `libQnnSystem.so`

### 8Gen2 必要なライブラリ:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`

### 8Gen3 必要なライブラリ:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV75Skel.so`
5. `libQnnHtpV75Stub.so`
6. `libQnnSystem.so`

---

## ステップ3: ライブラリを整理する

8Gen2を例に使用します。必要なライブラリをすべて`assets`および`libs/arm64-v8a`フォルダに配置してください。

### ライブラリ一覧:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`
7. `libcdsprpc.so`
8. `vendor.qti.hardware.dsp@1.0.so`
9. `libvmmem.so`
10. `libhidlbase.so`
11. `libhardware.so`
12. `libutils.so`
13. `libcutils.so`
14. `libdmabufheap.so`
15. `libc++.so`
16. `libbase.so`
17. `libdl_android.so`
18. `ld-android.so`
19. `libonnxruntime.so` (libs/arm64-v8aフォルダのみ)
20. `libomp.so` (libs/arm64-v8aフォルダのみ)

---

これらのステップに従うことで、Snapdragon 8Gen2デバイス上でQualcomm NPU-HTPを実行するために必要なセットアップが完了します。
