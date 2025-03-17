# Three Steps to Launch the Qualcomm NPU-HTP

## Overview
This guide will walk you through the steps to set up the Qualcomm NPU-HTP on your device. The process involves obtaining necessary libraries from your device and the official QNN SDK, and placing them in the appropriate directories.

---

## Step 1: Get Required Libraries from Your Device
To run the Qualcomm NPU-HTP, connect your mobile device and execute the following commands in the Terminal to retrieve the required libraries.

**Note:** These libraries are verified for Snapdragon 8Gen1, 8Gen2, 8Gen3; other devices may need different libraries.

### Execute the Commands in Terminal:
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

## Step 2: Get Required Libraries from the Official QNN SDK

You need additional libraries from the official Qualcomm AI Engine Direct SDK `(QNN SDK, demo version: 2.31.*)`. Download the SDK from the official website, and locate the libraries in the `2.31.*/lib/aarch64-android` directory. Ensure the SDK version matches the one used for compiling the `libonnxruntime.so` library.

**Note:** To obtain the latest SDK version, you must use the `Qualcomm® Package Manager (QPM3)`; otherwise, you will receive another version via the direct download link.<br> 
**Note:** Find your device from [here](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices).

### 8Gen1 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV69Skel.so`
5. `libQnnHtpV69Stub.so`
6. `libQnnSystem.so`

### 8Gen2 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`

### 8Gen3 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV75Skel.so`
5. `libQnnHtpV75Stub.so`
6. `libQnnSystem.so`


---

## Step 3: Organize Libraries

Use 8Gen2 as example. Place all the required libraries in both the `assets` and `libs/arm64-v8a` folders.

### List of Libraries:
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
19. `libonnxruntime.so` (libs/arm64-v8a folder only)
20. `libomp.so` (libs/arm64-v8a folder only)

---

Following these steps will ensure the necessary setup for running the Qualcomm NPU-HTP on a Snapdragon 8Gen2 device.

