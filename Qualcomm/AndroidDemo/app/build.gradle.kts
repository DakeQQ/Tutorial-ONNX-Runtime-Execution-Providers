plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.github.ortqnn.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.ortqnn.demo"
        minSdk = 27
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        // QNN's HTP loader must discover the backend/stub/skel libraries from
        // ApplicationInfo.nativeLibraryDir at runtime.
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    // QNN EP 2.5.0 supports ORT >=1.24.1 through the stable plugin ABI.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
    implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.5.0")

    // Public Maven runtime matching the source build's QAIRT 2.49 line. The AAR supplies licensed QNN
    // GPU/HTP/System/Prepare and HTP stub/skel libraries for arm64-v8a.
    // The optional QNN CPU reference backend must come from a matching QAIRT SDK.
    implementation("com.qualcomm.qti:qnn-runtime:2.49.0")
}
