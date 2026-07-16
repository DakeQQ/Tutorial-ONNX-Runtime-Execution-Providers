# The demo keeps minification disabled. Retain ORT classes if release minification is enabled later.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
