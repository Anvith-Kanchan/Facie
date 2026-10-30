# Facie — camera-only face authentication for Android

Native Kotlin app. It uses the front camera, ML Kit landmarks, and a local MobileFaceNet TFLite embedding model. It does **not** call Android Face Unlock, BiometricPrompt, or device biometric APIs.

## Included

- CameraX live preview and on-device face landmarks.
- Eye/eye/nose landmark alignment to a 112×112 crop.
- MobileFaceNet 128-dimensional embeddings and cosine similarity.
- Three enrollment samples, averaged and L2-normalized.
- Randomized active blink/head-turn challenges.
- AES-256-GCM encrypted template; key generated and retained in Android Keystore.
- No INTERNET permission; no image/embedding upload; Android backup disabled.
- Screen capture disabled while active; five failures trigger a 30-second cooldown.
- GitHub Actions APK artifact workflow.

## Build

Android Studio with JDK 17 and Android SDK 36, or run `gradle assembleDebug` with Gradle 8.13. The build workflow downloads the Apache-2.0 MobileFaceNet TFLite model from Qualcomm's public artifact, then builds and uploads `app-debug.apk`.

GitHub → Actions → **Build Facie APK** → latest successful run → Artifacts → `facie-debug-apk`.

Model card and license: https://huggingface.co/qualcomm/MobileFaceNet

## Security warning

This is a **research/demo authenticator**, not a secure replacement for Android hardware-backed biometrics or a production access-control system.

- Blink/head-turn prompts are active liveness checks, **not** a certified presentation-attack-detection (PAD) model. They do not reliably stop high-quality replay, camera injection, or real-time deepfakes.
- The 0.55 cosine threshold is a provisional starting point, not a validated FAR/FRR threshold. It needs device-specific calibration with genuine and impostor test sets.
- Landmark alignment is a simplified three-point transform, not a full production-grade five-point alignment pipeline.
- No passive anti-spoof model is bundled. A stronger release needs a licensed PAD model, calibrated recognition threshold, independent presentation-attack testing, and review on rooted/compromised devices.
- Keystore protects the template at rest; it cannot defend against a compromised OS or malicious camera pipeline.
- Facie authenticates only within this app. It does not unlock Android or other apps.

## Privacy

Frames are processed on-device and discarded. Only an encrypted 128-float template is persisted. Delete enrollment to clear the ciphertext and Keystore key. The app has no network permission.

## Third-party components

MobileFaceNet model (Qualcomm, Apache-2.0), AndroidX CameraX, Google ML Kit Face Detection, TensorFlow Lite.
