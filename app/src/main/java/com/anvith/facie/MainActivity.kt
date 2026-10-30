package com.anvith.facie

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var action: Button
    private lateinit var store: SecureTemplateStore
    @Volatile private var embedder: FaceEmbedding? = null

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val modelExecutor = Executors.newSingleThreadExecutor()
    private val detectorDelegate = lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .enableTracking()
                .build()
        )
    }

    @Volatile private var sessionActive = false
    @Volatile private var busy = false
    private val detector by detectorDelegate
    private var enrolling = false
    private var failed = 0
    private var lockedUntil = 0L
    private var challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
    private var challengeIndex = 0
    private var blinkSeen = false
    private val samples = mutableListOf<FloatArray>()

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "Camera permission is required. Grant it in Settings to use Facie."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, true)
        window.setFlags(8192, 8192)

        store = SecureTemplateStore(this)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            gravity = Gravity.CENTER_HORIZONTAL
        }
        status = TextView(this).apply {
            text = "Facie • Preparing secure face engine…"
            textSize = 18f
            gravity = Gravity.CENTER
        }
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        action = Button(this).apply {
            text = "Loading face engine…"
            isEnabled = false
        }
        val reset = Button(this).apply { text = "Delete enrolled face" }
        root.addView(preview, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(status, LinearLayout.LayoutParams(-1, 72))
        root.addView(action, LinearLayout.LayoutParams(-1, 56))
        root.addView(reset, LinearLayout.LayoutParams(-1, 52))
        setContentView(root)

        action.setOnClickListener {
            if (System.currentTimeMillis() < lockedUntil) {
                status.text = "Temporarily locked. Try again later."
                return@setOnClickListener
            }
            if (embedder == null) {
                status.text = "Face engine is unavailable. Restart the app or reinstall the latest build."
                return@setOnClickListener
            }
            enrolling = safeLoadTemplate() == null
            sessionActive = true
            samples.clear()
            challengeIndex = 0
            blinkSeen = false
            failed = 0
            challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
            status.text = if (enrolling) "Enrollment: complete the liveness prompts" else "Verification: ${challenge[0]}"
        }

        reset.setOnClickListener {
            runCatching { store.clear() }
                .onSuccess {
                    samples.clear()
                    enrolling = false
                    sessionActive = false
                    status.text = "Enrolled face deleted."
                    action.text = "Enroll my face"
                }
                .onFailure { status.text = "Could not delete the saved template: ${it.message ?: "storage error"}" }
        }

        initializeFaceEngine()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initializeFaceEngine() {
        status.text = "Loading on-device face model…"
        modelExecutor.execute {
            val result = runCatching { FaceEmbedding(this) }
            runOnUiThread {
                if (isFinishing || isDestroyed) {
                    result.getOrNull()?.close()
                    return@runOnUiThread
                }
                result.onSuccess { engine ->
                    embedder = engine
                    action.isEnabled = true
                    action.text = if (safeLoadTemplate() == null) "Enroll my face" else "Verify identity"
                    status.text = "Face engine ready. Position your face in the camera."
                }.onFailure { error ->
                    embedder = null
                    action.isEnabled = false
                    action.text = "Face engine unavailable"
                    status.text = "Could not load face model: ${error.message ?: error.javaClass.simpleName}"
                }
            }
        }
    }

    private fun safeLoadTemplate(): FloatArray? = runCatching { store.load() }.getOrNull()

    private fun startCamera() {
        try {
            val future = ProcessCameraProvider.getInstance(this)
            future.addListener({
                try {
                    val provider = future.get()
                    if (!provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                        status.text = "This device has no front camera supported by CameraX."
                        return@addListener
                    }
                    val previewCase = Preview.Builder().build().also {
                        it.setSurfaceProvider(preview.surfaceProvider)
                    }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                    analysis.setAnalyzer(cameraExecutor) { proxy -> analyze(proxy) }
                    provider.unbindAll()
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, previewCase, analysis)
                    status.text = if (embedder == null) "Camera ready. Loading face model…" else "Camera ready. Position your face."
                } catch (error: Exception) {
                    status.text = "Camera could not start: ${error.message ?: error.javaClass.simpleName}"
                }
            }, ContextCompat.getMainExecutor(this))
        } catch (error: Exception) {
            status.text = "Camera initialization failed: ${error.message ?: error.javaClass.simpleName}"
        }
    }

    private fun analyze(proxy: ImageProxy) {
        if (busy || !sessionActive || embedder == null) {
            proxy.close()
            return
        }
        val media = proxy.image
        if (media == null) {
            proxy.close()
            return
        }
        busy = true
        try {
            val input = InputImage.fromMediaImage(media, proxy.imageInfo.rotationDegrees)
            detector.process(input)
                .addOnSuccessListener(cameraExecutor) { faces ->
                    if (!sessionActive) return@addOnSuccessListener
                    try {
                        val face = faces.singleOrNull()
                        if (face == null) {
                            postStatus("Keep exactly one face in view.")
                            return@addOnSuccessListener
                        }
                        val left = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
                        val right = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
                        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
                        if (left == null || right == null || nose == null) {
                            postStatus("Face camera in good lighting and look toward it.")
                            return@addOnSuccessListener
                        }

                        val command = challenge.getOrNull(challengeIndex)
                        val leftOpen = face.leftEyeOpenProbability ?: 1f
                        val rightOpen = face.rightEyeOpenProbability ?: 1f
                        val live = when (command) {
                            "BLINK" -> {
                                if (leftOpen < 0.35f && rightOpen < 0.35f) blinkSeen = true
                                blinkSeen && leftOpen > 0.70f && rightOpen > 0.70f
                            }
                            "TURN LEFT" -> face.headEulerAngleY < -18f
                            "TURN RIGHT" -> face.headEulerAngleY > 18f
                            else -> true
                        }
                        if (!live) {
                            postStatus("Liveness check: ${command ?: "hold still"}")
                            return@addOnSuccessListener
                        }

                        challengeIndex++
                        if (challengeIndex < challenge.size) {
                            blinkSeen = false
                            postStatus("Liveness check: ${challenge[challengeIndex]}")
                            return@addOnSuccessListener
                        }

                        val engine = embedder ?: return@addOnSuccessListener
                        val bitmap = proxy.toUprightBitmap()
                        val vector = try {
                            engine.embed(bitmap, PointF(left.x, left.y), PointF(right.x, right.y), PointF(nose.x, nose.y))
                        } finally {
                            bitmap.recycle()
                        }

                        if (enrolling) {
                            samples.add(vector)
                            if (samples.size >= 3) {
                                val dimension = samples.first().size
                                require(samples.all { it.size == dimension }) { "Model returned inconsistent embedding sizes." }
                                val mean = FloatArray(dimension) { i -> samples.map { it[i] }.average().toFloat() }
                                val norm = sqrt(mean.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
                                runCatching { store.save(FloatArray(dimension) { mean[it] / norm }) }
                                    .onSuccess {
                                        sessionActive = false
                                        enrolling = false
                                        postStatus("ENROLLED • encrypted template saved locally")
                                        runOnUiThread { action.text = "Verify identity" }
                                    }
                                    .onFailure { error ->
                                        sessionActive = false
                                        postStatus("Could not securely save template: ${error.message ?: "storage error"}")
                                    }
                            } else {
                                challengeIndex = 0
                                blinkSeen = false
                                postStatus("Sample ${samples.size}/3 captured. Adjust your angle slightly.")
                            }
                        } else {
                            val reference = safeLoadTemplate()
                            val score = if (reference == null) -1f else FaceEmbedding.cosine(vector, reference)
                            if (score >= 0.55f) {
                                failed = 0
                                sessionActive = false
                                postStatus("MATCH • identity verified (%.3f)".format(score))
                            } else {
                                failed++
                                if (failed >= 5) lockedUntil = System.currentTimeMillis() + 30_000
                                sessionActive = false
                                postStatus("NO MATCH • access denied (%.3f)".format(score))
                            }
                            challengeIndex = 0
                            blinkSeen = false
                        }
                    } catch (error: Exception) {
                        sessionActive = false
                        postStatus("Face processing error: ${error.message ?: error.javaClass.simpleName}")
                    }
                }
                .addOnFailureListener(cameraExecutor) { error ->
                    if (sessionActive) postStatus("Face detection failed: ${error.message ?: "try again"}")
                }
                .addOnCompleteListener(cameraExecutor) {
                    busy = false
                    proxy.close()
                }
        } catch (error: Exception) {
            busy = false
            proxy.close()
            postStatus("Camera frame error: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun postStatus(message: String) {
        runOnUiThread { if (!isFinishing && !isDestroyed) status.text = message }
    }

    /** Converts YUV_420_888 respecting row/pixel strides, then applies CameraX rotation. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yBase = yBuf.position()
        val uBase = uBuf.position()
        val vBase = vBuf.position()
        val pixels = IntArray(width * height)
        for (row in 0 until height) for (col in 0 until width) {
            val yp = yBase + row * yPlane.rowStride + col * yPlane.pixelStride
            val uvRow = row / 2
            val uvCol = col / 2
            val up = uBase + uvRow * uPlane.rowStride + uvCol * uPlane.pixelStride
            val vp = vBase + uvRow * vPlane.rowStride + uvCol * vPlane.pixelStride
            val yy = (yBuf.get(yp).toInt() and 255) - 16
            val u = (uBuf.get(up).toInt() and 255) - 128
            val v = (vBuf.get(vp).toInt() and 255) - 128
            val y = max(0, yy)
            val r = (1.164f * y + 1.596f * v).toInt().coerceIn(0, 255)
            val g = (1.164f * y - 0.392f * u - 0.813f * v).toInt().coerceIn(0, 255)
            val b = (1.164f * y + 2.017f * u).toInt().coerceIn(0, 255)
            pixels[row * width + col] = -0x1000000 or (r shl 16) or (g shl 8) or b
        }
        val raw = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        val rotation = imageInfo.rotationDegrees
        if (rotation == 0) return raw
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true).also { raw.recycle() }
    }

    override fun onDestroy() {
        sessionActive = false
        if (detectorDelegate.isInitialized()) runCatching { detector.close() }
        embedder?.let { runCatching { it.close() } }
        cameraExecutor.shutdown()
        modelExecutor.shutdown()
        super.onDestroy()
    }
}
