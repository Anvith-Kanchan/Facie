package com.anvith.facie

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
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
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var action: Button
    private lateinit var store: SecureTemplateStore
    private lateinit var embedder: FaceEmbedding
    private val executor = Executors.newSingleThreadExecutor()
    private val detector by lazy {
        FaceDetection.getClient(FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .enableTracking().build())
    }
    private var enrolling = false
    @Volatile private var sessionActive = false
    private var busy = false
    private var failed = 0
    private var lockedUntil = 0L
    private var challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
    private var challengeIndex = 0
    private var blinkSeen = false
    private val samples = mutableListOf<FloatArray>()
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else status.text = "Camera permission is required."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.setFlags(8192, 8192)
        store = SecureTemplateStore(this)
        embedder = FaceEmbedding(this)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24,24,24,24); gravity = Gravity.CENTER_HORIZONTAL }
        status = TextView(this).apply { text = "Facie • Camera-only face authentication"; textSize = 18f; gravity = Gravity.CENTER }
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        action = Button(this).apply { text = if (store.load() == null) "Enroll my face" else "Verify identity" }
        val reset = Button(this).apply { text = "Delete enrolled face" }
        root.addView(status, LinearLayout.LayoutParams(-1, 64))
        root.addView(preview, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(action, LinearLayout.LayoutParams(-1, 56))
        root.addView(reset, LinearLayout.LayoutParams(-1, 52))
        setContentView(root)
        action.setOnClickListener {
            if (System.currentTimeMillis() < lockedUntil) { status.text = "Temporarily locked. Try again later."; return@setOnClickListener }
            enrolling = store.load() == null
            sessionActive = true
            samples.clear(); challengeIndex = 0; blinkSeen = false; failed = 0
            challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
            status.text = if (enrolling) "Enrollment: complete the liveness prompts" else "Verification: ${challenge[0]}"
        }
        reset.setOnClickListener {
            store.clear(); samples.clear(); enrolling = false; sessionActive = false
            status.text = "Enrolled face deleted."; action.text = "Enroll my face"
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val previewCase = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            analysis.setAnalyzer(executor) { proxy -> analyze(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, previewCase, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        if (busy || !sessionActive) { proxy.close(); return }
        val media = proxy.image
        if (media == null) { proxy.close(); return }
        busy = true
        val input = InputImage.fromMediaImage(media, proxy.imageInfo.rotationDegrees)
        detector.process(input).addOnSuccessListener(executor) { faces ->
            if (!sessionActive) return@addOnSuccessListener
            val face = faces.singleOrNull() ?: run { status.post { status.text = "Keep exactly one face in view."; }; return@addOnSuccessListener }
            val left = face.getLandmark(Face.LANDMARK_LEFT_EYE)?.position
            val right = face.getLandmark(Face.LANDMARK_RIGHT_EYE)?.position
            val nose = face.getLandmark(Face.LANDMARK_NOSE_BASE)?.position
            if (left == null || right == null || nose == null) { status.post { status.text = "Face camera in good lighting."; }; return@addOnSuccessListener }
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
            if (!live) { status.post { status.text = "Liveness check: ${command ?: "hold still"}"; }; return@addOnSuccessListener }
            challengeIndex++
            if (challengeIndex < challenge.size) { status.post { status.text = "Liveness check: ${challenge[challengeIndex]}"; }; return@addOnSuccessListener }
            val bitmap = proxy.toUprightBitmap()
            val vector = runCatching {
                embedder.embed(bitmap, PointF(left.x,left.y), PointF(right.x,right.y), PointF(nose.x,nose.y))
            }.getOrNull()
            bitmap.recycle()
            if (vector != null) {
                if (enrolling) {
                    samples.add(vector)
                    if (samples.size >= 3) {
                        val mean = FloatArray(128) { i -> samples.map { it[i] }.average().toFloat() }
                        val norm = sqrt(mean.sumOf { (it*it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
                        store.save(FloatArray(128) { mean[it] / norm })
                        sessionActive = false
                        status.post { status.text = "ENROLLED • encrypted template saved locally"; action.text = "Verify identity" }
                        enrolling = false
                    } else {
                        challengeIndex = 0; blinkSeen = false
                        status.post { status.text = "Sample ${samples.size}/3 captured. Adjust your angle slightly."; }
                    }
                } else {
                    val reference = store.load()
                    val score = if (reference == null) -1f else FaceEmbedding.cosine(vector, reference)
                    if (score >= 0.55f) {
                        failed = 0
                        sessionActive = false
                        status.post { status.text = "MATCH • identity verified (%.3f)".format(score) }
                    } else {
                        failed++
                        if (failed >= 5) lockedUntil = System.currentTimeMillis() + 30_000
                        sessionActive = false
                        status.post { status.text = "NO MATCH • access denied (%.3f)".format(score) }
                    }
                    challengeIndex = 0; blinkSeen = false
                }
            } else {
                challengeIndex = 0; blinkSeen = false
                status.post { status.text = "Could not create face template. Try again."; }
            }
        }.addOnFailureListener(executor) { status.post { status.text = "Face processing failed. Retry."; } }
            .addOnCompleteListener(executor) { busy = false; proxy.close() }
    }

    /** Correctly handles YUV_420_888 row/pixel strides, then applies CameraX rotation. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val yPlane = planes[0]; val uPlane = planes[1]; val vPlane = planes[2]
        val yBuf = yPlane.buffer; val uBuf = uPlane.buffer; val vBuf = vPlane.buffer
        val yBase = yBuf.position(); val uBase = uBuf.position(); val vBase = vBuf.position()
        val pixels = IntArray(width * height)
        for (row in 0 until height) for (col in 0 until width) {
            val yp = yBase + row * yPlane.rowStride + col * yPlane.pixelStride
            val uvRow = row / 2; val uvCol = col / 2
            val up = uBase + uvRow * uPlane.rowStride + uvCol * uPlane.pixelStride
            val vp = vBase + uvRow * vPlane.rowStride + uvCol * vPlane.pixelStride
            val yy = (yBuf.get(yp).toInt() and 255) - 16
            val u = (uBuf.get(up).toInt() and 255) - 128
            val v = (vBuf.get(vp).toInt() and 255) - 128
            val y = max(0, yy)
            val r = (1.164f*y + 1.596f*v).toInt().coerceIn(0,255)
            val g = (1.164f*y - 0.392f*u - 0.813f*v).toInt().coerceIn(0,255)
            val b = (1.164f*y + 2.017f*u).toInt().coerceIn(0,255)
            pixels[row*width+col] = -0x1000000 or (r shl 16) or (g shl 8) or b
        }
        val raw = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        val rotation = imageInfo.rotationDegrees
        if (rotation == 0) return raw
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(raw,0,0,raw.width,raw.height,matrix,true).also { raw.recycle() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close(); embedder.close(); executor.shutdown()
    }
}
