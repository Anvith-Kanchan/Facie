package com.anvith.facie

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
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
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import kotlin.math.abs

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
    private var busy = false
    private var failed = 0
    private var lockedUntil = 0L
    private var challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
    private var challengeIndex = 0
    private var blinkSeen = false
    private var lastVector: FloatArray? = null
    private var samples = mutableListOf<FloatArray>()
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else status.text = "Camera permission is required."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.setFlags(8192, 8192)
        store = SecureTemplateStore(this)
        embedder = FaceEmbedding(this)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24,24,24,24); gravity = Gravity.CENTER_HORIZONTAL }
        status = TextView(this).apply { text = "Facie • On-device face authentication"; textSize = 18f; gravity = Gravity.CENTER }
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
            samples.clear(); challengeIndex = 0; blinkSeen = false; failed = 0
            challenge = listOf("BLINK", "TURN LEFT", "TURN RIGHT").shuffled().take(2)
            status.text = if (enrolling) "Enrollment: follow the on-screen liveness checks" else "Verification: ${challenge[0]}"
        }
        reset.setOnClickListener { store.clear(); status.text = "Enrolled face deleted."; action.text = "Enroll my face" }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val previewUseCase = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            analysis.setAnalyzer(executor) { proxy -> analyze(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, previewUseCase, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        if (busy || !enrolling && lastVector != null) { proxy.close(); return }
        val media = proxy.image
        if (media == null) { proxy.close(); return }
        busy = true
        val input = InputImage.fromMediaImage(media, proxy.imageInfo.rotationDegrees)
        detector.process(input).addOnSuccessListener { faces ->
            if (!enrolling && lastVector != null) return@addOnSuccessListener
            val face = faces.singleOrNull() ?: run { status.post { status.text = "Keep exactly one face in view."; }; return@addOnSuccessListener }
            val left = face.getLandmark(Face.LANDMARK_LEFT_EYE)?.position
            val right = face.getLandmark(Face.LANDMARK_RIGHT_EYE)?.position
            val nose = face.getLandmark(Face.LANDMARK_NOSE_BASE)?.position
            if (left == null || right == null || nose == null) { status.post { status.text = "Move into good lighting and face the camera."; }; return@addOnSuccessListener }
            val yaw = face.headEulerAngleY
            val command = challenge.getOrNull(challengeIndex)
            val live = when (command) {
                "BLINK" -> {
                    val eye = face.leftEyeOpenProbability ?: 1f
                    if (eye < 0.35f) blinkSeen = true
                    blinkSeen && eye > 0.7f
                }
                "TURN LEFT" -> yaw < -18f
                "TURN RIGHT" -> yaw > 18f
                else -> true
            }
            if (!live) { status.post { status.text = "Liveness check: ${command ?: "hold still"}"; }; return@addOnSuccessListener }
            challengeIndex++
            if (challengeIndex < challenge.size) { status.post { status.text = "Liveness check: ${challenge[challengeIndex]}"; }; return@addOnSuccessListener }
            val bitmap = proxy.toBitmap()
            val points = listOf(left,right,nose).map { PointF(it.x, it.y) }
            val vector = runCatching { embedder.embed(bitmap, points[0], points[1], points[2]) }.getOrNull()
            bitmap.recycle()
            if (vector != null) {
                if (enrolling) {
                    samples.add(vector)
                    if (samples.size >= 3) {
                        val mean = FloatArray(128) { i -> samples.map { it[i] }.average().toFloat() }
                        val norm = kotlin.math.sqrt(mean.sumOf { (it*it).toDouble() }).toFloat()
                        store.save(FloatArray(128) { mean[it] / norm })
                        status.post { status.text = "Face enrolled. Templates encrypted on this device."; action.text = "Verify identity" }
                        enrolling = false
                    } else {
                        challengeIndex = 0; blinkSeen = false
                        status.post { status.text = "Sample ${samples.size}/3 captured. Reposition slightly."; }
                    }
                } else {
                    lastVector = vector
                    val reference = store.load()
                    val score = if (reference == null) -1f else FaceEmbedding.cosine(vector, reference)
                    val passed = score >= 0.55f
                    if (passed) {
                        failed = 0
                        status.post { status.text = "MATCH • identity verified (score %.3f)".format(score) }
                    } else {
                        failed++
                        if (failed >= 5) lockedUntil = System.currentTimeMillis() + 30_000
                        status.post { status.text = "NO MATCH • access denied (score %.3f)".format(score) }
                    }
                    lastVector = null
                    challengeIndex = 0; blinkSeen = false
                }
            }
        }.addOnFailureListener { status.post { status.text = "Face processing failed. Retry."; } }
            .addOnCompleteListener { busy = false; proxy.close() }
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val y = planes[0].buffer; val u = planes[1].buffer; val v = planes[2].buffer
        val ySize = y.remaining(); val uSize = u.remaining(); val vSize = v.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        y.get(nv21,0,ySize)
        for (i in 0 until vSize) { nv21[ySize + 2*i] = v.get(i); nv21[ySize + 2*i + 1] = u.get(i) }
        val out = ByteArrayOutputStream()
        YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null).compressToJpeg(Rect(0,0,width,height), 90, out)
        val raw = BitmapFactory.decodeByteArray(out.toByteArray(),0,out.size())
        val matrix = android.graphics.Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(raw,0,0,raw.width,raw.height,matrix,true).also { if (it !== raw) raw.recycle() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close(); embedder.close(); executor.shutdown()
    }
}
