package com.anvith.facie

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class FaceEmbedding(context: Context) {
    private val interpreter: Interpreter
    init {
        val fd = context.assets.openFd("mobilefacenet.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1,112,112,3))) {
            "Unexpected model input shape: ${interpreter.getInputTensor(0).shape().contentToString()}"
        }
    }

    /** Aligns eyes + nose to a canonical 112px crop before embedding. */
    fun embed(face: Bitmap, leftEye: PointF, rightEye: PointF, nose: PointF): FloatArray {
        val src = floatArrayOf(leftEye.x,leftEye.y, rightEye.x,rightEye.y, nose.x,nose.y)
        val dst = floatArrayOf(38f,42f, 74f,42f, 56f,66f)
        val transform = Matrix()
        check(transform.setPolyToPoly(src, 0, dst, 0, 3))
        val aligned = Bitmap.createBitmap(112,112,Bitmap.Config.ARGB_8888)
        Canvas(aligned).drawBitmap(face, transform, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
        val input = ByteBuffer.allocateDirect(1*112*112*3*4).order(ByteOrder.nativeOrder())
        for (y in 0 until 112) for (x in 0 until 112) {
            val p = aligned.getPixel(x,y)
            input.putFloat((((p shr 16) and 255) - 127.5f) / 128f)
            input.putFloat((((p shr 8) and 255) - 127.5f) / 128f)
            input.putFloat(((p and 255) - 127.5f) / 128f)
        }
        input.rewind()
        val output = Array(1) { FloatArray(128) }
        interpreter.run(input, output)
        aligned.recycle()
        val v = output[0]
        val norm = sqrt(v.sumOf { (it*it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
        return FloatArray(128) { v[it] / norm }
    }

    fun close() = interpreter.close()
    companion object {
        fun cosine(a: FloatArray, b: FloatArray): Float {
            require(a.size == b.size)
            return a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
        }
    }
}
