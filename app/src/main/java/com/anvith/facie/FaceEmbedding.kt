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

/**
 * Loads the bundled float MobileFaceNet model and adapts to its actual tensor
 * dimensions instead of assuming a particular embedding length.
 */
class FaceEmbedding(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputHeight: Int
    private val inputWidth: Int
    private val outputSize: Int

    init {
        val asset = context.assets.openFd("mobilefacenet.tflite")
        val modelBuffer = try {
            FileInputStream(asset.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
            }
        } finally {
            asset.close()
        }

        val created = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(2) })
        try {
            val input = created.getInputTensor(0)
            val shape = input.shape()
            require(input.dataType().name == "FLOAT32") {
                "Unsupported model input type ${input.dataType()}; expected FLOAT32."
            }
            require(shape.size == 4 && shape[0] == 1) {
                "Unsupported model input shape ${shape.contentToString()}."
            }
            val h: Int
            val w: Int
            when {
                shape[3] == 3 -> { h = shape[1]; w = shape[2] } // NHWC
                shape[1] == 3 -> { h = shape[2]; w = shape[3] } // NCHW
                else -> throw IllegalArgumentException("Model input must have three RGB channels: ${shape.contentToString()}.")
            }
            require(h > 0 && w > 0) { "Model has dynamic image dimensions; a fixed input size is required." }

            val output = created.getOutputTensor(0)
            require(output.dataType().name == "FLOAT32") {
                "Unsupported model output type ${output.dataType()}; expected FLOAT32."
            }
            val elements = output.shape().fold(1) { total, dimension -> total * dimension }
            require(elements >= 16) { "Model output is unexpectedly small: ${output.shape().contentToString()}." }

            inputHeight = h
            inputWidth = w
            outputSize = elements
            interpreter = created
        } catch (error: Exception) {
            created.close()
            throw IllegalArgumentException("MobileFaceNet model is incompatible: ${error.message}", error)
        }
    }

    /** Aligns eyes and nose to a canonical crop, then runs the model. */
    fun embed(face: Bitmap, leftEye: PointF, rightEye: PointF, nose: PointF): FloatArray {
        val src = floatArrayOf(leftEye.x, leftEye.y, rightEye.x, rightEye.y, nose.x, nose.y)
        val scaleX = inputWidth / 112f
        val scaleY = inputHeight / 112f
        val dst = floatArrayOf(
            38f * scaleX, 42f * scaleY,
            74f * scaleX, 42f * scaleY,
            56f * scaleX, 66f * scaleY
        )
        val transform = Matrix()
        check(transform.setPolyToPoly(src, 0, dst, 0, 3)) { "Could not align detected facial landmarks." }

        val aligned = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        try {
            Canvas(aligned).drawBitmap(face, transform, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
            val inputShape = interpreter.getInputTensor(0).shape()
            val channelsFirst = inputShape[1] == 3
            val input = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * 4)
                .order(ByteOrder.nativeOrder())

            for (y in 0 until inputHeight) for (x in 0 until inputWidth) {
                val pixel = aligned.getPixel(x, y)
                val red = (((pixel shr 16) and 255) - 127.5f) / 128f
                val green = (((pixel shr 8) and 255) - 127.5f) / 128f
                val blue = ((pixel and 255) - 127.5f) / 128f
                if (channelsFirst) {
                    // NCHW requires planar RGB; fill planes below instead.
                    continue
                } else {
                    input.putFloat(red)
                    input.putFloat(green)
                    input.putFloat(blue)
                }
            }

            if (channelsFirst) {
                input.clear()
                for (channel in 0..2) for (y in 0 until inputHeight) for (x in 0 until inputWidth) {
                    val pixel = aligned.getPixel(x, y)
                    val value = when (channel) {
                        0 -> ((pixel shr 16) and 255)
                        1 -> ((pixel shr 8) and 255)
                        else -> (pixel and 255)
                    }
                    input.putFloat((value - 127.5f) / 128f)
                }
            }
            input.rewind()
            val output = ByteBuffer.allocateDirect(outputSize * 4).order(ByteOrder.nativeOrder())
            interpreter.run(input, output)
            output.rewind()
            val raw = FloatArray(outputSize) { output.float }
            val norm = sqrt(raw.sumOf { (it * it).toDouble() }).toFloat()
            require(norm.isFinite() && norm > 1e-8f) { "Model returned an invalid face embedding." }
            return FloatArray(outputSize) { raw[it] / norm }
        } finally {
            aligned.recycle()
        }
    }

    override fun close() = interpreter.close()

    companion object {
        fun cosine(a: FloatArray, b: FloatArray): Float {
            require(a.size == b.size) { "Face templates use different embedding dimensions." }
            return a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
        }
    }
}
