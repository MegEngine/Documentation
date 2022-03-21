package com.example.cameraxapp

import android.content.res.AssetManager
import android.util.Log
import java.io.*
import java.lang.RuntimeException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean


class ImageClassifier {
    companion object {
        private const val TAG = "ImageClassifier"
        private const val SO_NAME = "MegEngineLiteAndroid"
        private const val ASSET_FILE_PREFIX = "file:///android_asset/"
        private val isSoLoaded = AtomicBoolean(false)

        const val DEFAULT_THRESHOLD = 0.1f

        init {
            System.loadLibrary(SO_NAME)
        }
    }

    // Load MegEngine Lite shared library to use native code interface below
    public fun prepareRun(): Boolean {
        if (!isSoLoaded.get()) {
            try {
                System.loadLibrary(SO_NAME)
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Load $SO_NAME library failed!" + e.message)
                return false
            }
            isSoLoaded.set(true)
            Log.v(TAG, "Load $SO_NAME library success!")
        }
        return true
    }

    // Read files from asset folder or the disk (load the dumped .mge model file in this case)
    public fun loadModel(assetManager: AssetManager, inputFile: String): ByteArray {

        Log.d(TAG, "Try reading $inputFile")
        val hasAssetPrefix = inputFile.startsWith(ASSET_FILE_PREFIX)
        val inputStream = try {
            val filePath = if (hasAssetPrefix) inputFile.split(ASSET_FILE_PREFIX)[1] else inputFile
            assetManager.open(filePath)
        } catch (e: IOException) {
            if (hasAssetPrefix) throw RuntimeException("Failed to load model $inputFile from assets.", e)
            try {
                FileInputStream(inputFile)
            } catch (e: IOException) {
                throw RuntimeException("Failed to load model $inputFile from assets or disk.", e)
            }
        }

        val inputBytes = try {
            inputStream.readBytes()
        } catch (e: IOException) {
            throw RuntimeException(e)
        }

        Log.v(TAG, "Read $inputFile (${inputBytes.size} bytes) success!")
        inputStream.close()

        return inputBytes
    }

    public external fun predict(model: ByteArray, image: IntArray, height: Int, width:Int) : String
}