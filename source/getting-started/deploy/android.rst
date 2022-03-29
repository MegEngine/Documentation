.. _deploy-android:

=======================
在 Android 平台部署模型
=======================
.. admonition:: 本教程涉及的内容
   :class: note

   * 如何将输入数据的预处理逻辑吸进模型推理流程中，减少对应 C++ 代码的编写与库依赖；
   * 如何借助 MegEngine Lite 将模型部署到 Android 平台，展示基本的开发流程；
   * 最终你将能够开发出一个利用设备后置摄像头进行实时分类的 Android APP.

.. admonition:: 预置知识说明
   :class: warning

   * 你必须已经掌握 MegEngine Lite 接口基本使用，可参考 :ref:`deploy-megengine-lite` 教程；
   * 想要跑通本教程，对 Android 开发基础知识的了解不是必需的，本教程中会有各步骤的简要解释。

.. seealso::

   本教程的所有源码在： :docs:`examples/deploy/android` , CameraXApp 可作为 Android Studio 项目打开。

概览
----

想要将能够在 Linux x86 平台上成功进行推理的 MegEngine 模型部署到 Android 平台（以 arm64-v8a 为例），
需要考虑到以下几个基本特点（我们会在接下来的小节进行具体的实践）：

* 跨平台特性：我们需要根据 NDK 提供的交叉编译工具链得到 MegEngine Lite 在目标平台的动态库；
* 接口封装与调用：Lite 提供的是 C/C++ 接口与实现，想要在 Android 项目中进行调用，要使用到 JNI;
* 安卓项目开发：我们需要了解开发出一个 APP 的基本思路和流程，好将 Lite 模型推理进行接入。

在实践过程中会遇到一些需要额外关注的细节，我们将在对应的小节再给出具体的解释。

注意：本教程中对于 MegEngine Lite 模型推理相关的部分提供较为具体的介绍，而 Android 开发具体步骤的讲解不是本教程关注的重点。
CameraXApp 中的用例代码主要参考自 Google 官方文档 `CameraX overview <https://developer.android.com/training/camerax>`_ ,
如果读者对 Android 开发背后的原理和细节感兴趣，可在进行到相关步骤时自行借助互联网查询相关概念。

.. seealso::

   摄像头捕获、分析图片并监听返回信息的的源代码实现在：

   :docs:`examples/deploy/android/CameraXApp/app/src/main/java/com/example/cameraxapp/MainActivity.kt`

   我们将主要关注其中 ImageAnalyzer 和 ImageClassifier 的设计，后者的推理接口本质上将调用 Lite 库。

获取预训练好的模型
------------------

执行 :docs:`examples/deploy/android/model.py` 中的代码，默认将会得到名为 ``snetv2_x100_deploy.mge`` 模型用于部署。

值得一提的是，本教程中所得到的 ``.mge`` 模型与上一个教程中略有不同（可对比查看脚本逻辑）。
考虑到输入数据总是要经过一定的预处理操作（例如我们在训练模型时经常用到
:mod:`~.data.transform` 模块进行预处理），在部署时如果用 C++ 做对应的实现通常会引入 OpenCV 第三方依赖，
且需要对推理结果进行等价性验证，整个流程比较繁琐。因此一种做法是：将预处理操作写进被 :class:`~.trace` 的推理函数，
连同模型的推理过程一同被 :meth:`~.trace.dump` 成 ``.mge`` 模型文件。

.. dropdown:: :fa:`eye,mr-1` 查看吸入模型内的预处理逻辑

   通常的预处理操作（需要在部署时写出等价的 C++ 逻辑）：

   .. literalinclude:: ../../../examples/deploy/android/model.py
      :language: python
      :lines: 63-70

   可以提前写成等价的 Tensor 有关操作：

   .. literalinclude:: ../../../examples/deploy/android/model.py
      :language: python
      :lines: 88-122

   在执行 :class:`~.trace` 时写进推理逻辑中：

   .. literalinclude:: ../../../examples/deploy/android/model.py
      :language: python
      :lines: 47-52

   注意其中获取图片长宽的代码，MegEngine 中 Tensor 的 :attr:`~.Tensor.shape` 并不总是一个元组，
   在被 :class:`~.trace` 时，Tensor 的形状将以 Tensor 的形式进行记录，以便进行有关的计算。

如果你希望使用其它的预训练模型，只需要修改 ``model.py`` 中获取、预处理和导出模型的逻辑即可；
也可以直接使用其它的 ``.mge`` 模型文件，但需要知道模型是否已经吸入了预处理操作，
如果没有的话，则需要在后面实现 C++ 推理接口时做等价的预处理实现（参考上一个教程）。

交叉编译 MegEngine Lite
-----------------------

.. note::

   如果你有对应平台预编译好的 Lite 库和头文件，也可以直接使用。

请自行参考 :ref:`build-from-source` 页面中的内容，完成 ARM-Android 的交叉编译，通常在如下路径获得 Lite:

.. code-block:: shell

   {path/to/MegEngine}/build_dir/android/{arm64-v8a}/Release/install/lite

其中 ``{path/to/MegEngine}`` 是编译 MegEngine 源码路径， ``{arm64-v8a}`` 是 
`Android ABI <https://developer.android.com/ndk/guides/abis>`_ , 本例中为 arm64-v8a.

我们需要将编译得到的动态链接库 ``liblite_shared.so`` 与相应的头文件拷贝到本次教程项目代码的 ``jni`` 文件夹下：

.. code-block:: shell

   CameraXApp/app/src/main/jni/lite       <----- Make sure the path is correct
   ├── include
   │   ├── lite
   │   │   ├── common_enum_c.h
   │   │   ├── global.h
   │   │   ├── macro.h
   │   │   ├── network.h
   │   │   └── tensor.h
   │   ├── lite-c
   │   │   ├── common_enum_c.h
   │   │   ├── global_c.h
   │   │   ├── network_c.h
   │   │   └── tensor_c.h
   │   └── lite_build_config.h
   └── lib
      └── aarch64
         └── liblite_shared.so

这些文件将会在我们下一小节实现 ImageClassifier 的推理接口时用到，我们即将介绍。

设计与实现 ImageClassifier
--------------------------

在此之前，让我们先在 Android 项目中设计和实现一个 ImageClassifier 类，看它需要提供什么样的接口：


.. code-block:: kotlin

   class ImageClassifier {
       public fun prepareRun(): Boolean
       public fun loadModel(assetManager: AssetManager, inputFile: String): ByteArray
       public external fun predict(model: ByteArray, image: IntArray, height: Int, width:Int) : String
   }

我们设计的 ImageClassifier 主要有三个可供调用的接口：

* ``prepareRun``: 进行一些准备工作，比如加载推理所需的 ``.so`` 动态库，使得相应的 C++ 接口可见；
* ``loadModel``: 即加载模型，在 Android APK 开发中我们有几种常见的思路获取和加载 ``.mge`` 模型。
  一种是允许用户从手机储存卡或网络地址中加载模型文件，但这需要 APP 向用户请求对应的读取和加载权限；
  另一种做法是将模型作为资源文件打包内置到 APK 中，这也是本教程所采取的做法，理解和实现起来更加简单；
* ``predict``: 根据模型和输入的图片信息，进行预测，并且返回相应的结果。

ImageClassifier 将在我们的 APP 启动后实例化并加载好 ``.mge`` 模型文件，
接着不断接受来自摄像头捕获的图片输入，执行推理分析，并返回结果。
ImageClassifier 类的完整实现代码在：

:docs:`examples/deploy/android/CameraXApp/app/src/main/java/com/example/cameraxapp/ImageClassifier.kt`

通过 JNI 调用 Lite 接口
~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: :fa:`question,mr-1` 什么是 JNI (Java Native Interface)

   我们希望能够在 Android 项目中调用 Lite 库，做法是借助 Java Native Interface, 简写为 JNI,
   它为 Android 定义了一种从托管代码（Java/Kotlin）编译的字节码与本地代码（C/C++）交互的方式。
   
   * Android 官网文档中的介绍 —— `JNI tips <https://developer.android.com/training/articles/perf-jni>`_
   * 规格标准 ——  `Java Native Interface Specification <http://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/jniTOC.html>`_

   本教程中的 ``ImageClassifier.predict`` 代码可以作为理解 JNI 使用方式的入门参考。


注意到 ``predict`` 接口的函数名前标识有 ``external`` 关键字，表明这是一个 JNI 函数，需要提供相应的 C++ 实现：

.. code-block:: cpp

   extern "C" {

   JNIEXPORT jstring JNICALL
   Java_com_example_cameraxapp_ImageClassifier_predict(
         JNIEnv *env,
         jobject thiz,
         jbyteArray model,
         jintArray image,
         jint height,
         jint width) {

      // Inference...
   }

   }

这个接口中需要实现的逻辑与常见的 Lite 模型推理逻辑基本一致，可参考 Lite 文档或上一个教程进行实现。

源代码位置：:docs:`examples/deploy/android/CameraXApp/app/src/main/cpp` -- inference.cpp
给出了一个参考实现，每次都返回 ImageNet 标签中模型预测概率最大的那个分类。

.. note::

   阅读 ``cpp`` 目录下的 ``CMakeLists.txt`` 可知，Android 项目在构建时，
   会将 ``inference.cpp`` 相关源码编译为 ``MegEngineLiteAndroid`` 动态库，
   它仅仅依赖 MegEngine Lite ARM-Android 库，不再需要用到 OpenCV（除非你确实需要用到其中的功能）。
   在 ``ImageClassifier`` 初始化和执行 ``prepareRun()`` 方法时，都会加载 ``MegEngineLiteAndroid`` 库，
   这样就能够实现最简单的 JNI 调用。

   想要让 Android 项目知道有哪些本地代码，还需要在 Gradle 中进行进行相应的配置：

   .. code-block::

      android {

          externalNativeBuild {
              cmake {
                  path file('src/main/cpp/CMakeLists.txt')
                  version '3.18.1'
              }
          }
      }

.. warning::

   但注意在本教程中，我们使用的 ``.mge`` 模型文件选择了将输入数据的预处理操作给 “吸了进去”，
   包括 ``Resize``, ``CenterCrop`` 等在内，这也意味着预处理操作直接在模型内完成，无需在 C++ 代码中进行实现。
   这就导致实际推理时，每次输入到模型中的初始数据的形状可能与执行 :class:`~.trace` 时输入 Tensor 的形状是不同的，
   准确来说，Layout 可能存在着差异，也可能由于数据类型的不一致导致占用的内存字节数不同，在拷贝时需注意。
   因此要求我们的 Lite Network 中的输入 Tensor 的 Layout 需要重新指定并分配内存，
   这也正是此处 ``predict`` 接口中要传入  ``height`` 和 ``width`` 参数的原因。(一些业务情景下可能更加复杂)

.. note::

   实际上，你也完全可以利用 JNI 封装出一个单独的 MegEngine Lite Android SDK, 提供
   Network 和 Tensor 等 C/C++ 接口的对应实现，方便在更多的 Android 项目中使用。

运行你的 Android 应用！
-----------------------

这个教程可能不会告诉你如何从零开发出一个 Android 应用，
但本教程中的 CameraXApp 是可以在 Android Studio 中作为完整的项目在安卓虚拟设备（Android Arm64 或高于 11 系统版本的 x86）
或者实际的安卓机器中运行，并进行调试的，不妨现在就尝试将这个应用真正地跑起来。
