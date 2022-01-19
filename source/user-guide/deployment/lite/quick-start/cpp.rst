.. _lite-quick-start-cpp:

===================================
MegEngine Lite C++ 模型部署快速上手
===================================

本文将从获取一个训练好的 ``shufflenet_v2`` 模型出发，
讲解如何使用 MegEngine Lite 的 C++ 接口将其部署到 CPU（Linux x86 / Android Arm）环境下运行。主要分为以下小节：

* :ref:`lite-model-dump`
* :ref:`lite-infer-code`
* :ref:`build-megengine-lite`
* :ref:`lite-compile-inference-code`
* :ref:`lite-model-deploy`

.. seealso::

   MegEngine Lite 还可以 :ref:`通过 Python 接口进行使用 <lite-quick-start-python>`, 使用方便但有局限性。

.. _lite-model-dump:

导出已经训练好的模型
--------------------

请参考 :ref:`get-model`。

.. _lite-infer-code:

编写 Inference 代码
-------------------

首先创建一个 ``main.cpp``, 在这个文件中将直接调用 MegEngine Lite 的接口运行 ``shufflenet_v2.mge`` 模型，
输入数据 ``input_tensor`` 是随机生成的，所以不用在乎计算结果。

.. code-block:: cpp

   #include <stdlib.h>
   #include <iostream>
   #include "lite/network.h"
   using namespace lite;

   int main(int argc, char** argv) {
       std::cout << " Usage: ./demo_deploy model_name" << std::endl;
       if (argc != 2) {
           std::cout << " Wrong argument" << std::endl;
           return 0;
       }

       std::string model_path = argv[1];

       //! create and load the network
       std::shared_ptr<lite::Network> network = std::make_shared<Network>();

       //! load the model
       network->load_model(model_path);

       //! get the input tensor of the network with name "data"
       std::shared_ptr<Tensor> input_tensor = network->get_io_tensor("data");

       //! fill the rand data to input tensor
       srand(static_cast<unsigned>(time(NULL)));
       size_t length =
               input_tensor->get_tensor_total_size_in_byte() / sizeof(float);
       float* in_data_ptr = static_cast<float*>(input_tensor->get_memory_ptr());
       for (size_t i = 0; i < length; i++) {
           in_data_ptr[i] =
                   static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
       }

       //! forward
       network->forward();
       network->wait();

       //! get the inference output tensor of index 0
       std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
       float* predict_ptr = static_cast<float*>(output_tensor->get_memory_ptr());
       float sum = 0.0f, max = predict_ptr[0];
       for (size_t i = 0; i < 1000; i++) {
           sum += predict_ptr[i];
           if (predict_ptr[i] > max) {
               max = predict_ptr[i];
           }
       }
       std::cout << "The output SUM is " << sum << ", Max is " << max << std::endl;
   }

上面代码主要完成了几个步骤，包括：

#. 创建默认配置的 Network；
#. 载入模型，MegEngine Lite 将读取并解析模型文件，并创建计算图；
#. 通过输入 Tensor 的名字获取模型的输入 Tensor, 并设置随机数作为输入数据；
#. 执行 Inference 逻辑;
#. 获取模型输出 Tensor, 并处理输出数据。

至此完成了一个 ``shufflenet_v2`` 模型的推理过程的 C++ 代码编写。

但在真正运行这段代码之前，还需要编译该 C++ 源文件，并链接 MegEngine Lite 库文件。 ⬇️  ⬇️  ⬇️  

.. _build-megengine-lite:

编译 MegEngine Lite
-------------------

.. note::

   * 这一步的目的是获得 MegEngine Lite 的静态链接库和动态链接库，供我们上面代码编译时候进行链接；
     编译的过程和 :ref:`从源码编译 MegEngine <build-from-source>` 中的介绍是一致的。
   * 下面将演示在 Linux x86 下使用动态链接，Android Arm 上使用静态链接的流程：


#. 首先需要 Clone 整个 MegEngine 工程，并进入到 MegEngine 的根目录：

   >>> git clone --depth=1 git@github.com:MegEngine/MegEngine.git
   >>> cd MegEngine

#. 环境准备 & 执行编译：

   .. panels::
      :container: +full-width
      :card:

      Linux x86
      ^^^^^^^^^
      准备编译依赖的子模块：

      >>> ./third_party/prepare.sh

      安装英特尔数学核心库（MKL）:

      >>> ./third_party/install-mkl.sh

      本机编译 MegEngine Lite:

      >>> scripts/cmake-build/host_build.sh
      ---
      Android Arm
      ^^^^^^^^^^^
      准备编译依赖的子模块：

      >>> ./third_party/prepare.sh

      从安卓 `官网 <https://developer.android.google.cn/ndk/downloads/>`_ 下载 NDK 并解压到某路径，
      并将改路径设置为 ``NDK_ROOT`` 环境变量：

      >>> export NDK_ROOT=/path/to/ndk

      交叉编译 MegEngine Lite:

      >>> scripts/cmake-build/cross_build_android_arm_inference.sh

.. admonition:: 编译完成之后 MegEngine Lite 库和头文件路径 /path/to/megenginelite-lib
   :class: note

   * Linux x86:   ``build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install/lite/``
   * Android Arm: ``build_dir/android/arm64-v8a/Release/install/lite/``

.. _lite-compile-inference-code:

编译 Inference 代码
-------------------

有了上一步得到的 MegEngine Lite 库文件，我们就可以在编译 Inference 代码的时候进行动态链接或静态链接。
下面分别用 Linux x86 和 Android Arm 来展示两种链接方式，演示编译 Inference 代码的步骤：

Linux x86 动态链接编译
~~~~~~~~~~~~~~~~~~~~~~

根据自身环境选择编译器（这里使用的是 clang++, 也可以用 g++），动态链接 ``liblite_shared.so`` 文件：

.. code-block:: bash

   export LITE_INSTALL_DIR=/path/to/megenginelite-lib #上一步中编译生成的库文件安装路径
   export LD_LIBRARY_PATH=$LITE_INSTALL_DIR/lib/x86_64/:$LD_LIBRARY_PATH

.. code-block:: bash

   clang++ -o demo_deploy \
     -I$LITE_INSTALL_DIR/include main.cpp \
     -llite_shared -L$LITE_INSTALL_DIR/lib/x86_64

编译完成之后，就得到了可执行文件 ``demo_deploy``.

Android Arm 静态链接编译
~~~~~~~~~~~~~~~~~~~~~~~~

Android Arm 编译为交叉编译（在 Linux 主机上编译 Android Arm 中运行的可执行程序）。

以链接 MegEngine Lite 的静态库作为示例，需要确保 NDK 环境准备完成，

.. code-block:: bash

   export LITE_INSTALL_DIR=/path/to/megenginelite-lib #上一步中编译生成的库文件安装路径
   export PATH=${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/:$PATH
   export CXX=aarch64-linux-android21-clang++

.. code-block:: bash

   ${CXX} -llog -lz -s \
     -I${LITE_INSTALL_PATH}/include main.cpp \
     ${LITE_INSTALL_PATH}/lib/aarch64/liblite_static_all_in_one.a \
     -o demo_deploy

编译完成之后，需要将 ``demo_deploy`` 和模型文件 ``shufflenet_v2.mge`` 拷贝到 Android Arm 机器上。

.. _lite-model-deploy:

执行 Inference 文件，验证结果
-----------------------------

最后执行编译好的文件，就可以看到推理结果：

.. code-block:: shell

   ./demo_deploy shufflenet_v2.mge

这样就快速完成了 X86 和 Arm 上简单的 demo 部署。

在本例中，最后计算结果可以看到：经过 ``softmax`` 之后，输出的结果中 ``sum = 1``, 符合 ``softmax`` 的输出特点。
