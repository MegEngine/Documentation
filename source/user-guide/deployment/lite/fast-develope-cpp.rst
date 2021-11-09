.. _fast-develope-cpp:

===============================================
快速上手MegEngine Lite C++部署模型
===============================================

MegEngine 的一大核心优势是 “训练推理一体化”，其中 “训练” 是在 Python 环境中进行的， 而 “推理” 则特指在 C++ 环境下使用训练完成的模型进行推理。
而将模型迁移到无需依赖 Python 的环境中，使其能正常进行推理计算，被称为 部署 。 部署的目的是简化除了模型推理所必需的一切其它依赖，使推理计算的耗时变得尽可能少，
比如手机人脸识别场景下会需求毫秒级的优化，而这必须依赖于 C++ 环境才能实现。

本文将从获取一个训练好的 shufflenet_v2 模型出发，讲解如何使用 MegEngine Lite 的 C++ 接口将其部署到 CPU（Linux x86 和 Android·Arm）环境下运行。

模型准备
--------

首先我们需要 :ref:`dump` ，这里主要使用 MegEngine 的 trace dump 功能将动态图 trace 为静态图，同时对静态图进行 Inference 相关的图优化。
下面将要用到的模型为 MegEngine 预训练的模型，来自 `模型中心 <https://megengine.org.cn/model-hub>`_ 。 安装 MegEngine 之后运行下面的 python 脚本将 dump 一个预训练的 shufflenet_v2.mge 模型。

.. code-block:: python

    import numpy as np
    import megengine.functional as F
    import megengine.hub
    from megengine import jit, tensor

    if __name__ == "__main__":
        net = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
        net.eval()

        @jit.trace(symbolic=True, capture_as_const=True)
        def fun(data, *, net):
            pred = net(data)
            pred_normalized = F.softmax(pred)
            return pred_normalized

        data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

        fun(data, net=net)
        fun.dump("shufflenet_v2.mge", arg_names=["data"])

上面代码最后 dump 模型时将模型命名为 shufflenet_v2.mge 并设置输入 Tensor 的名字为 "data"，后续将通过这个名字获取模型的输入 Tensor。

编写 Inference 代码
-------------------

首先创建一个 main.cpp，在这个 cpp 文件中将直接调用 MegEngine Lite 的接口运行 shufflenet_v2.mge 模型，输入数据是随机生成的，所以不用在乎计算结果。

.. code-block:: cpp

    #include <stdlib.h>
    #include <iostream>
    #include "lite/network.h"
    using namespace lite;

    int main(int argc, char** argv) {
        std::cout << " Usage: ./xornet_deploy model_name"
                << std::endl;
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

 * 创建默认配置的 Network

    .. code-block:: cpp

        std::shared_ptr<lite::Network> network = std::make_shared<Network>();


 * 模型载入，MegEngine Lite 将进行解析模型和计算图建立
 
    .. code-block:: cpp

        network->load_model(model_path);


 * 通过输入 Tensor 的名字获取模型的输入 Tensor，并设置随机数作为输入数据
 
    .. code-block:: cpp

        std::shared_ptr<Tensor> input_tensor = network->get_io_tensor("data");
        srand(static_cast<unsigned>(time(NULL)));
        size_t length =
                input_tensor->get_tensor_total_size_in_byte() / sizeof(float);
        float* in_data_ptr = static_cast<float*>(input_tensor->get_memory_ptr());
        for (size_t i = 0; i < length; i++) {
            in_data_ptr[i] =
                    static_cast<float>(rand()) / (static_cast<float>(RAND_MAX));
        }


 * 执行 Inference
 
    .. code-block:: cpp

        network->forward();
        network->wait();


 * 获取模型输出 Tensor，并处理输出数据，这里模型只有一个输出，直接调用 get_output_tensor 并传递 index=0
    
    .. code-block:: cpp

        std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
        float* predict_ptr = static_cast<float*>(output_tensor->get_memory_ptr());
        float sum = 0.0f, max = predict_ptr[0];
        for (size_t i = 0; i < 1000; i++) {
            sum += predict_ptr[i];
            if (predict_ptr[i] > max) {
                max = predict_ptr[i];
            }
        }

至此完成了一个 shufflenet_v2 模型的推理过程的 cpp 代码编写，真正运行起来，还需要编译该 cpp 源文件，并链接 MegEngine Lite 库文件。


编译 MegEngine Lite
-------------------

编译 MegEngine Lite 的目的是获得 MegEngine Lite 的静态链接库和动态链接库，供我们上面代码编译时候进行链接，这里在 Linux x86 下使用动态链接，Android Arm 上使用静态链接
这里的编译过程和 :ref:`从源码编译 MegEngine <build-from-source>` 是一样。

1. 首先需要 clone 整个 MegEngine 工程，并进入到 MegEngine 的根目录
2. 环境准备

   * Linux x86：运行 ./third_party/prepare.sh 脚本准备编译依赖的 submodule 和运行 ./third_party/install-mkl.sh 脚本安装 mkl。
   * Android Arm：准备编译依赖的 submodule 以及 NDK 环境

     1. 准备编译依赖的 submodule， 运行 ./third_party/prepare.sh 脚本。
     2. 从安卓 `官网 <https://developer.android.google.cn/ndk/downloads/>`_ 下载 NDK 并解压，并设置环境变量 export NDK_ROOT=/path/to/ndk。
3. 执行编译

   * Linux x86：运行脚本：scripts/cmake-build/host_build.sh
   * Android Arm：运行脚本：scripts/cmake-build/cross_build_android_arm_inference.sh

编译完成之后 MegEngine Lite 库和头文件安装在：

   * Linux x86：**库文件安装路径** build_dir/host/MGE_WITH_CUDA_OFF/MGE_INFERENCE_ONLY_ON/Release/install/lite/
   * Android Arm：**库文件安装路径** build_dir/android/arm64-v8a/Release/install/lite/

编译 Inference 代码，链接 MegEngine Lite 库文件，并运行
-------------------------------------------------------

Linux x86
^^^^^^^^^

Linux x86 动态链接 liblite_shared.so，需要根据自身环境选择编译器，下面以 clang++ ，你可以切换为 g++。

.. code-block:: bash

    export LITE_INSTALL_DIR= 上一步中编译生成的库文件安装路径
    clang++ -o demo_deploy -I$LITE_INSTALL_DIR/include main.cpp -L$LITE_INSTALL_DIR/lib/x86_64/ -llite_shared
    export LD_LIBRARY_PATH=$LITE_INSTALL_DIR/lib/x86_64/:$LD_LIBRARY_PATH
    ./demo_deploy shufflenet_v2.mge

完成之后将看到推理之后的输出结果。

Android Arm:
^^^^^^^^^^^^

Android Arm 中以链接 MegEngine Lite 的静态库作为示例，需要确保 NDK 环境准备完成，Android Arm 编译为交叉编译（在 Linux 主机上编译 Android Arm 中运行的可执行程序）。

.. code-block:: bash

    export PATH=${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/:$PATH
    export LITE_INSTALL_DIR=上一步中编译生成的库文件安装路径
    export CXX=aarch64-linux-android21-clang++
    ${CXX} -llog -lz -s -I ${LITE_INSTALL_PATH}/include main.cpp ${LITE_INSTALL_PATH}/lib/aarch64/liblite_static_all_in_one.a -o demo_deploy

编译完成之后，将 demo_deploy 和模型文件 shufflenet_v2.mge 拷贝到 Android Arm 机器上，运行 ./demo_deploy shufflenet_v2.mge 就可以看到结果。
 
这样就快速完成了 X86 和 Arm 上简单的 demo 部署，最后计算结果可以看到，经过 softmax 之后，输出的结果中 sum = 1，符合 softmax 的输出特点。
