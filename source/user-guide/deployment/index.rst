.. _deployment:

=====================
将模型部署到 C++ 环境
=====================

MegEngine 的一大核心优势是 “训练推理一体化”，其中 “训练” 是在 Python 环境中进行的，
而 “推理” 则特指在 C++ 环境下使用训练完成的模型进行推理。
而将模型迁移到无需依赖 Python 的环境中，使其能正常进行推理计算，被称为 **部署** 。
部署的目的是简化除了模型推理所必需的一切其它依赖，使推理计算的耗时变得尽可能少，
比如手机人脸识别场景下会需求毫秒级的优化，而这必须依赖于 C++ 环境才能实现。

本文将从一个训练好的异或（XOR）网络模型出发，讲解如何将其部署到 CPU（X86）环境下运行。

将模型序列化并导出
------------------

首先我们需要 :ref:`dump` , 用到的模型定义与序列化代码在 :src:`sdk/xor-deploy/xornet.py` . 

编写 C++ 程序读取模型
---------------------

接下来我们需要编写一个 C++ 程序，来实现我们期望在部署平台上完成的功能。

继续以上面导出的异或网络模型为例子，我们实现一个最简单的功能——
给定两个浮点数，输出对其做异或操作后结果为 0 的概率（以及为 1 的概率）。

在此之前，为了能够正常使用 MegEngine 底层 C++ 接口，
需要先按照 MegeEngine 中提供的编译脚本( :src:`scripts/` ) 从源码编译得到 MegEngine 的相关库, 
通过这些脚本可以交叉编译安卓（ARMv7，ARMv8，ARMv8.2）版本、Linux 版本（ARMv7，ARMv8，ARMv8.2）以及 iOS 相关库，
也可以本机编译 Windows/Linux/MacOS 相关库文件。

实现上述异或计算的示例 C++ 代码如下（引自 ``xor-deploy.cpp`` ）：

.. code-block:: cpp

   #include <stdlib.h>
   #include <iostream>
   #include "megbrain/serialization/serializer.h"
   using namespace mgb;

   cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                         HostTensorND& host) {
       auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
       return {dev, cb};
   }

   int main(int argc, char* argv[]) {
       std::cout << " Usage: ./xornet_deploy model_name x_value y_value"
                 << std::endl;
       if (argc != 4) {
           std::cout << " Wrong argument" << std::endl;
           return 0;
       }
       std::unique_ptr<serialization::InputFile> inp_file =
               serialization::InputFile::make_fs(argv[1]);
       float x = atof(argv[2]);
       float y = atof(argv[3]);
       auto loader = serialization::GraphLoader::make(std::move(inp_file));
       serialization::GraphLoadConfig config;
       serialization::GraphLoader::LoadResult network =
               loader->load(config, false);
       auto data = network.tensor_map["data"];
       float* data_ptr = data->resize({1, 2}).ptr<float>();
       data_ptr[0] = x;
       data_ptr[1] = y;
       HostTensorND predict;
       std::unique_ptr<cg::AsyncExecutable> func =
               network.graph->compile({make_callback_copy(
                       network.output_var_map.begin()->second, predict)});
       func->execute();
       func->wait();
       float* predict_ptr = predict.ptr<float>();
       std::cout << " Predicted: " << predict_ptr[0] << " " << predict_ptr[1]
                 << std::endl;
   }

简单解释一下代码的意思：

* 我们首先通过 ``GraphLoader`` 将模型加载进来，
* 接着通过 ``tensor_map`` 和上节指定的输入名称 ``data`` ，找到模型的输入指针，
* 再将运行时提供的输入 ``x`` 和 ``y`` 赋值给输入指针，
* 然后我们使用 ``network.graph->compile`` 将模型编译成一个函数接口，并调用执行，
* 最后将得到的结果 ``predict`` 进行输出，该输出的两个值即为异或结果为 0 的概率以及为 1 的概率。

另外可以配置上面加载模型时候的 ``config`` 来优化 Inference 计算效率，
为了加速一般在 ARM 上面配置 ``enable_nchw44_layout()`` ,
在x86 CPU 上面配置 ``enable_nchw88_layout()`` ，具体的配置方法参考 :ref:`load-and-run` .

编译与执行
----------

为了更完整地实现 “训练推理一体化”，我们还需要支持同一个 C++ 程序能够交叉编译到不同平台上执行，而不需要修改代码。
之所以能够实现不同平台一套代码，是由于底层依赖的算子库（内部称作 MegDNN）实现了对不同平台接口的封装，
在编译时会自动根据指定的目标平台选择兼容的接口。

.. note::

    目前发布的版本我们开放了对 CPU（X86、X64、ARMv7、ARMv8、ARMv8.2）
    和 GPU（CUDA）平台的 float 和量化 int8 的支持。

我们在这里以 CPU（X86）平台为例，首先直接使用 gcc 或者 clang （用 ``$CXX`` 指代）进行编译即可：

.. code-block:: bash

    $CXX -o xor_deploy -I$MGE_INSTALL_PATH/include xor_deploy.cpp -L$MGE_INSTALL_PATH/lib/ -lmegengine

上面的 ``$MGE_INSTALL_PATH`` 指代了编译安装时通过 ``CMAKE_INSTALL_PREFIX`` 指定的安装路径。
编译完成之后，通过以下命令执行即可：

.. code-block:: bash

    LD_LIBRARY_PATH=$MGE_INSTALL_PATH/lib:$LD_LIBRARY_PATH ./xor_deploy xornet_deploy.mge 0.6 0.9

这里将 ``$MGE_INSTALL_PATH/lib`` 加进 ``LD_LIBRARY_PATH`` 环境变量，确保 MegEngine 库可以被编译器找到。
上面命令对应的输出如下：

.. code-block:: none

    Predicted: 0.999988 1.2095e-05

至此我们便完成了从 Python 模型到 C++ 可执行文件的部署流程，
如果需要快速的运行模型以及测试模型性能，请参考 :ref:`load-and-run` .
