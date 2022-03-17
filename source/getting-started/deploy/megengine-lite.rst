.. _deploy-megengine-lite:

============================
使用 MegEngine Lite 进行推理
============================
.. admonition:: 本教程涉及的内容
   :class: note

   * （可选）理解静态图和动态图的区别，学会利用 :class:`~.trace` 接口完成转换；
   * （可选）学会将静态图模型借助 :meth:`~.trace.dump` 接口序列化并导出 ``.mge`` 文件；
   * 理解 megenginelite 的基本用法，使用它代替上一个教程中所用到的 Python 接口，并进行推理；
   * 理解 MegEngine Lite 的基本用法，学会在本机环境编译 Lite 并作为 C++ 库使用。

什么是 MegEngine Lite
---------------------

MegEngine Lite 是 MegEngine 的一层接口封装，主要目的是为用户提供更加简洁、易用、高效的推理接口，
充分发挥 MegEngine 的多平台的推理能力。更加详细的介绍请参考 :ref:`megengine-lite` 中的介绍。

本教程将衔接上一个教程，向你介绍 MegEngine Lite 最基本的用法，确保你跑通整个流程。

前置准备：获取模型文件
~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: 开发阶段和部署阶段的 “模型文件” 是不同的概念
   :class: warning

   想要使用 MegEngine Lite 进行模型部署和推理，则需要有对应的模型文件。
   这里提到的模型文件与模型开发中所提到的用 
   :func:`~megengine.save` 和 :func:`~megengine.load`
   接口保存和加载的 Pickle 格式的模型（权重）文件是不同的概念。
   这一小节主要介绍如何将已经开发完成的模型变成 Lite 可用的模型文件。

   如果在工程团队中已经由上游直接提供了 ``.mge`` 文件，则可以跳过这一小节。⬇️  ⬇️  ⬇️

.. seealso::

   * 我们将以使用 :mod:`~.hub` 模块获取到的 ShuffleNet 预训练模型为例进行说明；
   * MegEngine 官网模型中心对 `ShuffleNet V2
     <https://megengine.org.cn/model-hub/megengine_vision_shufflenet_v2>`_
     有详细的使用介绍（但没有用到 Lite 接口进行推理）；
   * 完整的脚本代码在 :docs:`examples/deploy/lite/model.py`,
     默认将会得到 ``snetv2_x100_deploy.mge`` 文件。

#. 获取预训练模型（你也可以加载使用自己开发的模型），切换到评估模式，以便用于推理：

   .. code-block:: python

      import megengine.hub as hub
   
      net = hub.load("megvii-research/basecls", "snetv2_x100", pretrained=True)
      net.eval()

   不清楚这一步骤的用户可以参考 :ref:`hub-guide` 中的介绍。

#. 编写模型推理步骤的函数代码，并使用 :class:`~.trace` 进行装饰：

   .. code-block:: python

      from megengine.jit import trace
      import megengine.functional as F
   
      @trace(symbolic=True, capture_as_const=True)
      def infer_func(data, *, model):
          pred = model(data)
          pred_normalized = F.softmax(pred)
          return pred_normalized

   通过引入一行代码，我们就已经将模型从动态图模式切换到了静态图模式。
   关于这一部分的概念和原理，已经在 :ref:`jit-guide` 和 :ref:`trace` 
   文档页面中进行了详细的解释。

   生成随机数据（注意要满足输入形状），并实际执行一次上面的函数，以便得到静态图信息：

   .. code-block:: python

      import numpy as np
      from megengine import Tensor

      data = np.random.random([1, 3, 224, 224]).astype(np.float32)
      infer_func(Tensor(data), model=net)

#. 将静态图通过 :meth:`~.trace.dump` 接口序列化导出成最终所需要的 ``.mge`` 模型：

   .. code-block::

      infer_func.dump("snetv2_x100_deploy.mge", arg_names=["data"])

   你会在当前目录下得到 ``snetv2_x100_deploy.mge`` 文件。
   
.. note::

   为了方便和 Lite 接口调用结果对比，实际执行脚本时默认会使用一张暹罗猫图片作为输入样本，
   并推理得到 Top-5 分类的预测结果（此处数值仅供参考，不同设备上可能会存在差异）。

   .. figure:: ../../_static/images/cat.jpg
      :height: 250

   .. code-block::

      0: class = Siamese_cat          with probability = 20.2 %
      1: class = lynx                 with probability = 13.3 %
      2: class = Egyptian_cat         with probability =  9.5 %
      3: class = Persian_cat          with probability =  4.9 %
      4: class = Angora               with probability =  3.0 %

使用 megenginelite 验证
~~~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   在上一个教程《 :ref:`deploy-fastapi` 》中我们展示了如何快速部署一个手写数字分类模型，
   当时使用的是 MegEngine 原生接口，而在这一小节，我们将展示 MegEngine Lite 的 Python 接口用法，
   用户可以自行比对两种做法之间的流程和代码差异，并尝试再次用 FastAPI 进行快速部署。

.. seealso:: 

   这里只展示核心逻辑，完整脚本代码在： :docs:`examples/deploy/lite/inference.py`

``megenginelite`` 包提供 MegEngine Lite 的 Python 接口，伴随着 MegEngine Python 包存在：

>>> import megenginelite

在 Lite 中，我们通过 ``LiteNetwork`` 创建一个网络：

>>> from megenginelite import LiteNetwork

接下来要做的便是将上一步导出的静态图模型加载到 Lite 中：

>>> network = LiteNetwork()
>>> network.load("snetv2_x100_deploy.mge")

推理的步骤和 MegEngine 原生接口基本一致，区别在于数据此时是 ``LiteTensor`` 而非 :class:`~.Tensor`:

.. literalinclude:: ../../../examples/deploy/lite/inference.py
   :lines: 35-53

如上操作，我们便能够使用 megenginelite 做到：给定一张图片，输出对它在 1000 个类别上的概率预测。

执行脚本，使用暹罗猫的图片进行验证：

.. code-block:: shell

   python3 inference.py -m snetv2_x100_deploy.mge

.. code-block::

   0: class = Siamese_cat          with probability = 20.2 %
   1: class = lynx                 with probability = 13.3 %
   2: class = Egyptian_cat         with probability =  9.5 %
   3: class = Persian_cat          with probability =  4.9 %
   4: class = Angora               with probability =  3.0 %

可以发现得到的结果与我们使用 MegEngine 原生接口推理结果一致。

.. dropdown:: :fa:`question,mr-1` 这和直接使用 MegEngine 原生接口相比，有什么区别

   由于此时我们使用的是静态图模型，推理用时相较于动态图会更少，资源占用也会更低。
   相较于原生接口，使用 megenginelite 是更加推荐的一种推理方式，能够满足我们对于推理性能的基本要求。

编译 MegEngine Lite
-------------------

.. seealso:: 

   使用 Lite 的 Python 接口固然方便，但我们仍然有更加复杂的需求情景，
   比如追求更高的性能、更低的资源占用，以及需要能够跨平台移植，这时我们则需要自行编译 Lite C++ 库。
   在本小节，我们将展示 Linux x86 环境下本地编译的流程。
   在后续的教程中，我们还将演示交叉编译流程，以便将模型部署到像 Android 和 IOS 这样的移动端设备上。
   相关介绍的详细版本均可在 :ref:`build-from-source` 页面中找到。

请自行参考上述链接，完成 MegEngine Lite 的 Host Build 过程，
最终应该在 Build 出的文件夹中得到 Lite 库和头文件，简记作 ``/path/to/megenginelite-lib``,
将其作为环境变量导出到当前的环境中：

.. code-block:: shell

   export LITE_INSTALL_DIR=/path/to/megenginelite-lib
   export LD_LIBRARY_PATH=$LITE_INSTALL_DIR/lib/x86_64/:$LD_LIBRARY_PATH

我们在这一步构建出的 Lite C++ 库将会在编译和链接 ShuffleNet 模型的 C++ 推理代码时用到。

编写 C++ 推理代码验证
---------------------

.. admonition:: 本小节的演示逻辑

   * 我们将先用随机数据作为输入，通过 Softmax 计算来验证 MegEngine C++ 库的正确性；
   * 然后，我们将引入 OpenCV 等 C++ 库，进行完整的 ShuffleNet 模型推理实现。

使用随机数据作为输入
~~~~~~~~~~~~~~~~~~~~

MegEngine Lite 的 C++ 接口和 Python 接口设计基本一致，因此整个流程相似：

.. dropdown:: :fa:`eye,mr-1` 展开完整代码

   源代码位置：:docs:`examples/deploy/lite/main.cpp`

   .. literalinclude:: ../../../examples/deploy/lite/main.cpp
      :language: cpp

编译并运行这份代码，将会看到输出 SUM 为 1, 符合 Softmax 的计算结果：

.. code-block:: shell

   clang++ -o lite_softmax \
           -I$LITE_INSTALL_DIR/include main.cpp \
           -llite_shared -L$LITE_INSTALL_DIR/lib/x86_64
   ./lite_softmax snetv2_x100_deploy.mge

使用图片数据作为输入
~~~~~~~~~~~~~~~~~~~~

.. admonition:: 使用 C++ 推理时的注意事项
   :class: warning

   由于我们在执行推理前，存在着一些对输入图片进行预处理的步骤，
   可能存在着对应的 Python 接口在 C++ 环境中没有提供，需要自行比对实现。
   为了保证整个推理过程的一致性，我们需要保证输入数据的预处理也是一致的。
   MegEngine 中对图片的预处理底层使用到了 ``opencv-python``,
   其本质上是对 OpenCV 的 C++ 库做了语言绑定，在底层调用了一致的接口。
   比如 ``imread``, ``resize`` 等等...

   同理，如果你的输入数据是其它的格式，也需要保证预处理的步骤是一致的，
   这样输入模型中的 Lite Tensor 的初始状态才能与 Python 推理一致。

.. admonition:: 将预处理流程作为模型的一部分

   另一种处理方式是，在被 :class:`~.trace` 的推理函数中加入预处理逻辑，
   即图片读入转为 Tensor 后，借助 :mod:`~.functional` 模块，
   在模型最开始进行预处理操作，这样就不用考虑编写一致的 C++ 预处理代码。

接下来我们需要用真实的图片进行验证，过程中需要用到 OpenCV C++ 库来处理图片：

.. code-block:: shell

   export OpenCV_DIR=/path/to/opencv-lib/

获取 OpenCV C++ 库的步骤不会在这里进行介绍，请参考 `OpenCV
<https://github.com/opencv/opencv>`_ 文档进行操作。
为了保证 Lite 的推理结果高度一致，建议在同样的环境下进行相同版本 OpenCV 的编译。
如果使用第三方提供的预编译版本，可能会导致图片读取进来的像素值存在细微差异，
最终导致推理的结果值有较大误差（尽管可能依旧预测出正确的分类）。

.. dropdown:: :fa:`eye,mr-1` 展开完整代码

   源代码位置：:docs:`examples/deploy/lite/inference.cpp`

   .. literalinclude:: ../../../examples/deploy/lite/inference.cpp
      :language: cpp

   在同一目录下提供了 CMakeLists.txt 文件，可点开了解细节。

我们在这一步选择借助 CMake 来编译构建 C++ 推理代码：

.. code-block:: shell

   cmake -S . -B build
   cmake --build build

你会在 `bin` 目录下得到 ``inference`` 二进制文件，执行推理：

.. code-block:: shell

   ./bin/inference snetv2_x100_deploy.mge ../../../source/_static/images/cat.jpg

就能看到最终的推理结果，对暹罗猫图片的类别进行了正确的预测。

.. code-block:: 

   The class tabby, tabby catwith probability = 2.5972%
   The class Persian catwith probability = 4.90259%
   The class Siamese cat, Siamesewith probability = 20.1514%
   The class Egyptian catwith probability = 9.49989%
   The class lynx, catamountwith probability = 13.277%
   The class Angora, Angora rabbitwith probability = 2.99404%
   The final predicted class is Siamese cat, Siamese with probability = 20.1514%

拓展材料
--------

.. dropdown:: :fa:`question,mr-1` 一定需要引入 OpenCV 第三方依赖吗

   在某些情景下，我们在开发 C++ 程序时希望尽可能地减少对第三方库的依赖。
   例如对于 ImageNet 的图像 Label 解析过程，官方提供的原本是一个 JSON 
   格式的文件，我们可以通过引入 `cJSON <https://github.com/DaveGamble/cJSON>`_
   来灵活地进行处理。但事实上在本教程中，我们选择了将 Label 处理成单个头文件。

   同理，思考一下我们对图片的处理过程，是否一定需要用到 OpenCV 呢？
   我们可以尝试使用 `stb <https://github.com/nothings/stb>`_ 来完成图片的读取逻辑，
   而 ``resize`` / ``centerCrop`` 等预处理步骤可以实现在 Tensor 推理逻辑中
   （注意此时应当使用 :mod:`~.functional` 中的接口来实现 :mod:`~.data.transform` 步骤），
   借助 MegEngine 的 :class:`~.trace` 和 :meth:`~.trace.dump` 机制序列化，
   就不用再实现对应的 C++ 预处理代码了，你可以自行尝试实现上述完整流程。

   在下一个 Andoird 部署教程中，我们将展示类似的做法。

