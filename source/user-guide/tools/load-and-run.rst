.. _load-and-run:

============================
如何使用 Load and Run（C++）
============================

Load and Run （简称 LAR）是 MegEngine 中的加载并运行模型的工具，具有以下功能：

* 编译出对应各个平台的二进制文件，可对比相同模型的速度；
* 测试验证不同模型优化方法的效果（直接执行 ``./load_and_run`` 显示相应帮助文档）；

.. note::

   * 目前发布的版本我们开放了对 CPU（x86, x64, ARM, ARMv8.2）和 GPU（CUDA）平台的支持。
   * 如二进制文件体积较大不利于使用，可选择使用 Load and Run 的 :ref:`Python 版本 <load-and-run-py>` 。

编译 Load and Run
-----------------

我们以 x86 和 ARM 交叉编译为例进行说明：

Linux x86 平台编译
~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   git clone https://github.com/MegEngine/MegEngine.git
   cd MegEngine && mkdir build && cd build
   cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=OFF
   make -j$(nproc)

编译完成后，我们可以在 ``build/sdk/load_and_run`` 目录找到 ``load_and_run`` .

Linux 交叉编译 ARM 版本
~~~~~~~~~~~~~~~~~~~~~~~
.. warning::

   请确保你的机器上已经设置好了 Android 所需开发环境：

   #. 到 Android 的官网下载 `NDK <https://developer.android.com/ndk/downloads>`_ 及相关工具，
      这里推荐 android-ndk-r21 以上的版本；
   #. 在 BASH 中设置 NDK_ROOT 环境变量：``export NDK_ROOT=ndk_dir``

在 Ubuntu (16.04/18.04) 用以下脚本进行 ARM-Android 的交叉编译：

.. code-block:: bash

   ./scripts/cmake-build/cross_build_android_arm_inference.sh

编译完成后，我们可以在 ``build_dir/android/arm64-v8a/release/install/bin/load_and_run`` 
目录下找到编译生成的可执行文件 ``load_and_run`` . 

.. note::

   * 上面的脚本默认没有开启 ARMv8.2-A+DotProd 的新指令集支持，
     如果在一些支持的设备（如 Cortex-A76 等），可以开启相关选项：
     
   .. code-block:: bash

        ./scripts/cmake-build/cross_build_android_arm_inference.sh -p

   * :ref:`量化模型 <quantization>` 推荐开启 ARMv8.2+DotProd 支持，
     能够充分利用 DotProd 指令集硬件加速。

查看 ``cross_build_android_arm_inference.sh`` 脚本源码可以了解更多选项的设置方法。

使用 Load and Run
-----------------
.. note::
   
   使用之前，需要先将模型文件的输入、:ref:`Dump <dump>` 出的预训练模型文件和 
   load_and_run (以及依赖 ``.so`` 的文件) 传到手机，并设置好环境变量 ``LD_LIBRARY_PATH`` . 
   示例代码如下：

   .. code-block:: bash

      adb push data.npy /data/local/tmp
      adb push model.mge /data/local/tmp
      adb push build_dir/android/arm64-v8a/release/install/bin/load_and_run /data/local/tmp
      adb push build_dir/android/arm64-v8a/release/install/lib/libmegengine.so /data/local/tmp
      adb shell && cd /data/local/tmp/ && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

举例说明，使用 Load and Run 的基础语法如下:

.. code-block:: bash

   ./load_and_run ./model.mge --input data.npy --iter 10

其中有几个基础参数：

``net``
  指定 mge graph 路径，例子中为 ``./model.mge``.

``--input INPUT_DATA``
  指定用作输入的 inputs data 路径，例子中为 ``./data.npy``.
  
  输入格式支持 ``.ppm/.pgm/.json/.npy`` 等文件格式和命令行。

``--iter ITER``
  正式运行测速的迭代数，例子中为 ``10``.

进阶参数设置
------------

.. _layout-optimize:

平台相关 Layout 优化
~~~~~~~~~~~~~~~~~~~~

``--enable-nchw44``
  目前 MegEngine 的网络是 NCHW 的 Layout, 但是这种 Layout 不利于充分利用 SIMD 特性，且边界处理异常复杂。
  为此我们针对 ARM 开发了 NCHW44 的 Layout.

``--enable-nchw88``
  如上所述，对于 x86 AVX 下，我们同样定义了 NCHW88 的 Layout 优化。

.. _fastrun:

开启 fastrun 模式
~~~~~~~~~~~~~~~~~

目前在 MegEngine 中，针对某些算子存在很多种不同的算法
（如 conv 存在 direct, winograd 或者 im2col 等算法），
而这些算法在不同的 shape 或者不同的硬件平台上，其性能表现差别极大，
导致很难写出一个有效的搜索算法，在执行时选择到最快的执行方式。
为此在 MegEngine 中集成了 fastrun 模式，
**在执行模型的时候会将每个算子的可选所有算法都执行一遍，然后选择一个最优的算法记录下来。**

.. note::

   整体来讲大概有 10% 的性能提速。

使用 fastrun 一般分为两个阶段，**需要顺序执行。**

搜参阶段：

``--fast-run [--winograd-transform] --fast-run-algo-policy CACHE_FILE``
  开启 fastrun 模式，同时将输出的结果存储到一个 cache 文件中

  其中 ``--winograd-transform`` 为可选项目，
  由于对于相同的卷积，多种 winograd 算法的理论加速比和实际性能表现有时会不一致，
  开启该选项可使其基于 fastrun 模式搜索的结果来决定做哪种 winograd 变换。

运行阶段：

``--fast-run-algo-policy CACHE_FILE``
  执行阶段: 带上之前的 cache 文件再次执行


正确性验证
----------

MegEngine 内置了多种正确性验证的方法，方便检查网络计算正确性。

开启 asserteq 验证正确性
~~~~~~~~~~~~~~~~~~~~~~~~

可以基于脚本 ``dump_with_testcase_mge.py`` 将输入数据和运行脚本时
使用当前默认的计算设备计算出的模型结果都打包到模型里， 这样在不同平台下就方便比较结果差异了。

.. code-block:: bash

    python3 $MGE/sdk/load_and_run/dump_with_testcase_mge.py ./model.mge --optimize -d data.jpg -o model.mdl

在执行 load_and_run 的时候就不需要再带上 ``--input`` ，因为输入已经打包进 ``model.mdl`` ,
同时在执行 ``dump_with_testcase_mge.py`` 脚本的时候，会在 XPU (如果有 GPU, 就在 GPU 上执行，
如果没有就在 CPU 上执行) 执行整个网络，将结果作为 ``ground-truth`` 写入模型中。

该脚本可用参数如下：

``input``
  **必须参数** ，执行需要添加输入的MegEngine模型文件地址

``-d --data``
  **必须参数** ，指定模型的输入数据，指定方法为：``<input0 name>:<data0>;<input1 name>:<data1>...`` 
  当模型只有一个输入，则可以省略 input 的名字。数据支持以下三种类型——

  #. 使用随机数据，以 "#rand" 开头：

     - 仅指定输入数据的最大最小值，其中 shape 由输入模型推出：--data #rand(0,255) 
     - 指定输入数据的最大最小值和 batchsize，其中 shape 由输入模型推出
       （注意省略号不可省略）：–data #rand(0,255,1,...)
     - 指定输入数据的全部维度：–data #rand(0,255,1,3,224,224)

  #. 使用图片或者 ``npy`` 文件：

     - 使用图片：--data image.png
     - 使用 npy：--data image.npy

  #. 使用包含多条数据的文本文件，以 "@" 开头，文件中的每一行都符合上面两种形式：--data image.txt

     image.txt里面的内容可能是这样的：

     .. code-block:: none

        var0:image0.png;va1:image1.npy
        var0:#rand(0,255);var1:image2.png

``-o --output``
  **必需参数** ，指定输出模型地址

``--repeat``
  默认值为 1，指定 -d 传递的输入数据会重复多少份，常用于性能测试。

``--silent``
  默认为 false，在启用推理正确性检查的时候，是否输出更加简洁的检查信息。比如说展示误差最大值。

``--optimize-for-inference``
  默认为 false，是否开启计算图优化，经过优化后的图结构可能会发生改变，但是可以获得更好地推理性能，
  详见 :ref:`optimieze-for-inference-options` 。

``--no-assert``
  默认为 false，是否禁用推理正确性检查，常用于性能测试。
  assert 比较的对象为：输入模型 + 输入数据的推理结果 VS 输出模型（此时数据已纳入模型中）的推理结果。

``--maxerr``
  默认为 1e-4，在开启推理正确性检查时允许的最大误差。

``--resize-input``
  默认为 false，是否采用 cv2 库把输入图片的尺寸 resize 到模型要求的输入尺寸。

``--input-transform``
  可选参数，有用户指定的一行 python 代码，用于操作输入数据。比如 ``data/np.std(data)`` .

``--discard-var-name``
  默认为 false，是否丢弃输入模型的变量 (varnode) 和参数 (param) 的名字。

``--output-strip-info``
  默认为 false，是否保存模型的输出信息到 JSON 文件，默认路径为输出模型名 + ".json" .
  文件中包含模型 hash 码，所有输出的 opr 类型和计算数据类型。

dump 输出结果
~~~~~~~~~~~~~
``--bin-out-dump``
  在指定的文件夹内保存输出结果，可以用 load-and-run 在目标设备上跑数据集

使用方式如下：

.. code-block:: bash

    mkdir out
    ./load_and_run ./model.mge --input ./data.npy --iter 2 --bin-out-dump out

然后可以在 python 里打开输出文件：

.. code-block:: python

   import megengine as mge

   v0 = mge.utils.load_tensor_binary('out/run0-var1602')
   v1 = mge.utils.load_tensor_binary('out/run1-var1602')

dump 每层结果
~~~~~~~~~~~~~
我们很多时候会遇到这种情况，就是模型输出结果不对，
这个时候就需要打出网络每一层的结果作比对，看看是哪一层导致。
目前有两种展现方式，一个是 ``io-dump``, 另一个是 ``bin-io-dump``.

为了对比结果，需要假定一个平台结果为 ``ground-truth`` ，
下面假定以 x86 的结果为 ``ground-truth`` ，验证 x86 和 CUDA 上的误差产生的原因
（下面会使用 ``host_build.sh`` 编译出来的 ``load_and_run`` 来演示）。

文本形式对比结果：

.. code-block:: bash

    ./load_and_run ./model.mge --input data.npy --iter 10 --cpu --io-dump cpu.txt
    ./load_and_run ./model.mge --input data.npy --iter 10 --io-dump cuda.txt # 默认跑在cuda上
    vimdiff cpu.txt cuda.txt

文档形式只是显示了部分信息，比如 Tensor 的前几个输出结果，整个 Tensor 的平均值、标准差之类，
如果需要具体到哪个值错误，需要用 ``bin-io-dump`` 会将每一层的结果都输出到一个文件。

raw 形式对比结果：

.. code-block:: bash

    mkdir cpu && mkdir cuda
    ./load_and_run ./model.mge --input data.npy --iter 10 --cpu --bin-io-dump cpu
    ./load_and_run ./model.mge --input data.npy --iter 10 --bin-io-dump cuda
    $mge/tools/compare_binary_iodump.py cpu cuda

如何进行性能调优
---------------- 

Load and Run 支持传入 ``--profile`` 参数：

``--profile PROFILE``
  开启后使用 GraphProfiler 记录 profile 信息并将结果的 json 内容写到 PROFILE 文件路径中

该 ``PROFILE`` 文件可后续用于 profile_analyze.py 分析

megengine.utils.profile_analyze 的示例用法：

.. code-block:: bash

    # 输出详细帮助信息
    python3 -m megengine.utils.profile_analyze -h

    # 输出前 5 慢的算子
    python3 -m megengine.utils.profile_analyze ./profiling.json -t 5

    # 输出总耗时前 5 大的算子的类型
    python3 -m megengine.utils.profile_analyze ./profiling.json -t 5 --aggregate-by type --aggregate sum

    # 按 memory 排序输出用时超过 0.1ms 的 ConvolutionForward 算子
    python3 -m megengine.utils.profile_analyze ./profiling.json -t 5 --order-by memory --min-time 1e-4  --type ConvolutionForward

输出将是一张表格，每列的含义如下：

``device self time``
  算子在计算设备上（例如 GPU ）的运行时间

``cumulative``
  累加前面所有算子的时间

``operator info``
  打印算子的基本信息

``computation``
  算子需要的浮点数操作数目

``FLOPS`` 
  算子每秒执行的浮点操作数目，由 ``computation`` 除以 ``device self time`` 并转换单位得到

``memory``
  算子使用的存储（例如 GPU 显存）大小

``bandwidth``
  算子的带宽，由 ``memory`` 除以 ``device self time`` 并转换单位得到

``in_shapes``
  算子输入张量的形状

``out_shapes``
  算子输出张量的形状

