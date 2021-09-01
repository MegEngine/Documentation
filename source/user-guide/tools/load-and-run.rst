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
目录下找到编译生成的可执行文件 ``load_and_run`` . 查看脚本源码可以了解更多选项的设置方法。

.. note::

   * 上面的脚本默认没有开启 ARMv8.2-A+DotProd 的新指令集支持，
     如果在一些支持的设备（如 Cortex-A76 等），可以开启相关选项：
     
   .. code-block:: bash

        ./scripts/cmake-build/cross_build_android_arm_inference.sh -p

   * :ref:`量化模型 <quantization-guide>` 推荐开启 ARMv8.2+DotProd 支持，
     能够充分利用 DotProd 指令集硬件加速。

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

   from megengine.tools.compare_binary_iodump import load_tensor_binary

   v0 = load_tensor_binary('out/run0-var1602')
   v1 = load_tensor_binary('out/run1-var1602')

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
  记录信息并将结果的 ``JSON`` 内容写到 ``PROFILE`` 文件路径中

该 ``PROFILE`` 文件可后续用于 :ref:`profile-analyze` 。

