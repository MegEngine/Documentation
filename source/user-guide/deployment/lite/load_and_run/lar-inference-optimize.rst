.. _lar-inference-optimize:

使用 Load and run 进行推理优化实验
==================================

MegEngine 提供了大量的推理优化设置，针对不同的模型设置合适的推理优化选项能够十分有效的提升推理时的性能.
Load and run 提供了很多推理优化的接口，可以用于探索各种设置对与推理性能的影响，获取在特定模型，
特定后端上比较合适的优化设置方案，为模型最终部署时性能的保证提供相应依据。下面就常用的一些优化设置方法进行介绍。

fast-run 优化
----------------------

MegEngine 在底层提供了一套用于在多种算法中选择最优算法的一种机制， 称为 fast-run 机制。
该机制为具有多种算法实现的算子，如 conv，matmul 等，提供了一套算法选择的机制。
通过设置相应的算法选择策略实现对不同模型多算法算子的配置，从而达到推理优化的目标。
其具体使用方法参考 :py:func:`~.set_execution_strategy` , :ref:`execution_optimize` ，这里主要介绍 Load and run 中的使用。选项说明参考 :ref:`fast-run-options`

在优化后的算法中根据 profile 的时间选择算法( fast-run 策略)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MegEngine 在通用算法的基础上为 conv，matmul 以及 pooling 等算子，提供了大量优化的算法。
。这些算法在性能上往往优于通用的算法实现，fast-run 策略在就是在这些优化的算法中根据 profile 的时间，选择最优的一种。

.. note::

   * fast-run 机制发生在模型推理之前，因为需要 profile 各个算法的性能，所以算法搜索过程成中会有一定耗时。
   * fast-run 提供了一套离线缓存机制来减少这部分耗时，该机制将一次或多次搜索的结果缓存下来，下次运行时可以直接加载缓存中的搜索结果，进而减少搜索耗时

**使用 Load and run 运行 fast-run 策略**

.. code:: bash

   load_and_run <model_path> --input <input_name:input_data_description> --fast-run

对于 float32 的 resnet50 网络模型，在 cuda 上，使用 fast-run 和不使用 fast-run 时各个阶段的用时（多次测试的平均值）如下：

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * -  
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 开启 fast-run
     - 1.4
     - ``1821.5``
     - ``3.9``
   * - 不开启 fast-run
     - 1.6
     - 28.5
     - 5.2

可以看出开启 fast-run 时，在模型预热过程耗时巨大，但经过算法搜索后，模型推理时的性能时有一定提升的。模型预热过程中的耗时可以通过离线缓存机制来减少。

**获取离线缓存**

.. code:: bash

   load_and_run <model_path> --input <input_name:input_data_description> --fast-run --fast-run-algo-policy <algo_cache_file>

**使用离线缓存**

.. code:: bash

   load_and_run <model_path> --input <input_name:input_data_description> --fast-run-algo-policy <algo_cache_file>

这时模型运行各个阶段的耗时如下：

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * -  
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 使用 fast-run 离线缓存
     - 1.7
     - 28.9
     - ``3.8``
   * - 不使用 fast-run 离线缓存
     - 1.5
     - 28.7
     - 4.9

可以看出在保证模型配置和模型预热时间基本不变的情况下，fast-run 策略与离线缓存机制组合使用可以有效的提高模型推理时的性能。

直接根据 profile 的时间选择算法( full-run 策略)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

full-run 策略与fast-run 策略类似，差别在与算法选择的范围更大，包含了算子的一些通用算法。

同样有对于 float32 的 resnet50 网络模型，在 cuda 上，使用 full-run 和不使用 full-run 时各个阶段的用时（多次测试的平均值）如下：

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * -  
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 开启 full-run
     - 1.6
     - ``1964.4``
     - ``3.8``
   * - 开启 fast-run
     - 1.8
     - ``1883.1``
     - ``3.8``
   * - 不开启 full-run 或 fast-run
     - 1.5
     - 28.6
     - 4.9

可以看出，full-run 在模型预热阶段相比于 fast-run 耗时增大。
而使用离线缓存后，full-run 与 fast-run 策略的模型运行各个阶段的耗时相差不多，但同样的，相比于不使用离线缓存是有性能提升的。


layout相关优化
----------------------

对于不同的推理后端以及不同数据类型，不同的内存布局 （layout）对于提高数据搬运的效率，最大化计算资源的利用有着很大的影响。
如：NVIDIA显卡上 fp32 的 conv 在 NCHW 的 layout 下性能最佳，但 int8 的 conv 在 NCHW4 下的 layout 下性能最佳。
因而针对不同的推理后端以及不同的数据类型选择合适的 layout 对于模型的推理性能也会产生很大的提升。

模型各个算子的 layout 在 MegEngine 中是有默认值的，默认 layout 下的算法不一定是性能最优的算法。
为了解决 layout 带来的性能差异的问题，MegEngine 提供了 layout 转换机制用于切换不同的 layout，期望通过 layout 的切换，达到提升推理性能的目的。

Load and run 中集成了这些 layout 转换的接口，因而可以用 Load and run 来探索 layout 转换可能带来的推理性能提升。接口相关说明参考 :ref:`layout-optimize-options`

单一 layout 优化
^^^^^^^^^^^^^^^^^^^^^^

开启单一 layout 优化目前有两种方式，其一是使用 MegEngine 提供的 :ref:`dump` 函数来开启，其二是通过 Load and run 的设置选项开启。
MegEngine 提供的 :ref:`dump` 函数开启的方法可以参考 :py:meth:`~.jit.trace.dump` 。

CPU 平台
""""""""""""""""""""""

如下为 CPU 平台上可以用到 layout 优化选项

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 平台架构与模型类型
   * - ``--enable-nchw44``
     - Arm CPU float32/int8 量化模型。
   * - ``--enable-nchw88``
     - x86 CPU（支持 avx256）flloat32模型。
   * - ``--enable-nchw44-dot``
     - Arm CPU arch>=8.2 量化模型。

**ARM64 CPU 平台 layout 优化**


float32 resnet50 模型下 nchw44 和 nchw44-dot 都会在推理时有加速，但因为模型 layout 的转换，模型配置阶段需要更多时间

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - float32 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - nchw44 优化
     - 819.9
     - ``485.3``
     - ``236.5``
   * - nchw44-dot 优化
     - 803.5
     - ``483.5``
     - ``236.6``
   * - 无优化设置
     - 4.5
     - 478.6
     - 247.3

对于量化 int8 resnet50 模型而言，nchw44 会有负优化，而 nchw44-dot 会有明显加速，但同样的会在模型配置阶段耗时会有一定提升。

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - 量化int8 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - nchw44 优化
     - 439.7
     - ``332.6``
     - ``212.1``
   * - nchw44-dot 优化
     - 625.5
     - ``200.9``
     - ``69.7``
   * - 无优化设置
     - 4.3
     - 257.8
     - 82.7


**x86 CPU 平台上 layout 优化**

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - float32 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - nchw88 优化
     - 464.7
     - ``238.3``
     - ``120.1``
   * - 无优化设置
     - 1.6
     - 325.2
     - 136.9

通过调用 mkl 的卷积算子，x86 上的 float32 模型有一定加速提升。

NVIDIA GPU 平台
""""""""""""""""""""""

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 平台架构与模型类型
   * - ``--enable-nchw4``
     - GPU int8 模型。
   * - ``--enable-chwn4``
     - NVIDIA tensorcore int8模型。
   * - ``--enable-nchw32``
     - NVIDIA tensorcore int8模型。
   * - ``--enable-nchw64``
     - NVIDIA tensorcore`` `fast int4 <https://developer.nvidia.com/blog/int4-for-ai-inference/>`__ 模型。

对于不同模型 NVIDIA GPU 上相关 layout 优化性能如下所示。

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - 量化 int8 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - nchw4 优化
     - 70.3
     - ``29.7``
     - ``5.0``
   * - chwn4 优化
     - 98.6
     - ``42.6``
     - ``23.3``
   * - chwn4 优化
     - 95.7
     - ``29.8``
     - ``14.9``
   * - nchw32 优化
     - 114.2
     - ``29.7``
     - ``4.3``
   * - nchw64 优化
     - 64.6
     - ``28.1``
     - ``4.3``
   * - 无优化设置
     - 1.1
     - 103.5
     - 40.8

可以看出使用优化设置后，模型推理性能均有一定的提升。

全局 layout 优化
^^^^^^^^^^^^^^^^^^^^^^

单个 layout 优化的只有在特定平台以及特定模型下有加速，这些设定比较繁琐，
使用时对于初步接触 MegEngine 的读者而言，使用不是很方便。另外这些单个的优化知识局部最优的一个解，并没有达到整个计算图上的最优。
全局 layout 优化可以很好的解决些问题。选项说明参考 :ref:`layout-optimize-options`

基本用法如下：

.. code:: bash

   # 在线使用全局 layout 优化
   load_and_run <model_path> --input <input_name:input_data_description> --layout-transform <backend_type>

   # dump 全局 layout 优化之后的模型并运行
   load_and_run <model_path> --input <input_name:input_data_description> --layout-transform <backend_type> --layout-transform-dump <model_path_after_layout_transform>
   load_and_run <model_path_after_layout_transform>

下面是各个平台上的推理性能（模型为resnet50）

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``CUDA`` float 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 在线全局 layout 优化
     - 2517.4
     - ``21.8``
     - ``4.3``
   * - 模型离线 layout 优化
     - 1.2
     - ``27.8``
     - ``4.3``
   * - 无优化设置
     - 1.7
     - 29.8
     - 5.2

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``CUDA`` int8 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 在线全局 layout 优化
     - 1470.1
     - ``16.2``
     - ``2.1``
   * - 模型离线 layout 优化
     - 1.3
     - ``21.6``
     - ``2.5``
   * - 无优化设置
     - 1.2
     - 75.2
     - 45.7

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``x86 CPU`` float 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 在线全局 layout 优化
     - 10166.1
     - ``223.6``
     - ``112.1``
   * - 模型离线 layout 优化
     - 1.2
     - ``280.5``
     - ``114.2``
   * - 无优化设置
     - 1.6
     - 328.4
     - 130.9

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``arm CPU`` float 模型
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 在线全局 layout 优化
     - 18038.3
     - ``449.8``
     - ``235.8``
   * - 模型离线 layout 优化
     - 7.5
     - ``527.7``
     - ``235.9``
   * - 无优化设置
     - 8.9
     - 548.7
     - 248.2

由上述几个表中数据可以看出，全局 layout 优化在各类推理后端上都可以有很大的性能提升，且相比于单一 layout 只需要指定相关推理后端即可。
另外全局图优化后的模型可以 dump 离线，使用 dump 后的模型可以在减少优化过程的用时的同时，提升模型推理性能。

算子融合以及其他优化
----------------------

使用 weight 预处理优化加快卷积运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MegEngine 中针对 conv 算子的实现，有 winograd/im2col/direct 等一系列不同的实现。
对于 winograd/im2col 的实现而言，其算法在真正做卷积计算之前需要对卷积核的权重做预先的处理，推理时，模型卷积核的权重大部分情况下都是常量，
因此通过集中的权重预处理可以进一步加快整个推理的算法性能。

**使用 Load and run 进行 weight 预处理过程**

.. code:: bash

   # 获取离线缓存
   load_and_run <model_path> --input <input_name:input_data_description> --fast-run --fast-run-algo-policy <algo_cache_file> --weight-preprocess --cpu
   # 使用离线缓存
   load_and_run <model_path> --input <input_name:input_data_description> --fast-run-algo-policy <algo_cache_file> --weight-preprocess --cpu

在 arm64 CPU 设备上，对于 float32 的 resnet50 网络模型，模型运行各个阶段的用时如下所示：

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - 
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - fast-run 缓存 + weight 预处理
     - 8.4
     - ``652.4``
     - ``186.4``
   * - 只使用 fast-run 缓存
     - 11.3
     - ``571.3``
     - ``227.9``
   * - 只使用 weight 预处理
     - 8.4
     - ``581.1``
     - ``229.8``
   * - 无优化设置
     - 8.6
     - 562.4
     - 245.8

可以看出，weight 预处理在 arm64 CPU 设备上能够有效的提高推理性能，代价是预热阶段的耗时变大了。但预热在实际应用中相比于推理占比较小，所以整体性能是有很大提升的。

.. note::

   weight 预处理只针对使用 winograd/im2col 实现的卷积算法，因此只有在对应推理后端上有相应实现时才能有加速。相应实现在 MegEngine 的 `dnn <https://github.com/MegEngine/MegEngine/tree/master/dnn/src>`__ 中。 

使用 fake-first 加速预热过程
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

无论是 fast-run，还是 weight 前处理，在提升推理性能的同时，不免会引起预热阶段的一些耗时增加。
虽然预热部分在应用中占比不高，但有些应用下不免会成为瓶颈所在。 fake-first 这一优化，
可以用于预先执行内存分配，队列初始化等操作，在此过程中不会进行计算任务，从而减少预热时间，提升整体性能。
 
**使用 Load and run 加速预热过程**

.. code:: bash

   # 获取离线缓存
   load_and_run <model_path> --input <input_name:input_data_description> --fast-run --fast-run-algo-policy <algo_cache_file> --weight-preprocess --cpu
   # 使用离线缓存
   load_and_run <model_path> --input <input_name:input_data_description> --fast-run-algo-policy <algo_cache_file> --weight-preprocess --cpu --fake-first --warmup-iter 2

在 arm64 CPU 设备上，对于 float32 的 resnet50 网络模型，模型运行各个阶段的用时如下所示：

.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * -  
     - 模型配置（ms）
     - 模型预热（首次/第二次 ms） 
     - 模型推理（ms）  
   * - fast-run 缓存 + weight 预处理 + fake-fisrt
     - 7.8
     - 30.1/410.1
     - ``184.8``
   * - 只使用 fast-run 缓存 + fake-first
     - 9.7
     - 29.9/298.8
     - ``226.4``
   * - 只使用 weight 预处理 + fake-first
     - 11.2
     - 31.3/361.8
     - ``229.8``
   * - 无优化设置 + fake-first
     - 8.1
     - 36/316.7
     - 245.8
   * - 无优化设置 
     - 7.1
     - 551.6/----
     - 246.4

可以看出，使用fake-fisrt 时，能够很好的降低预热所需时间，唯一的问题时预热部分需要执行至少两次。

使用算子融合优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
对于一些常见的算子，其往往以某一固定组合的形式出现，这样的算子通常可以融合为一个算子，从而减少数据搬运，提升整体的资源利用率。
常见的融合包括 elemwise 算子与shape 或者 type 变换算子的融合，卷积与非线性算子的融合，卷积，加法以及非线性算子的融合（类 resnet 的网络中大量存在）。
Load and run 为这几类算子融合的设置提供了相应配置选项，参考 :ref:`preprocess-fuse-options`

``--enable-fuse-preprocess`` 选项用于前处理相关的，如类型转换，dimshuffle等算子的融合。对于单个模型而言性能变化不大。
这里主要介绍后两种算子融合的优化。


.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``CPU`` float模型 
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 卷积与非线性算子的融合
     - 3.3
     - 286.7
     - ``129.1``
   * - 卷积，加法以及非线性算子的融合
     - 3.4
     - 314.7
     - ``127.2``
   * - 无优化设置 
     - 1.6
     - 313.6
     - 130.3


.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``CUDA`` float模型 
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 卷积与非线性算子的融合
     - 3.7
     - 26.9
     - ``4.3``
   * - 卷积，加法以及非线性算子的融合
     - 4.1
     - 28.2
     - ``4.7``
   * - 无优化设置 
     - 2.1
     - 29.7
     - 5.0

算子融合有一定优化，但总体提升有限。

kern record 优化
------------------------------

MegEngine 算法在运行时会根据传入的运行时信息分配相应的 kern 进行计算。这一过程是运行时决定的。所以会有一定耗时。
针对这部分耗时，MegEngine 提供了 record 的机制来记录运行时分配的 kern。在后面运行时直接使用 这部分 kern 的记录来加速推理。
Load and run 选项说明参考 :ref:`preprocess-fuse-options`

record 优化通常在一些低端平台上有一定优化。主要的使用方法：

.. code:: bash

   # 只记录运行时调用的 kern
   load_and_run <model_path> --input <input_name:input_data_description> --record-comp-seq
   # 释放计算图的上的一些数据
   load_and_run <model_path> --input <input_name:input_data_description> --no-sanity-check --record-com-seq2
  
.. list-table::
   :widths: 30 25 20 18
   :header-rows: 1

   * - ``arm64 CPU`` float模型 
     - 模型配置（ms）
     - 模型预热（ms） 
     - 模型推理（ms）  
   * - 开启 record level 1
     - 7.7
     - 564.2
     - ``245.2``
   * - 开启record level 2
     - 37.1
     - 311.7
     - ``243.5``
   * - 无优化设置 
     - 6.8
     - 540.3
     - 245.5


有关常见推理测速的优化设置总结
---------------------------------------

**x86 CPU 平台推理测速**

.. code:: bash

   load_and_run <model_path> --cpu --input <input_name:input_data_description> --layout-transform cpu --layout-transform-dump <model_path_after_layout_transform> 
   load_and_run <model_path_after_layout_transform> --cpu --fast-run --fast-run-algo-policy <algo_cache_file>  --weight-preprocess
   load_and_run <model_path_after_layout_transform> --cpu --fast-run-algo-policy <algo_cache_file> --weight-preprocess --fake-first

**ARM CPU 平台推理测速**

.. code:: bash

   load_and_run <model_path> --cpu --input <input_name:input_data_description> --layout-transform cpu --layout-transform-dump <model_path_after_layout_transform>
   load_and_run <model_path_after_layout_transform> --cpu --fast-run --fast-run-algo-policy <algo_cache_file>  --weight-preprocess --enable-fuse-conv-bias-nonlinearity
   load_and_run <model_path_after_layout_transform> --cpu-default --fast-run-algo-policy <algo_cache_file> --weight-preprocess  --enable-fuse-conv-bias-nonlinearity --record-comp-seq

**NVIDIA GPU 平台推理测速**

.. code:: bash

   load_and_run <model_path> --cuda --input <input_name:input_data_description> --layout-transform cuda --layout-transform-dump <model_path_after_layout_transform>
   load_and_run <model_path_after_layout_transform> --cuda  --fast-run --fast-run-algo-policy <algo_cache_file>  --weight-preprocess 
   load_and_run <model_path_after_layout_transform> --cuda --fast-run-algo-policy <algo_cache_file> --weight-preprocess --enable-fuse-conv-bias-nonlinearity --record-comp-seq
