.. _lar-options-list:

Load and run 设置选项列表及说明
================================

.. _basic-options:

基本设置选项
-------------------

.. note::

   以下选项用于 Load and run 运行时的一些基本设置，如推理运行次数，输入设置等

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--iter <iter_number>``
     - 设置模型测速的迭代次数(默认值为 ``10`` )
   * - ``--warmup-iter <warmup_number>``
     - 模型测速前warm up的次数（默认值为 ``1`` ）
   * - ``--thread <thread_number>`` 
     - 验证当前程序是否支持thread，可以提供多个线程跑多个模型的验证（默认值为 ``1`` ）
   * - ``--input "<data_0_name:data_0>;...;<data_n_name:data_n>"``
     - 设置输入数据
     
 

.. warning::

   ``--thread`` 选项主要是为了验证 Load and run 是否支持多个线程跑多个模型，不会有计算加速。
   如果需要使用多线程来加速推理，可以参考 :ref:`device-options` 中 ``multithread`` 相关的设置

.. note::

   ``--input`` 目前支持以下几种输入格式：

   .. list-table:: 
      :widths: 10 30 20
      :header-rows: 1

      * - 格式名称 
        - 格式定义
        - 格式说明
      * - 字符串输入
        - “<data_name>:[[data00,data01,...],[data10,data11,...],...]]”
        - 这一方式只适用于简单的验证性测试时使用
      * - json文件
        - 如表后所示
        - 与 string 方式相差不多，raw 部分数据需要表示为一个字符串列表
      * - numpy 数据 
        - 使用python numpy 保存的数据
        - 比较常用
      * - ppm 以及 pgm 格式图像
        - `ppm 以及 pgm <https://en.wikipedia.org/wiki/Netpbm#File_formats>`__
        - 数据格式比较原始，相当于 json 格式的二进制版

   如下所示为 json 格式数据的一个简单示例：

   .. code:: json

      {"data_f32":
           
          "shape": [1,3],
          "type": "int32",
          "raw": [3,4,5]
          }
      }

   .. warning::

      json数据输入时需要保证与上述示例格式一致，输入数据名称要与模型中的输入相同

   numpy 数据生成：

   .. code:: python
    
      import numpy as np 
      data=np.random.rand(1,3,224,224)
      np.save("input_data_uint8.npy",data.astype(np.uint8))

使用示例
^^^^^^^^^^^^^^^^^

.. code:: bash

   <<path_of_load_and_run>>/load_and_run <model_path> --iter 20

   <<path_of_load_and_run>>/load_and_run <model_path> --warmup-iter 10

   <<path_of_load_and_run>>/load_and_run <model_path> --thread 2

   <<path_of_load_and_run>>/load_and_run <model_path> --input "<data_name>:<data_array_in_string>"

   <<path_of_load_and_run>>/load_and_run <model_path> --input "<json_data_file>"

   <<path_of_load_and_run>>/load_and_run <model_path> --input "<data_name>:<numpy_data_file.npy>"

   <<path_of_load_and_run>>/load_and_run <model_path> --input "<data_name>:<ppm_data_file.ppm>"


.. _fast-run-options:

fast-run 相关设置
--------------------

fast-run 的设置主要用于在存在多种算法实现的算子中选出其中在当前情况下性能最好的算法。

.. note::

   * 使用 fast-run 相关配置前需要保证 fast-run 部分代码的可用性， MegEngine 使用宏 ``MGB_ENABLE_FASTRUN`` 来控制。编译时加上选项 ``-DMGB_ENABLE_FASTRUN=1`` 即可。
   * MegEngine 默认会开启 ``MGB_ENABLE_FASTRUN``。

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--full-run``
     - profile 各算子所有的算法（包括 naive 的算法），选择其中性能最好的算法进行推理。对应的 Strategy 为：``PROFILE``
   * - ``--fast-run``
     - profile 优化的算法，选择其中性能最好的算法进行推理。对应的 Strategy 为：``PROFILE+OPTIMIZED``
   * - ``--fast-run-algo-policy <cache_file>``
     - 设置给定文件中的缓存算法作为推理时用到的算法，或者将推理时选择到的算法缓存到给定文件中。
   * - ``--reproducible``
     - 在可复现的算法集合中选择算法，用到的算法可以保证前后两次推理结果的一致性。对应的 Strategy 为：``REPRODUCIBLE``
   * - ``--fast-run-shared-batch-size <size>``
     - 使用统一给定的 batch size 来选择相应算法，忽略模型的 batch size 变化， 该选项设置算法 negativate 属性为：``USABLE_DEPEND_ON_SHAPE``
   * - ``--binary-equal-between-batch``
     - 在精度对 batch 不敏感的算法中选择。 该选项设置算法 negativate 属性为：``ACCURACY_DEPEND_ON_BATCH``，同时会设置 Strategy 为：``REPRODUCIBLE``
     
.. note::

   * Megengine 算法选择默认的 Strategy 为：``HEURISTIC``  
   * 所谓精度对 batch 敏感，是指在多 batch 的情况下，即使各 batch 的输入内容完全一致，其对应的输出也完全不同。

.. warning::

   有些特殊的算子可能没有除 naive 以外的算法，此时运行终止，报相关错误信息

使用示例
^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 在全部的算法中选择性能最优的算法存储到给定文件中
   <<path_of_load_and_run>>/load_and_run <model_path> --full-run --fast-run-algo-policy <algo_cache_file>
  
   # 在优化的算法选择性能最优的算法到给定文件中
   <<path_of_load_and_run>>/load_and_run <model_path> --fast-run --fast-run-algo-policy <algo_cache_file>

   # 加载之前缓存的算法进行推理
   <<path_of_load_and_run>>/load_and_run <model_path> --fast-run-algo-policy <algo_cache_file>

   # 忽略模型 batch 变化，使用给定的 batch size 搜索算法
   <<path_of_load_and_run>>/load_and_run --fast-run | --full-run | --fast-run-algo-policy <algo_cache_file> --fast-run-shared-batch-size <size>


.. _IO-options:

IO相关设置
--------------------

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--input "data_name:data_file|data_string;...;..."``
     - 输入用户自定义数据，支持的数据格式：json 文件，ppm pgm 图像，npy 数据，自定义数据字符串， 参考 :ref:`basic-options` 。
   * - ``--io-dump``
     - 以文本的形式 dump 计算图中算子的输入输出
   * - ``--io-dump-stdout|--io-dump-stderr``
     - 与 ``--io-dump`` 功能相同，只是将相关文本输出到标准输出或标准错误中。
   * - ``--bin-io-dump <dir_name>``
     - 以二进制的形式 dump算子的 IO 信息，输出二进制文件到 dir_name 的文件夹中,文件名称为各个算子的输出 tensor 的内部 id
   * - ``--bin-out-dump <dir_name>``
     - 以二进制的形式 dump算子的输出信息，输出与 ``--bin-io-dump`` 类似。
   * - ``--copy-to-host``
     - 将 device 上的输出 copy 到 host 上，默认情况下不会进行输出 d2h 的 copy 操作。该设置选项用来设置输出 tensor 从 device 到 host 的拷贝，用于测速实际应用中真正用到的运行时间。
     
.. note::

   文本形式输出信息如下所示：

   .. code:: bash

       var123 produced: name=interstellar2c_branch2a layout={1(200704),64(3136),56(56),56(1) Float32} owner_opr=ADD(conv[117],dimshuffle[120])[122]{Elemwise} opr122
       deps:
       [i0]var116: [263.2, 241.2, 238.7, 236.5, 241.9, ...] s
       [i1]var121: [0, 0, 0, 0, 0, ...] s
       val: [263.2, 241.2, 238.7, 236.5, 241.9, ...]min=-618 max=513 mean=4.79 l2=109 sd=109 s

   主要包括变量 tensorid 变量 tensor 节点所在 opr,变量依赖节点 tesnor id，以及变量 tensor 相关值等


   二进制输出文件格式定义（参考 `dump_tensor <https://github.com/MegEngine/MegEngine/blob/master/src/core/impl/utils/debug.cpp#L447>`__）为：

   .. code:: 

      struct Header {
          uint32_t name_len;
          uint32_t dtype;
          uint32_t max_ndim;
          uint32_t shape[TensorShape::MAX_NDIM];
          char name[0];
      } header;
      char tensor_name[name_len];// name 中包涵了算子以及关联的输入 tensor ID
      char tesnor_raw_value[value_len];

    
   参照该格式可以解析算子的相关信息。具体解析的实现细节可以参考 MegEngine 提供的 binary io 比较工具 `megengine.tools.compare_binary_iodump <https://github.com/MegEngine/MegEngine/blob/master/imperative/python/megengine/tools/compare_binary_iodump.py>`__

   ``--bin-out-dump`` 输出文件的名称格式定义：

   .. code:: 

      std::string file_name = ssprintf(“run%zu-var%zd”, iteration_ID, var_ID);


   细节参考 `outdumper <https://github.com/MegEngine/MegEngine/blob/master/Lite/load_and_run/src/helpers/outdumper.cpp>`__


    

使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # dump text
   <path_of_load_and_run>/load_and_run <model_path> --input <data_description> --cpu --io-dump cpu.txt
   <path_of_load_and_run>/load_and_run <model_path> --input <data_description> --cuda --io-dump cuda.txt

   # dump binary 

   mkdir cpu && <path_of_load_and_run>/load_and_run <model_path> --input <data_description> --cpu --bin-io-dump cpu
   mkdir cpu && <path_of_load_and_run>/load_and_run <model_path> --input <data_description> --cuda --bin-io-dump cuda

   # compare text
   diff cpu.txt cuda.txt

   # compare binary 
   <megengine_path>/tools/compare_binary_iodump.py cpu cuda

.. note:: 

   text 形式只是显示了部分信息，比如 Tensor 的前几个输出结果，整个 Tensor 的平均值、标准差之类，
   如果需要具体到哪个值错误，通常用 binary 的方式进行验证



.. _layout-optimize-options:

layout 优化相关设置
------------------------

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--enable-nchw4``
     - 使用 ``{N, C/4, H, W, 4}`` layout 格式的优化，``GPU int8`` 模型有加速。
   * - ``--enable-chwn4``
     - 使用 ``{C/4, H, W, N, 4}`` 的 layout 格式的优化，``NVIDIA tensorcore int8`` 模型有加速。（该格式为 Megengine 定义）
   * - ``--enable-nchw44``
     - 使用 ``{N/4, C/4, H, W, 4, 4}`` 的 layout 格式的优化，``Arm CPU float32`` 模型有加速。
   * - ``--enable-nchw88``
     - 使用 ``{N/8, C/8, H, W, 8, 8}`` 的 layout 格式的优化，``x86 CPU（支持 avx256）flloat32`` 模型有加速。
   * - ``--enable-nchw32``
     - 使用 ``{N, C/32, H, W, 32}`` 的 layout 格式的优化，``NVIDIA tensorcore int8`` 模型有加速。
   * - ``--enable-nchw64``
     - 使用 ``{N, C/64, H, W, 64}`` 的 layout 格式的优化，``NVIDIA tensorcore`` `fast int4 <https://developer.nvidia.com/blog/int4-for-ai-inference/>`__ 模型有加速。
   * - ``--enable-nhwcd4``
     - 使用 ``{N, H, W, (C+3)/4, 4} `` 的 layout 格式的优化，移动平台 ``GPU float16`` 模型有加速。
   * - ``--enable-nchw44-dot``
     - 使用 ``{N/4, C/4, H, W, 4, 4}`` 的 layout 格式的优化，``Arm CPU arch>=8.2`` 量化模型有加速。


各种 layout 的细节可以参考 `layout_manager <https://github.com/MegEngine/MegEngine/blob/master/src/gopt/include/megbrain/gopt/reformat_manager.h>`__ 。


.. note::

   * 对于 ``--enable-nchw32`` 使用时需要开启 ``--enable-fuse-conv-bias-nonlinearity``, 可以选择性开启 ``--enable-fuse-conv-bias-with-z`` 
   * 选项可以在 dump 时开启，参考 :ref:`dump` 的推理优化设置选项。
   * 使用 ``--enable-nchw44-dot`` 编译选项需要加上 ``-march=armv8.2-a+fp16+dotprod``， Megengine 提供的编译脚本会自动进行环境检测开启这一选项 



全局 layout 优化
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    上述单一的 layout 转换实现简单，只能在 **固定平台以及特定算子** 上有明显的加速，另外 layout 转换的开销使得 **局部最优的 layout 转换不一定是全局上最优的**。
    
    基于上述两个问题, MegEngine 引入了全局 layout 优化的机制，该机制通过统一的 layout 管理，针对不同后端 **profile 不同的 layout 转换性能，全局规划，自动选择最合适的 layout 转换** ，得到全局最优的 layout 转换路径，从而实现推理加速。
    
    全局 layout 优化可以直接将计算图中的 layout 优化融合到计算密集的算子中，并将其中冗余的 layout 转换消除，可以直接得到优化后的模型计算图，从而可以直接获取优化后的模型，减少了部署时额外的优化设置。 
    

Load and run 为全局 layout 优化提供了如下两个设置接口

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--layout-transform <backend_type>`` 
     - 使用给定 backend 的全局 layout 优化 pass ,支持的 backend 包括： ``cpu`` ，``cuda`` 等。该选项用来设置全局 layout 优化的后端类型，并启用这一优化选项。
   * - ``--layout-transform-dump <model_path_after_layout_transform>``
     - 将进行全局 layout 优化之后的模型重新进行 dump，得到layout优化之后的模型。

.. note::  

   ``--layout-transform-dump`` 选项使用时需要与全局 layout 优化的设置 ``--layout-transform`` 同时使用。



使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # GPU 

   <path_of_load_and_run>/load_and_run <int8_model_path>  --cuda  --enable-nchw4
        
   <path_of_load_and_run>/load_and_run <int8_model_path> --cuda  --enable-chwn4
        
   <path_of_load_and_run>/load_and_run <int8_model_path> --cuda  --enable-nchw32
        
   <path_of_load_and_run>/load_and_run <int4_model_path> --cuda  --enable-nchw64

   # x86 CPU
   <path_of_load_and_run>/load_and_run <float32_model_path> --cpu --enable-nchw88

   # ARM CPU
        
   <path_of_load_and_run>/load_and_run <model_path> --cpu --enable-nchw44
        
   <path_of_load_and_run>/load_and_run <model_path> --cpu --enable-nchw44-dot

   # 全局 layout 优化 
   <path_of_load_and_run>/load_and_run <model_path>  --layout-transform <backend_type>
        
   <path_of_load_and_run>/load_and_run <model_path> --layout-transform <backend_type> --layout-transform-dump <model_path_with_transform>
   <path_of_load_and_run>/load_and_run <model_path_with_transform>


.. _preprocess-fuse-options:

算子融合以及其他优化
-------------------------

这些优化选项主要包括前处理以及可融合的算子优化，预热优化，存储优化以及计算 kern record 优化，通过这些设置期望减少推理的运行时间.

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--enable-fuse-preprocess``
     - 允许前处理融合，如融合 astype + pad_channel + dimshuffle 等算子。实现细节参考 `fuse_nchw4_int8_preprocess <https://github.com/MegEngine/MegEngine/blob/master/src/gopt/impl/fuse_nchw4_int8_preprocess.cpp>`__
   * - ``--weight-preprocess``
     - 允许 weight 前处理，此时，会返回执行前的 kern,用于前处理，因此可能会占用较多内存（常用于 winograd/im2col 等 conv 算法的 fast-run 中）
   * - ``--enable-fuse-conv-bias-nonlinearity``
     - 允许convolution, bias add, relu oprs 的 fuse，三者融合成一个 ConvBiasForward opr.
   * - ``--enable-fuse-conv-bias-with-z``
     - 允许 ConvBias, z(binary elemwise) oprs 的 fuse，将二者融合为 ConvBiasForward op
   * - ``--const-shape``
     - 将所有 SharedDeviceTensor 和 Host2DeviceCopy的tensor shape 设置为不可变的。
   * - ``--fake-first``
     - 允许下次执行时，仅执行非计算任务，如内存分配，队列初始化等。常用来减少预热时间，且在执行完后会置为 false
   * - ``--no-sanity-check``
     - 不在首次执行时进行变量合理性检查，此时需要用户保证其变量合理。
   * - ``--disable-mem-opt``
     - 不允许计算序列的内存优化，主要禁止静态内存的再使用以及内存规划。用于测试在原生的内存分配策略下的推理性能
   * - ``--workspace-limit <size>``
     - 设置 workspace 的上限，设备存储有限是，需要限制workspace上限来保证推理正确进行
   * - ``--record-comp-seq | --record-comp-seq2``
     - 第一次执行的时候, 记录整个计算过程中会调用的 kern，在移动端 GPU 上有很大提升。
   * - ``--enbale-jit``
     - JIT 开关，打开时可以在允许运行时的计算图编译，设置　JIT level 为1，即仅对 elemwise 类的算子起作用，level 为 2 时，会进一步包含 reduce 的 opr

.. note::

   record 设置有两级，``--record-comp-seq`` 为常用，设置开启时会记录整个计算过程中会调用的 kern。

   ``--record-comp-seq2`` 除了记录计算 kern 之外，会析构掉存储在graph上的一些信息。起到节省内存的作用

   .. warning::

      两个使用时都有一定的限制，如下：

      level1限制条件：

      1. 所有变量静态分配内存且 tensor shape 必须保持不变，执行时被 record 的 kern 才不会失效 。 

      2. 数据同步只会在运行结束后进行，否则同步以及同步结果无法保证正确性（同步过程中可能存在无法记录的非计算逻辑）。

      3. 计算图中只有一种计算设备的抽象在执行，也就意味，record 只在单一的固定设备上生效。

      level2限制条件：

      1. 预热 fake_next_exec 以及变量合理性检查 var_sanity_check_first_run 需要关掉

      2. 计算图编译之前，变量 shape 需要设置合适，如 ``--const-shape`` 设置在可变的 Tensor 上时会导致record 失败

      参考 :ref:`record_optimize`

    

使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # 算子融合优化
   <path_of_load_and_run>/load_and_run <model_path>  --enable-fuse-preprocess
    
   <path_of_load_and_run>/load_and_run <model_path>  --enable-fuse-conv-bias-nonlinearity
    
   <path_of_load_and_run>/load_and_run <model_path>  --enable-fuse-conv-bias-with-z
        
   # record computing sequence 优化
   <path_of_load_and_run>/load_and_run <model_path>  --const --record-comp-seq
    
   <path_of_load_and_run>/load_and_run <model_path>  --no-sanity-check --record-comp-seq2
        
   # 存储优化限制
   <path_of_load_and_run>/load_and_run <model_path>  --disable_mem_opt
        
   <path_of_load_and_run>/load_and_run <model_path>  --workspace_limit 10000
        
   # 预热优化
   <path_of_load_and_run>/load_and_run <model_path>  --fake-first

   # 使用 JIT
   <path_of_load_and_run>/load_and_run <model_path>   --enable_jit 

   # record 1
   <path_of_load_and_run>/load_and_run <model_path> --const-shape --record-comp-seq
    
   # record 2, 默认情况下 fake_next_exec 不开启
   <path_of_load_and_run>/load_and_run <model_path> --no-sanity-check --record-comp-seq2

.. _device-options:

设备相关设置选项
-----------------------

.. note::

   Load and run 可以指定推理用到的后端设备，设备被抽象为 CompNode, 通过制定 CompNode 的映射信息来指定对应推理后端。

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--cuda``
     - 设置 CompNode 为 cuda 上的 CompNode
   * - ``--cpu``
     - 设置 CompNode 为 cpu 上的 CompNode
   * - ``--cpu-default``
     - 将所有任务分派到 caller 线程上, 对于低端 CPU 设备，能够减少同步所需时间提高推理性能
   * - ``--multithread <thread_number>``
     - 设置 CompNode 为 multithread 上的 CompNode，多线程推理加速
   * - ``--multithread-default <thread_number>``
     - 将任务分派到线程池的各线程上，caller 线程为主线程。
   * - ``--multithread <thread_number> --multi-thread-core-ids <id0,id1,...>``
     - 设置 multithread 绑核，对应 cpu id 由 id0,id1等给出。（常用于 ARM 设备上的绑核操作，验证不同核上的推理性能）
   * - ``--rocm``
     - 设置为 ROCm 平台上执行，暂时只支持非 MegEngine Lite 的模型，设备主要支持 AMD GPU（支持 ROCm），编译时开启 编译选项：``-DMGE_WITH_ROCM=1``
   * - ``--rocm-enable-miopen-search``
     - 使用 `MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`__ 相关算法自动 tuning
   * - ``--tensorrt```
     - 使用 tesorRT作为后端进行推理
   * - ``--tensorrt-cache <cache_path>``
     - 使用 tensorRT engine 来预生成 ICudaEngine，缓存到给定文件中

.. note::

   tensorRT 编译时需开启 ``-DMGB_ENABLE_TENSOR_RT=1`` , MegEngine 的脚本是默认开启该选项的
  
使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # XPU 设备
   <path_of_load_and_run>/load_and_run <model_path> --cuda 

   <path_of_load_and_run>/load_and_run <model_path> --cpu

   <path_of_load_and_run>/load_and_run <model_path> --cpu-default

   <path_of_load_and_run>/load_and_run <model_path> --multithread

   <path_of_load_and_run>/load_and_run <model_path> --multithread-default

   <path_of_load_and_run>/load_and_run <model_path> --multithread <thread_num> --multi-thread-core-ids id_0,id_1,...,id_n

   # ROCm设备
   <path_of_load_and_run>/load_and_run <model_path> --rocm 

   <path_of_load_and_run>/load_and_run <model_path> --rocm --rocm-enable-miopen-search

   # TensorRT
   <path_of_load_and_run>/load_and_run <model_path> --tensorrt
        
   <path_of_load_and_run>/load_and_run <model_path> --tensorrt --tensorrt-cache tmpdir/TRT_cache
        
   <path_of_load_and_run>/load_and_run <model_path> --tensorrt-cache tmpdir/TRT_cache



.. _plugin-options:

插件相关设置选项
-----------------------

.. note::

   这些选项主要用于对 MegEngine 中的 `plugin <https://github.com/MegEngine/MegEngine/tree/master/src/plugin/include/megbrain/plugin>`__ 进行设置

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--check-dispatch``
     - 检查 cpu dispatch 情况,当算子在 cpu 上没有调用 dispatch 时，会输出警告到标准输出上
   * - ``--range <range_number>``
     - 检查计算图中所有数字的绝对值是否在给定范围内。超出范围会抛出异常信息
   * - ``--check-var-value switch_interval:init_var_idx``
     - 检查计算图中计算序列的第 init_var_idx++ 个变量节点，在执行 switch_interval 次后变量的值。需要进行多次迭代才能使用。
   * - ``--profile <profile_cache>``
     - 记录计算图中各算子的运行信息，将其以 json 文件的格式保存。编译时开启编译选项： ``-DMGB_ENABLE_JSON=1`` ，Megengine 提供的脚本默认开启。json 数据的分析参考 :ref:`lar-profile-model`
   * - ``--profile-host <porfile_cache>`` 
     - 只记录在 host 上运行的算子信息，以快速得到相关性能情况。（device 上的 profile 可能十分缓慢）

使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # kernel dispatch检查
   <path_of_load_and_run>/load_and_run <model_path> --check-dispatch

   # varnode变量固定迭代次数检查
   <path_of_load_and_run>/load_and_run <model_path> --check-var-value <switch_interval:start_idx>

   # varnode变量范围检查
   <path_of_load_and_run>/load_and_run <model_path> --range <abs_number_of_range>

   # 性能分析
   <path_of_load_and_run>/load_and_run <model_path> --profile <profile_json_file>

   <path_of_load_and_run>/load_and_run <model_path> --profile-host <profile_host_json_file>

.. _debug-options:

debug用到的一些设置选项
----------------------------

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--mode-info``
     - 以表格形式展示模型的输入输出信息。
   * - ``--verbose``
     - 设置 MegEngine 以及 MegEngine Lite 的 log 级别为 debug 级别，用于展示更多运行时信息（ debug，warning 以及 error )
   * - ``--disable-assert-throw``
     - 不在计算图执行时进行 assert 操作，常用于性能调优（前提是运行结果默认正确）。
   * - ``--get-static-mem-info <dir_name>``
     - 获取计算图以及运行显存信息的 json 文件用于显存和性能可视化，参考 :ref:`lar-debug` 。编译时开启编译选项：``-DMGB_ENABLE_JSON=1``
   * - ``--wait-gdb`` 
     - 输出当前进程 PID 给 gdb 工具 attach 用。


使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # debug
   <path_of_load_and_run>/load_and_run <model_path>  --model-info

   <path_of_load_and_run>/load_and_run <model_path>  --verbose

   <path_of_load_and_run>/load_and_run <model_path> --disable-assert-throw 

   <path_of_load_and_run>/load_and_run <model_path>  --get-static-mem-info <staticMemInfoDir>
        
   # view the graph with given url (usally: http://localhost:6006/)
   mkdir <staticMemInfoDirLogs> &&  python3 imperative/python/megengine/tools/graph_info_analyze.py -i <staticMemInfoDir> -o <staticMemInfoDirLogs>
   pip3 install tensorboard && tensorboard --logdir <staticMemInfoDirLogs>

.. _external-C-opr-options:

外部定义的Copr加载选项
----------------------------

选项列表
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: 
   :widths: 30 25
   :header-rows: 1

   * - 设置选项 
     - 功能
   * - ``--c-opr-lib <dynamic_lib_path>``
     - 将第三方的算子库封装为 MegEngine 可以识别接口，传入 MegEngine 进行调用。
   * - ``--c-opr-lib-with-param``
     - 使用外部的参数来运行相关的 opr,主要是包括算子执行时需要的输入输出信息以及算子执行设备的信息。

.. note::

   外部算子库要进行封装时，需要提供四个主要的 C API 供 MegEngine 接入：

   - 库入口：``MGB_C_OPR_INIT_FUNC``
   - 内存分配：``copr_param_device_ptr_malloc``
   - 内存在 host 与 device 上的迁移：``copr_param_device_ptr_h2d``
   - 内存释放：``copr_param_device_ptr_free``

   其中后三个 API 为可选实现，库入口 API 必须实现

使用示例
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   <<path_of_load_and_run>>/load_and_run <model_path> --c-opr-lib <dynamic_loader_lib_path>


