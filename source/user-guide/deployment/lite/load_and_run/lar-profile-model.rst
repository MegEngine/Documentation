.. _lar-profile-model:

使用 Load and run 进行模型性能分析
==================================

``--profile`` 以及 ``--profile-host`` 可以获取推理时模型各个算子运行时间等一系列信息。
利用这一信息可以分析模型中各个算子的性能，获取影响模型推理性能的瓶颈所在，为后续的性能优化提供依据。下面就模型性能分析的一些常见场景进行介绍。

.. note::

   MegEngine 提供了 profile 相关的 python 接口， 使用方法可以参考 :ref:`profiler-guide`，这里只介绍与 Load and run 有关的一些内容的补充

获取模型运行性能相关信息
------------------------

使用 ``--profile`` 或 ``--profile-host`` （仅获取host上的运行信息）获取模型运行性能相关的 json 文件（参考 :ref:`plugin-options` ）。

.. code:: bash

   load_and_run <model_path> --input <input_name:input_data_description> <other_config_options> --proifle profile_file.json

生成的 json 相应的格式大致如下所示：

.. code::

   {
    "profiler": {
        "opr_internal_pf": { },
        "host": {
            "4": {"@": {"end": 0.006568, "kern": 0.006566, "start": 0.006564}},
            ...
        },
        "opr_footprint": {
            "4": {
                "param": { },
                "out_shapes": [ [ 3]], 
                "in_shapes": [], 
                "memory": 12, 
                "computation": null
            },
            ...
        },
        "device": {
                "4": {
                "CompNode(\"cpu0:0\" from \"xpux:0\")": 
                {
                    "end": 0.000651, 
                    "kern": 0.00065, 
                    "start": 0.000648
                }
            },
            ...
        },
        "graph_exec": {
            "comp_seq": [ "4", "0", "2", "6"], 
            "mem_chunk": {
            "chk5": 
            {
                "dev_ptr": 94106063605632, 
                "owner_var": "5", 
                "size": 12, 
                "id": "chk5", 
                "node_type": 
                "mem_chunk"
            },
            ...
            },
            "var": {
                "5": {
                    "prev_dev_ptr_end": 94106063605644, 
                    "flag": [ 
                        "NO_MEM_RECLAIM", 
                        "ALLOW_EMPTY_SHAPE", 
                        "PERSISTENT_DEVICE_VALUE", 
                        "DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC"
                    ], 
                    ...
                }
                ...
            "operator": {
                "4": {"waiting_spec": { },}
                ...
            }
        }
   }

其中包含各个算子在 host 和 device 上对应的运行时间，以及其他的一些重要信息。

使用 MegEngine 提供的分析工具进行性能分析
------------------------------------------------------------

MegEngine 提供了内置的 `megengine.tools.profile_analyze <https://github.com/MegEngine/MegEngine/blob/master/imperative/python/megengine/tools/profile_analyze.py>`__ 
用于分析上述 json 文件中包含的相关信息。该工具提供了多种角度的性能展示方式。

基本用法
^^^^^^^^^^^^^^

.. code:: bash

   # 脚本方法
   python3 <megengine_path>/imperative/python/megengine/tools/profile_analyze.py <profile_json_file>

   # 模块方法（已安装 MegEngine ）
   python3 -m megengine.tools.profile_analyze <profile_json_file>

这时会默认输出前三个耗时最高的算子的相关信息以及总的运行时间信息:

.. code:: bash

   -----------------  --------
   total device time  0.126343
   total host time    0.002728
   -----------------  --------

   ╒════════════════════╤══════════════╤════════════════════════════════╤═══════════════╤═════════╤══════════╤═════════════╤═══════════════╤════════════════╕
   │ device self time   │ cumulative   │ operator info                  │ computation   │ FLOPS   │ memory   │ bandwidth   │ in_shapes     │ out_shapes     │
   ╞════════════════════╪══════════════╪════════════════════════════════╪═══════════════╪═════════╪══════════╪═════════════╪═══════════════╪════════════════╡
   │ #0                 │ 0.00514      │ conv(h2d[48],const{64,3,7,7}[5 │ 236.03        │ 45.94   │ 3.67     │ 714.79      │ {1,3,224,224} │ {1,64,112,112} │
   │ 0.00514            │ 4.1%         │ -  0])[52]                     │ MFLO          │ GFLOPS  │ MiB      │ MiB/s       │ {64,3,7,7}    │                │
   │ 4.1%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 52                             │               │         │          │             │               │                │
   ├────────────────────┼──────────────┼────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼───────────────┼────────────────┤
   │ #1                 │ 0.00936      │ conv(FUSE_ADD_RELU[423],const{ │ 231.21        │ 54.74   │ 9.48     │ 2.19        │ {1,512,14,14} │ {1,512,7,7}    │
   │ 0.00422            │ 7.4%         │ -  512,512,3,3}[425])[427]     │ MFLO          │ GFLOPS  │ MiB      │ GiB/s       │ {512,512,3,3} │                │
   │ 3.3%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 427                            │               │         │          │             │               │                │
   ├────────────────────┼──────────────┼────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼───────────────┼────────────────┤
   │ #2                 │ 0.0136       │ conv(FUSE_ADD_RELU[484],const{ │ 231.21        │ 55.17   │ 9.19     │ 2.14        │ {1,512,7,7}   │ {1,512,7,7}    │
   │ 0.00419            │ 10.7%        │ -  512,512,3,3}[486])[488]     │ MFLO          │ GFLOPS  │ MiB      │ GiB/s       │ {512,512,3,3} │                │
   │ 3.3%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 488                            │               │         │          │             │               │                │
   ╘════════════════════╧══════════════╧════════════════════════════════╧═══════════════╧═════════╧══════════╧═════════════╧═══════════════╧════════════════╛

脚本的使用方法可以通过运行脚本的 ``--help`` 查看，常见的一些设置选项如下：

.. list-table:: 
   :widths: 40 20 20
   :header-rows: 1

   * - 选项 
     - 用途
     - 可选参数 
   * - ``-t`` 或 ``--top``
     - 设置需要展示的算子数目，默认为 ``3``
     - 数字
   * - ``--opr-name <name_regex_string>``
     - 筛选与给定名称正则表达式匹配的 opr
     - 带关键字的正则字符串
   * - ``--type <opr_type>``
     - 筛选给定类型的算子展示
     - ``”ConvolutionForward“`` ，``”PoolingForward“`` ，``”MatrixMul“`` 等
   * - ``--order-by <table_column_name>``
     - 按照表格中的某一列降序排列（前面带 ``+`` 时表示升序排列）
     - ``”computation“`` ，``“FLOPS”`` ，``“memory”`` ，``“bandwidth“`` 等（选项说明参考 ::`profile-analyze`）
   * - ``--aggregate-by type --aggregate <op_name>``
     - 根据类型进行收缩，收缩时以<op_name>中的规则为收缩标准
     - ``“max”`` ，``“min”`` ，``“sum”`` ，``“mean”``
   * - ``--top-end-key <end_desc>``
     - 设置计算时间的范围，包括 device 上全部的用时以及仅 kern 在 device 上执行的时间
     - ``“end”`` (device 上全部用时)，``“kern”`` (仅 kern 执行用时) 
   * - ``--min-time <number_of_time>``
     - 设置输出到界面上的最小用时阈值
     - 浮点数
   * - ``--max-time <number_of_time>``
     - 设置输出到界面上的最大用时阈值
     - 浮点数
   * - ``--print-only <key_word>``
     - 设置需要输出的信息类型
     - ``“summary”`` （简短的总结）， ``“device”`` （device 上的时间），``“host”`` （host 上的时间）

常见使用场景
^^^^^^^^^^^^^

**分析模型中 conv 算子的性能**

.. code:: bash

   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 3 --opr-name "conv*" 
   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 3 --type ConvolutionForward

得到如下输出：

.. code:: bash

   -----------------  --------
   total device time  0.130127
   total host time    0.002241
   -----------------  --------

   ╒════════════════════╤══════════════╤════════════════════════════════╤═══════════════╤═════════╤══════════╤═════════════╤═══════════════╤════════════════╕
   │ device self time   │ cumulative   │ operator info                  │ computation   │ FLOPS   │ memory   │ bandwidth   │ in_shapes     │ out_shapes     │
   ╞════════════════════╪══════════════╪════════════════════════════════╪═══════════════╪═════════╪══════════╪═════════════╪═══════════════╪════════════════╡
   │ #0                 │ 0.00572      │ conv(reshape[52],const{64,3,7, │ 236.03        │ 41.26   │ 3.67     │ 641.95      │ {1,3,224,224} │ {1,64,112,112} │
   │ 0.00572            │ 4.4%         │ -  7}[54])[56]                 │ MFLO          │ GFLOPS  │ MiB      │ MiB/s       │ {64,3,7,7}    │                │
   │ 4.4%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 56                             │               │         │          │             │               │                │
   ├────────────────────┼──────────────┼────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼───────────────┼────────────────┤
   │ #1                 │ 0.0101       │ conv(FUSE_ADD_RELU[461],const{ │ 231.21        │ 52.81   │ 9.19     │ 2.05        │ {1,512,7,7}   │ {1,512,7,7}    │
   │ 0.00438            │ 7.8%         │ -  512,512,3,3}[463])[465]     │ MFLO          │ GFLOPS  │ MiB      │ GiB/s       │ {512,512,3,3} │                │
   │ 3.4%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 465                            │               │         │          │             │               │                │
   ├────────────────────┼──────────────┼────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼───────────────┼────────────────┤
   │ #2                 │ 0.0144       │ conv(FUSE_ADD_RELU[488],const{ │ 231.21        │ 53.91   │ 9.19     │ 2.09        │ {1,512,7,7}   │ {1,512,7,7}    │
   │ 0.00429            │ 11.1%        │ -  512,512,3,3}[490])[492]     │ MFLO          │ GFLOPS  │ MiB      │ GiB/s       │ {512,512,3,3} │                │
   │ 3.3%               │              │ ConvolutionForward             │               │         │          │             │               │                │
   │                    │              │ 492                            │               │         │          │             │               │                │
   ╘════════════════════╧══════════════╧════════════════════════════════╧═══════════════╧═════════╧══════════╧═════════════╧═══════════════╧════════════════╛

**分析模型中各类算子的总耗时的多少**

.. code:: bash

   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 5 --aggregate-by type --aggregate sum

.. note::
 
   输出用时总和降序排列的各个算子的信息

**分析模型中内存占用最多的算子**

.. code:: bash

   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 3 --order-by memory 
   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 3 --order-by +memory

.. note::
 
   输出用时占用内存降序或升序排列的各个算子的信息

**分析用时超过给定阈值的算子**

.. code:: bash

   python3 -m megengine.tools.profile_analyze <profile_json_file> -t 3 --min-time <given_time>

.. note::
 
   输出用时超过给定阈值的各个算子的信息


