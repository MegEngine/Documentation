.. _lar-debug:

使用 Load and run debug 模型推理
==================================

使用 MegEnigne 进行模型推理时，可能会因为各种原因导致推理时出现 bug。
Load and run 提供了一些选项用于推理时的 bug 排查。

获取模型输入输出 tensor 的简要信息
----------------------------------

有些情况下，不正确的模型输入会导致推理出现问题。
另外，还有可能出现部分模型的输入输出信息暂时未知。
这些情况下，可以使用 Load and run 提供的 ``--model-info`` 来查看模型输入输出信息。
具体使用方法如下：

.. code:: bash

   load_and_run <model_path> --model-info 

模型的输入输出信息会以表格的形式输出到终端：

.. code:: bash

   +Runtime Model Info---------------+---------+
   |  type  |  name  |     shape     |  dtype  |
   +--------+--------+---------------+---------+
   | INPUT  |  data  | {1,3,224,224} | float32 |
   +--------+--------+---------------+---------+
   | OUTPUT | fc.ADD |   {1,1000}    | float32 |
   +--------+--------+---------------+---------+

主要包括输入输出的名称，shape 以及数据类型。

获取模型运行时 debug 信息
----------------------------------

部分模型在推理运行过程中会出现中断，报错等问题，为了更加方便的定位问题所在，
Load and run 提供了设置 Megengine 内部 log 等级的选项 ``--verbose`` ,这一选项可以展示更多的运行时信息，辅助用户 debug。

.. code:: bash

   load_and_run <model_path> --verbose

相关信息会输出到对应终端。大概信息如下：

.. code:: bash

   [14 11:23:44 @:718][DEBUG] model feature: aligned=0 has_dtype_map=1 has_opr_priority=0 has_output_dtype=0
   [14 11:23:44 @:159][DEBUG] create CompNode cpu0:0 from logical xpux:0
   load model: 5.730ms
   [14 11:23:44 @:816][DEBUG] opr seq of length 4: var_static=4 var_dynamic_shape=0 var_dynamic_storage=0 no_sys_alloc=0
   [14 11:23:44 @:121][WARN] run testcase: 0 
   === prepare: 0.109ms; going to warmup

   [14 11:23:44 @:60][DEBUG] static memory allocation:
   comp_node           alloc                      lower_bound         upper_bound
   CompNode("cpu0:0" from "xpux:0")      0.00MiB(        64bytes)      0.00MiB(100.00%)      0.00MiB( 37.50%)
   [14 11:23:44 @:73][DEBUG] static storage on CompNode("cpu0:0" from "xpux:0"): size=0.00MiB addr_range=[0x56088d820e00, 0x56088d820e40). 
   [14 11:23:44 @:385][DEBUG] static memory allocation: nr_opr=4 nr_var=4 realtime=0.10msec (plan0.08 alloc0.02)

使用时会输出存储分配，算法选择，推理优化等一些推理时的重要信息。

获取模型结构以及运行时显存相关信息
----------------------------------

直观的查看模型结构以及模型中各个算子运行时占用显存的大小，对于模型优化参考价值很大。Megengine 提供了模型以及静态显存可视化的工具，用于前述操作。

使用方法：

.. code:: bash    

   load_and_run <model_path>  --get-static-mem-info <staticMemInfoDir>
        
   # view the graph with given url (usally: http://localhost:6006/)
   mkdir <staticMemInfoDirLogs> &&  python3 imperative/python/megengine/tools/graph_info_analyze.py -i <staticMemInfoDir> -o <staticMemInfoDirLogs>
   # pip3 install tensorboard
   tensorboard --logdir <staticMemInfoDirLogs>

