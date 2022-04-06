.. _lar-accuracy-analysis:

使用 Load and run 进行精度分析
====================================

在推理的过程中，有时候会出现：使用相同模型以及输入进行推理得到的输出，其精度抖动不符合预期的情况。
当这种情况发生时，可以使用 Load and run 进行问题定位，分析产生这一问题的原因。

下面就使用 Load and run 进行精度分析的使用方法进行介绍。

问题复现
----------------

出现上述的精度抖动时，首先需要确认问题出现的环境并对问题进行精简，建立一个可以使用 Load and run 运行的最小复现环境。

因此首先需要确定以下的一些基本信息：

* 确定出现问题的 **设备平台** ，明确推理时用到的应后端，使用时才能参考 Load and run :ref:`device <device-options>` 相关的选项进行设置。

* 确认出现问题的 **MegEngine 版本** ，选择对应版本编译相应的 Load and run.
  
  .. code:: python
       
     import megengine
     print(megengine.__version__) 

* dump 出现问题的 **模型以及输入** ，保证模型和输入能够被 Load and run 识别。

* 确定运行时有关模型的 **配置信息** ，如开启了那些 **优化选项** ，**模型的数据类型** 等。

在以上基本的信息的基础上编译获取相应平台可用的 Load and run 程序，确定需要设置的配置接口之后就可以使用 Load and run 进行问题复现了

.. note::

   * Load and run 的一些设置选项需要在编译时配置相应的宏才可以开启，编译时需要注意，可以参考各个设置选项的说明。
   * 如果无法复现问题，需要进一步排查是否遗漏了一些配置信息（部分配置信息可能有默认值），如果没有发现遗漏且无法复现，那么问题可能出在模型前后处理的过程中

问题排查
--------------------

问题复现后需要对出现问题的原因进行进一步排查，主要目的是定位精度出现问题的网络层以及相关的算子. 下面就一些常见的场景进行介绍。

计算错误
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在添加新的算子实现时，因为部分情况未覆盖等原因，导致有些模型推理过程中出现计算错误。这时需要确定出错算子所在位置。
对于这种情况需要指定运行时的算法为 naive 属性与没有指定时的输入输出进行对比。具体排查步骤如下：

**使用 naive 算法获取正确的输出结果**

.. code:: bash 

   # dump文本形式数据
   MGB_USE_MEGDNN_DBG=2 load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --io-dump io_naive.txt
   # dump 二进制数据
   MGB_USE_MEGDNN_DBG=2 load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --bin-io-dump vars_naive

.. note::

   ``MGB_USE_MEGDNN_DBG=2`` 运行时选项用于设置运行时的算法选择，这时会使用 naive 的算子实现运行推理过程。

**获取默认设置下的输出结果**

.. code:: bash 

   # dump文本形式数据
   load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --io-dump io_default.txt
   # dump 二进制数据
   load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --bin-io-dump vars_default

**比较各节点输出差异**

得到两次不同算法设置的输出信息后即可以使用 MegEngine 提供的数据比对工具进行输出结果对照的过程。

.. code:: bash

   python3 -m megengine.tools.compare_binary_iodump vars_naive vars_default > diff_var.txt

获得的文件中包含了各个错误节点的相关信息。

**确定最先出错的输出节点信息**

利用前述步骤获得的信息，其中变量节点序号最小的节点为最先出现错误的节点，其对应的算子即为出现错误的算子。
排除计算错误之后，剩下的问题便是和计算精度有关的问题。

算法本身引入的精度抖动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有些算法在实现上本身就有一定的精度抖动，这类算法的属性通常是非 ``REPRODUCIBLE`` 的, 因此首先需要排除这类算法的干扰。
具体使用方法如下：

.. code:: bash

   load_and_run <model_path> --input <input_name:input_data_description> --reproducible <other_config_options>

设置 ``--reproducible`` 选项后如果不再出现精度抖动问题，那么就可以确定是因为没有合理使用 ``REPRODUCIBLE`` 所致，模型配置信息中需要添加相应的设置

精度不符合给定预期精度
^^^^^^^^^^^^^^^^^^^^^^^

部分场景下，推理结果有一定的精度要求，有些算法可能没有达到相应的精度条件，这时就需要进行定位没有达到精度要求的算法所在位置。

MegEngine 模型的计算图在运行时是以计算序列的形式逐步执行的，使用 Load and run 的输入输出 dump 相关设置选项就可以得到计算序列中各个计算节点对应的输入输出。
利用这些输入输出就可以定位精度没有达到要求的算法所在位置。下面就这个方法进行详细介绍。

**获取模型各个计算节点处的变量节点信息**

.. code:: bash

   # dump文本形式数据
   load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --io-dump io.txt
   # dump 二进制数据
   load_and_run <model_path> --input <input_name:input_data_description>  <other_config_options> --bin-io-dump vars

.. note::
   
   * 文本形式的数据主要用于定位精度差十分大的情况，优点是有一定可见性，需要比对的吧内容较少。当然其缺点也很明显，文本形式得到的信息是简要缩略的，不适合精度差值较小的情况。
   * 二进制形式的数据包含了相关计算节点完整的输入输出信息，可以进行精确的定位。但缺点是模型计算序列间的依赖关系被打散到各个变量文件中，对于结构比较复杂的模型定位比较麻烦。
   * 推荐使用方法，使用文本形式的数据进行计算序列依赖关系的查看，然后使用二进制数据进行精确的比对进行定位

**比较各变量节点的精度误差**

使用 MegEngine 提供的脚本比较各变量节点的精度误差

.. code:: bash

   python3 -m megengine.tools.compare_binary_iodump vars_first_run vars_second_run -e 1e-7 > diff.txt

误差结果记录在 ``diff.txt`` 中。

``compare_binary_iodump.py`` 也可以直接在 python 中使用：

.. code:: python

   from megengine.tools.compare_binary_iodump  import load_tensor_binary, check 

   val0, name0 = load_tensor_binary("vars_first_run/<file_name0>")
   val1, name1 = load_tensor_binary("vars_second_run/<file_name0>")
   name = "{}: \n{}\n{}\n".format(
            i, "\n  ".join(textwrap.wrap(name0)), "\n  ".join(textwrap.wrap(name1))
        )

   check(val0, val1, name, <max_err>)

**寻找发生精度抖动的根节点**

MegEngine 提供了定位精度抖动根节点的工具 `megengine.tools.accuracy_shake_var_tree <https://github.com/MegEngine/MegEngine/blob/master/imperative/python/megengine/tools/accuracy_shake_var_tree.py>`__
利用该工具可以获取精度发生抖动的根节点对应的数据信息。

.. code:: python
     
   python3 -m megengine.tools.accuracy_shake_var_tree diff.txt

运行上述代码就可以输出发生精度抖动的根节点的 id 以及其相应的变量信息。根据节点 id 还可以进一步分析其相应的依赖节点。

.. code:: python
     
   from megengine.tools.accuracy_shake_var_tree import varNode, get_dependence, parse
    
   # 获取 root 节点
   root = parse("diff.txt")
    
   # 根据节点变号得到节点信息
   node = varNode.get_varNode("id::3")
    
   # 获取被给定节点误差影响到的节点 id
   ref_node = node.get_reference_list()

   # 获取给定节点的依赖节点
   depend_node = node.get_dependence_list()

由上述工具得到的产生精度抖动的根节点的名称中可以定位到产生精度抖动的节点所在算法名称，根据这些信息定位了问题所在，然后根据算法的实现来进一步确认产生精度的问题所在

输出精度误差过大
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

因为不同推理后端下各个算法实现有差异，所以没法保证不同推理后端精度保持一致，有时会出现输出精度误差突然过大的问题。
这种情况下各个节点的精度误差不是很一致，需要设置一个合适的阈值来过滤一些非关键的节点。

这里可以设一个较小的阈值来初步获取出现精度误差的节点，统计这些节点中误差较大的数据来作为需要筛选的备选阈值。
然后利用后面统计得到的阈值进行节点筛选，再利用根节点定位的方法定位产生精度误差的根部节点的信息。与前述步骤一致

.. note::
   * 一个节点可能有多个输入，某一节点的精度误差变化通常指输入的最大精度误差与输出精度误差的对比
   * 定位到产生精度抖动的算法所在之后，就可以利用算法名称等信息在 MegEngine 的应用代码中分析算法所在位置以及算法相应的实现细节，从而确定产生问题的根本原因。



