.. _graphsurgeon:

==============
图手术操作指南
==============

.. note::

   图手术相关 API 在 :py:mod:`megengine.utils.network` 中。
   需要先通过 :ref:`trace <trace>` 和 :ref:`dump <dump>` 得到导出后的 MegEngine 模型，
   如果需要对保存的模型进行修改（保存 opr/var 的名字），建议 :py:meth:`~.trace.dump` 时加上选项：

   .. code-block:: python

      optimize_for_inference=False, keep_var_name=2,  keep_opr_name=True

使用 Network 加载/保存模型
--------------------------

相关的定义和实现在 :py:class:`~.Network` 类，其中：

* 模型的加载使用 :py:meth:`.Network.load` 接口，参数 ``model_path`` 指模型路径，``outspec`` 指加载 varnode 依赖的子图
* 模型的保存使用 :py:meth:`.Network.dump` 接口，与 :py:class:`~.trace` 的 :py:meth:`~.trace.dump`
  接口基本一致，可以指定 ``optimize_for_inference`` 选项， 会 dump 模型输出节点（ ``Network.output_vars`` ）依赖的计算图

假设 ``xxx.mge`` 的实现如下：

.. code-block::

   x = a * b
   y = a + b


则可以参考下面的例子进行加载和保存：

.. code-block:: python

   from megengine.utils.network import Network as Net

   # 模型加载
   net = Net.load("/path/to/xxx.mge") # 未指定 outspec，加载完整的模型，x = a * b, y= a + b
   net1 = Net.load("/path/to/xxx.mge", outspec=["x"]) # 指定 outspec，加载 x 依赖的子图，即 x = a * b

   # 模型修改，在本页后边部分将提到几类修改模型的方法
   ...

   # 模型保存
   net.dump("/path/to/new_xxx.mge") # 保存修改后的.mge 模型
   net.dump("/path/to/new_xxx.mge", enable_io16xc32=True) # 保存修改后的 .mge 模型，开启了 enable_io16xc32 优化

使用 Nodefilter 查找 Node
-------------------------

通过调用 :py:class:`~.NodeFilter` 类中的实现，可以按一定条件对计算图中的 VarNode/OpNode 进行查找。

NodeFilter 提供了按类型、名字，是否以某个 VarNode 作为输入等条件进行查找的功能：

* 按类型：:py:meth:`.NodeFilter.type` ``(cls)`` , 查找类型是 ``cls`` 的 Node
* 按名字：:py:meth:`.NodeFilter.name` ``(name)`` , 查找名字为 ``name`` 的 Node, 可以是 UNIX 正则表达式（如 ``*bias`` ）
* 按输入：:py:meth:`.NodeFilter.has_input` ``(var)`` , 查找以 ``var`` 为输入的 Node

NodeFilter 还支持将返回结果变成列表/字典/单个 Node 类型（默认返回 NodeFilter），方便后续操作：

* :py:meth:`.NodeFilter.as_list` : 返回 Node 列表
* :py:meth:`.NodeFilter.as_dict` : 返回 Node 名字和 Node 组成的字典
* :py:meth:`.NodeFilter.as_unique` : 如果查找到的 Node 只有一个，直接返回该 Node, 否则报错
* :py:meth:`.NodeFilter.as_count` : 返回查找到 Node 的数量

此外，在 Network 中提供了一些 opr/var 的 Nodefilter 用于在计算图中查找节点：

* :py:meth:`.Network.opr_filter` : 所有 opr 的 Nodefilter
* :py:meth:`.Network.var_filter` : 所有 var 的 Nodefilter
* :py:meth:`.Network.params_filter` : 所有的 ImmutalbeTensor opr (parameter provider) 的 Nodefilter, ImmutalbeTensor 为模型提供参数。 
* :py:meth:`.Network.data_providers_filter` : 所有 Host2DeviceCopy opr (data provider) 的 Nodefilter, Host2DeviceCopy 为模型提供输入。

对应地，Network 中提供了常见的按条件查询的 API，
其中 ``unique=True`` 表示按若该条件只能查到一个 Node，返回该 Node. 而当 ``unique=False`` 时，返回一个 Node 列表：

* 按类型查找 OpNode: :py:meth:`.Network.get_opr_by_type` ``(oprcls, unique=True)`` 

  等价于 :py:meth:`.Network.opr_filter`. :py:meth:`~.NodeFilter.type` ``(oprcls).``:py:meth:`~.NodeFilter.as_unique`/:py:meth:`~.NodeFilter.as_list`

* 按名字查找 OpNode: :py:meth:`.Network.get_opr_by_name` ``(name, unique=True)`` 

  等价于 :py:meth:`.Network.opr_filter`. :py:meth:`~.NodeFilter.name` ``(name).``:py:meth:`~.NodeFilter.as_unique`/:py:meth:`~.NodeFilter.as_list`

* 按名字查找 VarNode: :py:meth:`.Network.get_var_by_name` ``(name, unique=True)`` 

  等价于 :py:meth:`.Network.var_filter`. :py:meth:`~.NodeFilter.name` ``(name).``:py:meth:`~.NodeFilter.as_unique`/:py:meth:`~.NodeFilter.as_list` 

* :py:meth:`.Network.get_var_receive_oprs` ``(var)`` , 获取以 var 为输入的 OpNode
* :py:meth:`.Network.get_dep_oprs` ``(var)`` , 获取 var 依赖的所有 opr（即得到 var 计算结果所需的所有 opr）

一些使用示例
~~~~~~~~~~~~

.. code-block:: python

   from megengine.utils.network import Network as Net

   net = Net.load("/path/to/xxx.mge")

   # 使用 var_filter 查 找varnode
   arg_0 = net.var_filter.name("arg_0").as_unique() # 返回名字是 "arg_0" 的 varnode
   args = net.var_filter.name("arg_*").as_list()    # 返回名字是 "arg_" 开头的 varnode 的列表

   # 使用 opr_filter 查找 opnode
   # 按类型查找opnode查找
   from megengine.utils.network_node import ConvolutionForward

   conv = net.opr_filter.type(ConvolutionForward).as_list()         # 返回所有的卷积 Op
   not_conv = net.opr_filter.not_type(ConvolutionForward).as_list() # 返回所有的非卷积 Op
   
   # 按输入是否包含某个 varnode 查找
   has_input_a = net.opr_filter.has_input(vara).as_list()           # 返回所有以 vara 作为输入的 Op

   # 使用 params_filter 查找网络的 parameter
   all_bias = net.params_filter.name("*bias") # 查找以 bias 结尾的 parameter
   data = all_bias[0].numpy()                 # 可以通 过numpy() 读取 parameter 的值
   all_bias[0].set_value(data/2)              # 可以通过 set_value() 修改 parameter 的值

   # 使用 data_providers_filter 查找网络的输入
   input0 = net.data_providers_filter.name("arg_0") # 获取名字是 "arg_0" 的输入，可以通过 shape, dtype 等查看/修改 input node 属性。

修改模型的方法
--------------

替换节点
~~~~~~~~

我们可以通过替换 OpNode/VarNode 来修改图结构：

* :py:meth:`.Network.replace_vars` ``(repl_dict)`` / :py:meth:`.Network.replace_oprs` ``(repl_dict)`` ,
  其中 ``repl_dict`` 存储由 被替换节点、新节点 组成的字典，该方法将图中的旧节点替换为新节点。
* :py:meth:`.Network.add_dep_oprs` ``(*vars)`` , 把 ``vars`` 依赖的 mgb OperatorNode/VarNode 转换成 Network 的 OpNode/VarNode
  添加到图中，返回值是 ``var`` 对应的 Network VarNode.

下面的例子通过 ``replace_vars/replace_oprs`` 将 ``(a+b)*2`` 修改为 ``relu(a*b)*2`` ：

.. code-block:: python

   import megengine.functional as F
   from megengine.utils.network import Network as Net

   graph = Net.load("/path/to/xxx.mge")

   vara = graph.var_filter.name("a").as_unique() # 找到 vara
   varb = graph.var_filter.name("b").as_unique() # 找到 varb

   # 使用 megengine functional api 构造 relu(a*b) 计算图
   out = F.mul(vara.var, varb.var)
   out = F.relu(out)

   # 将out 及其依赖的 opnode 添加到 graph 中，返回值是添加到 graph 中的 out 对应的 VarNode
   var_list = graph.add_dep_oprs(out)

   # 找到需要被替换的 opnode，即a+b的输出
   ori_opnode = graph.opr_filter.has_input(vara).as_unique()

   # 通过替换 vars，修改图结构
   repl_dict = {ori_opnode.outputs[0]: var_list[0]}
   graph.replace_vars(repl_dict)

   # 通过替换 oprs，修改图结构
   repl_dict = {ori_opnode: var_list[0].owner}
   graph.replace_oprs(repl_dict)

添加新的参数/输入
~~~~~~~~~~~~~~~~~

通常分为以下两种情况：

* 通过 :py:meth:`.Network.make_const` 产生一个 ImmutableTensor Opr， 返回该 opr 的输出 varnode 作为 parameter
* 通过 :py:meth:`.Network.make_input_node` 产生一个 Host2DeviceCopy Opr,  返回该 opr 输出 varnode 作为 input

比如将 ``(a+b)*2`` 修改为 ``(a+3)*2`` :

.. code-block:: python

   import megengine.functional as F
   from megengine.utils.network import Network as Net

   graph = Net.load("/path/to/xxx.mge")

   const_b = graph.make_const(3, name="b")
   varb = graph.var_filter.name("b").as_unique()
   repl_dict = {varb: const_b}
   graph.replace_vars(repl_dict)

或者将 ``(a+b)*2`` 修改为 ``(a+b)*c`` :

.. code-block:: python

   inp_c = graph.make_input_node((1,), np.int32, name="c")
   const_c = graph.params_filter.as_unique().outputs[0]
   repl_dict = {const_c: inp_c}
   graph.replaces_vars(repl_dict)


添加/删除输出节点
~~~~~~~~~~~~~~~~~

由于网络在 :py:meth:`~.trace.dump` 时只会将 ``output_vars`` 列表中的 varnode 所依赖的 opnode/varnode 保存到 .mge 文件中，
因此需要提供一些方法对 ``output_vars`` 列表进行修改：

* :py:meth:`.Network.add_output` ``(*vars)``  : 将某些 varnode 添加到网络输出节点列表 ``output_vars`` 中
* :py:meth:`.Network.remove_output` ``(*vars)`` : 将某些 varnode 从网络输出节点列表 ``output_vars`` 中移除

例如将 ``(a+b)*2`` 修改为 ``relu((a+b)*2)``

.. code-block:: python

   import megengine.functional as F
   from megengine.utils.network import Network as Net

   orig_output = graph.output_vars[0] # 获取模型输出
   graph.remove_output(orig_output)   # 将 orig_output 从 graph.output_vars 列表删除

   out = F.relu(orig_output.var)
   new_output = graph.add_dep_oprs(out)[0]

   graph.add_output(new_output) # 将 new_output 添加到 graph.output_vars 列表中

修改 Opr 名字
~~~~~~~~~~~~~
:py:meth:`.Network.modify_opr_names` ``(modifier)``
  批量修改 opr 名字： 其中 ``modifier`` 可以是字符串/函数——
  
  * 当 ``modifier`` 是字符串 ``s`` 时，会在所有 opr 原有名字前添加 ``"s."`` 前缀
  * 如果 ``modifier`` 是函数，该函数接收原有名字做参数，返回新名字

修改 Batch Size
~~~~~~~~~~~~~~~
:py:meth:`.Network.reset_batch_size` ``(batchsize, blacklist)``
  修改所有输入节点（Host2DeviceCopy）的 ``batchsize`` ，在 ``blacklist`` 内的节点不会被修改。

模型修改示例
~~~~~~~~~~~~

下面的代码使用 :py:class:`~.trace` 构造静态图，并在原模型输出上添加 :py:meth:`~.warp_perspective` 变换：

.. code-block:: python

   import megengine as mge
   from megengine import module as M
   import megengine.functional as F
   from megengine.jit.tracing import trace
   from megengine.utils.network import Network
   import numpy as np


   @trace(symbolic=True, capture_as_const=True)
   def perspective_transform(data):
       # M = xxxx 省略计算 transform matrix 代码
       result = F.vision.warp_perspective(data, M, (48, 160))
       return result

   def edit():

       # 通过 trace 得到 perspec_transform 对应的计算图
       perspective_transform(data)
       perspective_transform.dump("transform.mge", optimize_for_inference=False)

       # 使用Network API 加载原模型和 perspective_transform 子模块
       origin_model = Network.load("/path/to/origin_model")
       transform = Network.load("transform.mge")
     
     
       # 获取原模型输出
       orig_output = origin_model.output_vars[0]

       # 把 perspective transform 对应的计算图添加到原模型中
       # Network 中的图结构包括了输出节点依赖的所有 var/op，因此只需要把 perspective_transform 输出加到原模型的 output_vars 列表中即可
       origin_model.add_output(*transform.output_vars)

       # 获取 perspective_transform 输入
       transform_input = transform.data_providers_filter.as_unique()

       # 通过替换输入VarNode，把原模型输出接到 perspective_transform 输入上，完成模型拼接
       for opr in origin_model.opr_filter.has_input(transform_input):
           opr.inputs[0] = orig_output

       # 如果只需要拿到 warp perspective 结果，应该把原模型输出从 output_vars 列表中移除
       origin_model.remove_output(orig_output)
       origin_model.dump("out.mge")

.. warning::

   #. :py:class:`megengine.tensor.Tensor` 只用于动态图，不能与 VarNode 混用，
      例如 Tensor 与 VarNode 相互赋值，functional API 同时接收 Tensor 和 VarNode 作为输入等。
      对应地，向计算图中添加常量可以使用 :py:meth:`.Network.make_const` ``(data)`` ，不能使用 ``megengine.tensor(data)`` ;
      添加新输入可以使用 :py:meth:`.Network.make_input_node` .

   #. 向 ``.mge`` 模型的计算图中添加新 opr 可以使用 megengine functional API. 
      functional API 接 收mgb VarNode 作为输入时会向 VarNode 所属计算图插入 opr，返回该 opr 输出的 mgb VarNode

   #. 目前 Network 提供的 make_const make_input_node, 各类查找 VarNode API 返回值类型为 network_node.VarNode 
      (network_node.VarNode 中的 var 属性是mgb VarNode)，因此使用 Network API 获取 network_node.VarNode 后，
      如果需要传给 functional 造计算图，需要手动调用 ``.var`` 后再传给 functional （具体可参考上述.mge模型修改示例）。
      近期会添加 functional API 直接接收 Network.VarNode 的支持。

   #. 目前  network_node.VarNode  不支持 array method（不支持 + - * / 等操作符 和 advance indexing)/. 
      遇到 VarNode 不支持的操作（例如advance indexing），可以考虑用 trace + 动态图 构造出静态图，
      把该静态图拼接到原模型上。近期会为network_node.VarNode 加上 arraymethodmixin.

