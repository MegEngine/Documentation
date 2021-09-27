.. _api-example:

===============
API 与 使用方式
===============
.. note::

   注意：TracedModule API 在未来一段时间会根据使用反馈进行调整，请关注 github release note 获取变更。欢迎在文档或 Github 提交使用反馈，一起让模型到应用更快更便捷！

以 resnet18 为例介绍 TracedModule 的使用方式，model.py 可从 `这里 <https://github.com/MegEngine/Models/blob/master/official/vision/classification/resnet/model.py>`__ 下载。
通过 :py:func:`~.trace_module` 方法将一个普通的 Module 转变成 TracedModule。接口形式如下：

.. code-block:: python

    def trace_module(module: Module, *inputs, **kwargs) -> TracedModule:
        """
        module: 要被 trace 的原 Module
        inputs/kwargs: Module.forward 所需的参数
        """
        ...
        return traced_module

将自定义的 resnet18（Module）转换为 TracedModule：

.. code-block:: python

    import megengine.functional as F
    import megengine.module as M
    import megengine as mge
    import model

    # resnet : Module
    resnet = model.__dict__["resnet18"]()
    
    import megengine.traced_module as tm
    inp = F.zeros(shape=(1,3,224,224))

    # traced_resnet : TracedModule
    traced_resnet =  tm.trace_module(resnet, inp)

Node 、Expr 、InternalGraph 的常用属性和方法
============================================

TracedModule.graph
------------------
查看 TracedModule 对应的 InternalGraph，以及子 TracedModule 对应的 InternalGraph。通过 ``"{:ip}".format(InternalGraph)`` 查看 Expr 的 id，Node 的 id 和 name。在一个 InternalGraph 中每个 Expr 和 Node 都有一个唯一的 id 与其对应。通过这个 id 可以区分和定位不同的 Expr 与 Node。

.. admonition:: TracedModule.graph
    :class: dropdown

    .. code:: python

        print(traced_resnet.graph)
        '''
        ResNet.Graph (self, x) {
                %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
                %6:     conv1_out = conv1(x, )
                %7:     bn1 = getattr(self, "bn1") -> (BatchNorm2d)
                %8:     bn1_out = bn1(conv1_out, )
                %9:     relu_out = nn.relu(bn1_out, )
                %10:    maxpool = getattr(self, "maxpool") -> (MaxPool2d)
                %11:    maxpool_out = maxpool(relu_out, )
                %12:    layer1 = getattr(self, "layer1") -> (Module)
                %13:    layer1_out = layer1(maxpool_out, )
                %50:    layer2 = getattr(self, "layer2") -> (Module)
                %51:    layer2_out = layer2(layer1_out, )
                %94:    layer3 = getattr(self, "layer3") -> (Module)
                %95:    layer3_out = layer3(layer2_out, )
                %138:   layer4 = getattr(self, "layer4") -> (Module)
                %139:   layer4_out = layer4(layer3_out, )
                %182:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, )
                %183:   flatten_out = tensor.flatten(avg_pool2d_out, 1, )
                %184:   fc = getattr(self, "fc") -> (Linear)
                %185:   fc_out = fc(flatten_out, )
                return fc_out
        }
        注：
        ResNet.forward 中的 x = self.conv1(x) 将会被解析为以下两个 Expr，即先获取 conv1 = self.conv1 ，再执行 conv1_out = conv1(x)
        %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
        %6:     conv1_out = conv1(x, )
        
        其中百分号后的数字为 Expr 的 id。
        conv1 为一个 ModuleNode 其指向 self.conv1
        conv1_out 为一个 TensorNode, 表示调用 self.conv1 的输出， 一般 TensorNode 的 name 以 _out 结尾。
        '''

        # resnet18 的子 Module 如 layer1 同样是一个 TracedModule，可以访问其 graph
        print(traced_resnet.layer1.graph)
        '''
        layer1.Graph (self, inp) {
                %16:    _0 = getattr(self, "0") -> (Module)
                %17:    _1 = getattr(self, "1") -> (Module)
                %18:    _0_out = _0(inp, )
                %34:    _1_out = _1(_0_out, )
                return _1_out
        }
        '''

        # 同样也可以获取 layer1 中 0 的 graph
        print(getattr(traced_resnet.layer1, "0").graph)
        '''
        _0.Graph (self, x) {
                %21:    conv1 = getattr(self, "conv1") -> (Conv2d)
                %22:    conv1_out = conv1(x, )
                %23:    bn1 = getattr(self, "bn1") -> (BatchNorm2d)
                %24:    bn1_out = bn1(conv1_out, )
                %25:    relu_out = nn.relu(bn1_out, )
                %26:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %27:    conv2_out = conv2(relu_out, )
                %28:    bn2 = getattr(self, "bn2") -> (BatchNorm2d)
                %29:    bn2_out = bn2(conv2_out, )
                %30:    downsample = getattr(self, "downsample") -> (Identity)
                %31:    downsample_out = downsample(x, )
                %32:    iadd_out = bn2_out.__iadd__(downsample_out, )
                %33:    relu_out_1 = nn.relu(iadd_out, )
                return relu_out_1
        }
        '''

        # 通过 format 显示更多的 Node 信息，参数 i 表示输出 Node 的 id, 参数 p 表示输出 Node 前缀名
        print(traced_resnet.layer1.graph)  # 输出 Node 名字，清晰明了，容易与 源码 对应
        
        print("{:p}".format(traced_resnet.layer1.graph))  # 输出 Node 完整的名字，可用于 get_node_by_name 来查找对应的 Node
        
        print("{:i}".format(traced_resnet.layer1.graph))  # 输出 Node 的 id + _name，可用于 get_node_by_id 来查找对应的 Node
        
        print("{:ip}".format(traced_resnet.layer1.graph)) # 输出 Node 的 id 以及完整的名字
        '''
        layer1.Graph (self, inp) {
                %16:    _0 = getattr(self, "0") -> (Module)
                %17:    _1 = getattr(self, "1") -> (Module)
                %18:    _0_out = _0(inp, )
                %34:    _1_out = _1(_0_out, )
                return _1_out
        }
        layer1.Graph (ResNet_layer1_self, ResNet_layer1_inp) {
                %16:    ResNet_layer1_0 = getattr(ResNet_layer1_self, "0") -> (Module)
                %17:    ResNet_layer1_1 = getattr(ResNet_layer1_self, "1") -> (Module)
                %18:    ResNet_layer1_0_out = ResNet_layer1_0(ResNet_layer1_inp, )
                %34:    ResNet_layer1_1_out = ResNet_layer1_1(ResNet_layer1_0_out, )
                return ResNet_layer1_1_out
        }
        layer1.Graph (%12_self, %13_inp) {
                %16:    %14_0 = getattr(%12_self, "0") -> (Module)
                %17:    %15_1 = getattr(%12_self, "1") -> (Module)
                %18:    %31_0_out = %14_0(%13_inp, )
                %34:    %47_1_out = %15_1(%31_0_out, )
                return %47_1_out
        }
        layer1.Graph (%12_ResNet_layer1_self, %13_ResNet_layer1_inp) {
                %16:    %14_ResNet_layer1_0 = getattr(%12_ResNet_layer1_self, "0") -> (Module)
                %17:    %15_ResNet_layer1_1 = getattr(%12_ResNet_layer1_self, "1") -> (Module)
                %18:    %31_ResNet_layer1_0_out = %14_ResNet_layer1_0(%13_ResNet_layer1_inp, )
                %34:    %47_ResNet_layer1_1_out = %15_ResNet_layer1_1(%31_ResNet_layer1_0_out, )
                return %47_ResNet_layer1_1_out
        }
        '''

InternalGraph.exprs
-------------------
遍历 Graph 中的 Expr。通过访问 :py:meth:`.InternalGraph.exprs` 可得到该 graph 按执行顺序的 Expr 序列。

:py:meth:`.InternalGraph.exprs` ``(recursive : bool = True)``
    按 Expr 执行顺序获取 Expr 执行序列
    
    * ``recursive``:  是否获取子 Graph 中的 Expr，默认为 True

.. admonition:: InternalGraph.exprs
    :class: dropdown

    .. code:: python

        # recursive = False，只遍历当前 graph 中的 Expr；recursive = True, 同时遍历子 Graph 的 Expr
        exprs = traced_resnet.graph.exprs(recursive=False).as_list()
        for expr in exprs:
            print(expr)
        '''
        得到如下结果：
        %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
        %6:     conv1_out = conv1(x, )
        %7:     bn1 = getattr(self, "bn1") -> (BatchNorm2d)
        %8:     bn1_out = bn1(conv1_out, )
        %9:     relu_out = nn.relu(bn1_out, )
        %10:    maxpool = getattr(self, "maxpool") -> (MaxPool2d)
        %11:    maxpool_out = maxpool(relu_out, )
        %12:    layer1 = getattr(self, "layer1") -> (Module)
        %13:    layer1_out = layer1(maxpool_out, )
        %50:    layer2 = getattr(self, "layer2") -> (Module)
        %51:    layer2_out = layer2(layer1_out, )
        %94:    layer3 = getattr(self, "layer3") -> (Module)
        %95:    layer3_out = layer3(layer2_out, )
        %138:   layer4 = getattr(self, "layer4") -> (Module)
        %139:   layer4_out = layer4(layer3_out, )
        %182:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, )
        %183:   flatten_out = tensor.flatten(avg_pool2d_out, 1, )
        %184:   fc = getattr(self, "fc") -> (Linear)
        %185:   fc_out = fc(flatten_out, )
        '''

InternalGraph.nodes
-------------------
遍历 Graph 中的 Node。通过访问 :py:meth:`.InternalGraph.nodes` 可得到该 graph 中的 Node 序列。

:py:meth:`.InternalGraph.nodes` ``(recursive : bool = True)``
    按 id 从小到大返回 Graph 中的 Node
    
    * ``recursive``:  是否获取子 Graph 中的 Node，默认为 True

.. admonition:: InternalGraph.nodes
    :class: dropdown

    .. code:: python

        # recursive = False，只遍历当前 graph 中的 Node；recursive = True, 同时遍历子 Graph 的 Node
        nodes = traced_resnet.graph.nodes(recursive=False).as_list()
        for node in nodes:
            print(node)
        '''
        得到如下结果：
        self
        x
        conv1
        conv1_out
        bn1
        bn1_out
        relu_out
        maxpool
        maxpool_out
        layer1
        layer1_out
        layer2
        layer2_out
        layer3
        layer3_out
        layer4
        layer4_out
        avg_pool2d_out
        flatten_out
        fc
        fc_out
        '''

Expr.inputs & Expr.outputs
--------------------------
通过访问 Expr 的 inputs 和 outputs 属性，可获得该 Expr 的输入和输出 Node。

:py:attr:`.Expr.inputs` ``: List[Node]``
:py:attr:`.Expr.outputs` ``: List[Node]``

.. admonition:: Expr.inputs & Expr.outputs
    :class: dropdown

    .. code:: python

        # 通过调用 InternalGraph 的 get_expr_by_id 的方法可以直接获取到 graph 中的某个 Expr
        expr = traced_resnet.graph.get_expr_by_id(5).as_unique()
        print(expr)
        print(expr.inputs)
        print(expr.outputs)
        '''
        得到如下结果：
        %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
        [self]
        [conv1]
        '''

Node.expr
---------
通过访问 Node 的 expr 属性，可获得该 Node 是由哪个 Expr 生成的。

:py:attr:`.Node.expr` ``: Expr``

.. admonition:: Node.expr
    :class: dropdown

    .. code:: python

        expr = traced_resnet.graph.get_expr_by_id(6).as_unique()
        inp_0 = expr.inputs[0] # inp_0 : ModuleNode
        print(inp_0.expr)
        '''
        得到如下结果：
        %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
        '''

Node.users
----------
通过访问 Node 的 users 属性，可获得该 Node 是将会被哪些 Expr 作为输入。

:py:attr:`.Node.users` ``: Lsit[Expr]``

.. admonition:: Node.users
    :class: dropdown

    .. code:: python

        expr = traced_resnet.graph.get_expr_by_id(6).as_unique()
        oup_0 = expr.outputs[0] # oup_0 : TensorNode
        print(oup_0.users)
        '''
        得到如下结果：
        [%8:    bn1_out = bn1(conv1_out, )]
        '''

ModuleNode.owner
----------------
通过访问 ModuleNode 的 owner 属性，可直接访问该 ModuleNode 所对应的 Module。

:py:attr:`.ModuleNode.owner` ``: Module``

.. admonition:: ModuleNode.owner
    :class: dropdown

    .. code:: python

        expr = traced_resnet.graph.get_expr_by_id(6).as_unique()
        print(expr)
        m_node = expr.inputs[0] # m_node : ModuleNode
        
        conv = m_node.owner # conv : Conv2d
        print(conv)
        print(type(conv))
        print(type(conv.weight))
        '''
        得到如下结果：
        %6:     conv1_out = conv1(x, )
        Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        <class 'megengine.module.conv.Conv2d'>
        <class 'megengine.tensor.Parameter'>
        '''

Node.top_graph & Expr.top_graph
-------------------------------
通过访问 Node 或 Expr 的 top_graph 属性，可直获得该 Node 或 Expr 所属的 InternalGraph。

:py:attr:`.Node.top_graph` ``: InternalGraph``

:py:attr:`.Expr.top_graph` ``: InternalGraph``

.. admonition:: Node.top_graph & Expr.top_graph
    :class: dropdown

    .. code:: python

        # 获取 layer1 的 graph
        layer1_graph = traced_resnet.layer1.graph
        print(layer1_graph)
        
        # 通过 get_node_by_id 方法 获取 layer1 中的 id 为 31 的 TensorNode
        node_31 = traced_resnet.graph.get_node_by_id(31).as_unique()
        
        assert node_31.top_graph == layer1_graph
        assert node_31.top_graph != traced_resnet.graph
        
        # 获取生成 node_31 的 Expr，并判断其是否在 layer1_graph 中
        expr_18 = node_31.expr
        assert expr_18.top_graph == layer1_graph
        assert expr_18.top_graph != traced_resnet.graph
        
        '''
        获得如下结果：
        ResNet_layer1.Graph (self, inp) {
                %16:    _0 = getattr(self, "0") -> (Module)
                %17:    _1 = getattr(self, "1") -> (Module)
                %18:    _0_out = _0(inp, )
                %34:    _1_out = _1(_0_out, )
                return _1_out
        }
        '''

InternalGraph.eval
------------------
通过访问 InternalGraph 的 eval 方法，可以直接运行该 Graph。

:py:meth:`.InternalGraph.eval` ``(*inputs)``
    将 Tensor 直接输入 Graph 并返回按 Expr 执行序列执行后的结果
    
    * ``inputs`` 模型的输入

.. admonition:: InternalGraph.eval 
    :class: dropdown

    .. code:: python

        resnet_graph = traced_resnet.graph
        
        # 执行 resnet 的 graph
        inp_0 = F.zeros(shape = (1,3,224,224))
        fc_out = resnet_graph.eval(inp_0)   # resnet_graph : InternalGraph
        print(fc_out[0].shape)
        
        # 执行 resnet 中 layer2 的 graph
        layer2_graph = traced_resnet.layer2.graph # layer2_graph : InternalGraph
        inp_1 = F.zeros(shape = (1,64,56,56))
        layer2_out = layer2_graph.eval(inp_1)
        print(layer2_out[0].shape)
        '''
        获得如下结果：
        (1, 1000)
        (1, 128, 28, 28)
        '''

Node 和 Expr 的查找方法
=======================

BaseFilter
----------
:py:class:`~.BaseFilter` 是一个可迭代的类，其提供了一些方法将迭代器转换为 ``list``, ``dict`` 等。

:py:class:`~.NodeFilter` 和 :py:class:`~.ExprFilter` 继承于 :py:class:`~.BaseFilter`，NodeFilter 负责处理 Node，ExprFilter 负责处理 Expr。

* :py:meth:`.BaseFilter.as_list` : 返回 Node 或 Expr 列表
* :py:meth:`.BaseFilter.as_dict` : 返回 Node 或 Expr 的 id 和 Node 或 Expr 组成的字典
* :py:meth:`.BaseFilter.as_unique` : 如果查找到的 Node 或 Expr 只有一个，直接返回该 Node 或 Expr, 否则报错
* :py:meth:`.BaseFilter.as_count` : 返回查找到 Node 或 Expr 的数量

get_node_by_id
--------------
通过 Node 的 id 从 Graph 中获取对应 id 的 Node。

:py:meth:`.InternalGraph.get_node_by_id` ``(node_id: List[int] = None, recursive=True)``
    获取 InternalGraph 中 id 在 node_id 里的 Node，支持一次查找多个 Node

    * ``node_id`` 待查找 Node 的 id 列表
    * ``recursive`` 是否查找子 Graph 中的 Node，默认为 True

.. admonition:: get_node_by_id
    :class: dropdown

    .. code:: python

        graph = traced_resnet.graph # graph : InternalGraph
        
        nodes = graph.get_node_by_id([4, 8, 31]).as_list()

        print(nodes)
        print(["{:i}".format(n) for n in nodes])

        node_4, node_8, node_31 = nodes
        assert node_4.top_graph == node_8.top_graph
        assert node_4.top_graph != node_31.top_graph  # 可以直接获取其子 Module 的 graph 中的 Node
        
        '''
        获得如下结果：
        [conv1, relu_out, _0_out]
        ['%4_conv1', '%8_relu_out', '%31_0_out']
        '''

get_expr_by_id
--------------
与 get_node_by_id 类似，该方法通过 Expr 的 id 从 Graph 中获取对应 id 的 Expr

:py:meth:`.InternalGraph.get_expr_by_id` ``(expr_id: List[int] = None, recursive=True)``
    获取 InternalGraph 中 id 在 expr_id 里的 Expr，支持一次查找多个 Expr

    * ``expr_id`` 待查找 Expr 的 id 列表
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

.. admonition:: get_expr_by_id
    :class: dropdown

    .. code:: python

        graph = traced_resnet.graph # graph : InternalGraph
        
        exprs = graph.get_expr_by_id([5, 9, 18]).as_list()
        print(exprs)
        
        expr_5, expr_9, expr_18 = exprs
        assert expr_5.top_graph == expr_9.top_graph
        assert expr_5.top_graph != expr_18.top_graph   # 可以直接获取其 子 Module 的 graph 中的 Expr
        '''
        获得如下结果：
        [%5:    conv1 = getattr(self, "conv1") -> (Conv2d),
        %9: relu_out = nn.relu(bn1_out, ),
        %18:     _0_out = _0(inp, )]
        '''
        '''
        Expr 的字符串形式的百分号之后的数字即为该 Expr 的 id
        '''

get_function_by_type
--------------------
通过该方法查找 Graph 中调用了某个 function 的 CallFunction Expr

:py:meth:`.InternalGraph.get_function_by_type` ``(func: Callable = None, recursive=True)``
    获取 InternalGraph 中 ``self.func == func`` 的 CallFunction

    * ``func`` 可调用的函数
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

.. admonition:: get_function_by_type
    :class: dropdown

    .. code:: python

        # 获取 Layer1 中所有描述 F.relu 的 Expr
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        exprs = graph.get_function_by_type(F.relu).as_list()
        
        print(exprs)
        '''
        获得如下结果：
        [%25:  relu_out = nn.relu(bn1_out, ),
        %33:  relu_out_1 = nn.relu(iadd_out, ),
        %41:  relu_out = nn.relu(bn1_out, ),
        %49:  relu_out_1 = nn.relu(iadd_out, )]
        '''

get_method_by_type
------------------
通过该方法查找 Graph 中调用了某个 method 的 CallMethod Expr

:py:meth:`.InternalGraph.get_method_by_type` ``(method: str = None, recursive=True)``
    获取 InternalGraph 中 ``self.method == method`` 的 CallMethod

    * ``method`` 待查找某对象的方法的名字（该方法是一个可调用的函数）
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

.. admonition:: get_method_by_type
    :class: dropdown

    .. code:: python

        # 获取 Layer1 中所有描述 Module.forward 的 Expr （等价于 Module.__call__）
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        exprs = graph.get_method_by_type("__call__").as_list()
        
        print(exprs)
        '''
        获得如下结果：
        [%18:    _0_out = _0(inp, ),
        %22:    conv1_out = conv1(x, ),
        %24:    bn1_out = bn1(conv1_out, ),
        %27:    conv2_out = conv2(relu_out, ),
        %29:    bn2_out = bn2(conv2_out, ),
        ...]
        '''

get_module_by_type
------------------
通过该方法查找 Graph 中对应某种 Module 的 ModuleNode

:py:meth:`.InternalGraph.get_module_by_type` ``(module_cls: Module, recursive=True)``
    获取 InternalGraph 中对应于 ``module_cls`` 的 ModuleNode

    * ``module_cls`` Module 某个子类
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

.. admonition:: get_module_by_type
    :class: dropdown

    .. code:: python

        # 获取 Layer1 中所有描述 BatchNorm2d 的 ModuleNode
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        nodes = graph.get_module_by_type(M.BatchNorm2d).as_list()
        
        print(nodes)
        
        for n in nodes:
            # 通过 ModuleNode.owner 直接访问其真正对应的 Module
            assert isinstance(n.owner, M.BatchNorm2d)
        
        '''
        获得如下结果：
        [bn1, bn2, bn1, bn2]
        '''

图手术常用方法
==============

add_input_node
--------------
为最顶层的 InternalGraph 增加一个输入，此输入会作为一个 free_varargs 参数（即无形参名称）。当调用该方法的 Graph 是一个子 Graph 时，将会报错。

:py:meth:`.InternalGraph.add_input_node` ``(shape, dtype="float32", name="args")``
    为顶层 Graph 新增一个输入

    * ``shape`` 新增输入的 shape
    * ``dtype`` 新增输入的 dtype，默认为 "float32"
    * ``name``  新增输入的名字，默认为 "args"，若该名字在 Graph 种已存在，则会在 name 后添加后缀，以保证 name 在 Graph 在的唯一性。

.. admonition:: add_input_node
    :class: dropdown

    .. code:: python

        graph = traced_resnet.graph
        
        graph = traced_resnet.graph # graph : InternalGraph
        new_inp_node = graph.add_input_node(shape=(1,3,224,224), dtype="float32", name="new_data")
        print(graph)
        print("new input: ",new_inp_node)
        
        '''
        获得如下结果：
        ResNet.Graph (self, x, new_data) {
                %5:     conv1 = getattr(self, "conv1") -> (Conv2d)
                ...
                return fc_out
        }
        new input:  new_data
        '''

add_output_node
---------------
为最顶层的 InternalGraph 增加一个输出，此输入会作为输出元组种的最后一个元素。当调用该方法的 Graph 是一个子 Graph 时，将会报错。

:py:meth:`.InternalGraph.add_output_node` ``(node: TensorNode)``
    将 Graph 种的某个 Node 作为 Graph 的一个输出

    * ``node`` Graph 中的某 Node

.. admonition:: add_output_node
    :class: dropdown

    .. code:: python

        graph = traced_resnet.graph
        
        fc_inp_node = graph.get_node_by_id(182).as_unique()
        
        graph.add_output_node(fc_inp_node)
        
        print(graph)
        
        fc_out, fc_inp = traced_resnet(inp)
        
        print("fc input shape: ", fc_inp.shape)
        print("fc output shape: ", fc_out.shape)
        
        '''
        获得如下结果：
        ResNet.Graph (self, x) {
                ...
                %183:   flatten_out = tensor.flatten(avg_pool2d_out, 1, )
                %184:   fc = getattr(self, "fc") -> (Linear)
                %185:   fc_out = fc(flatten_out, )
                return fc_out, flatten_out
        }
        fc input shape:  (1, 512)
        fc output shape:  (1, 1000)
        '''

reset_outputs
-------------
重新设置最顶层 InternalGraph 的输出。当调用该方法的 Graph 是一个子 Graph 时，将会报错。

当要改变的输出较多时，一个一个调用 ``add_output_node`` 较为麻烦，通过 ``reset_outputs`` 方法一次性重置输出内容于结构。

:py:meth:`.InternalGraph.reset_outputs` ``(node: outputs)``
    重置 Graph 的输出

    * ``node`` 由 Graph 中的 TensorNode 构成的某种结构，支持 ``list``, ``dict``, ``tuple`` 等（最底层的元素必须是 TensorNode）。 

.. admonition:: reset_outputs
    :class: dropdown

    .. code:: python

        graph = traced_resnet.graph
        
        avgpool_inp_node = graph.get_node_by_id(180).as_unique()
        fc_inp_node = graph.get_node_by_id(182).as_unique()
        fc_out_node = graph.outputs[0]
        
        # 把 fc 的输入和输出以 Dict 形式输出 并与 avgppol 的输入组成 tuple
        new_outputs = ({"fc_inp": fc_inp_node, "fc_out": fc_out_node }, avgpool_inp_node)
        # 将 new_outputs 作为 graph 的新的输出
        graph.reset_outputs(new_outputs)
        
        print(graph)
        
        fc, avgpool_inp = traced_resnet(inp)
        print("avgpool output shape: ", avgpool_inp.shape)
        print("fc input shape: ", fc["fc_inp"].shape)
        print("fc output shape: ", fc["fc_inp"].shape)
        '''
        获得如下结果：
        ResNet.Graph (self, x) {
                ...
                %138:   layer4 = getattr(self, "layer4") -> (Module)
                %139:   layer4_out = layer4(layer3_out, )
                %182:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, )
                %183:   flatten_out = tensor.flatten(avg_pool2d_out, 1, )
                %184:   fc = getattr(self, "fc") -> (Linear)
                %185:   fc_out = fc(flatten_out, )
                return flatten_out, fc_out, layer4_out
        }
        avgpool output shape:  (1, 512, 7, 7)
        fc input shape:  (1, 512)
        fc output shape:  (1, 512)
        '''

replace_node
------------
替换 InternalGraph 中的指定 Node。可用于新增 Expr 后替换一些 Node，或结合 :py:meth:`.InternalGraph.compile` 删某些 Expr。

:py:meth:`.InternalGraph.replace_node` ``(repl_dict : Dict[Node, Node])``
    替换 Graph 中的 ``key`` 替换为 ``value``

    * ``repl_dict`` 为一个 ``key`` 和 ``value`` 都为 Node 的字典，且 ``key`` 和 ``value`` 必须在同一个 Graph 中。生成 ``value`` 的 Expr 之后的所有将 ``key`` 作为输入的 Expr 的输入将被替换为 ``value``。 

.. admonition:: replace_node
    :class: dropdown

    .. code:: python

        # 将 layer1 中的所有描述 F.relu 的 Expr 删除
        
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        # 获取 layer1 中所有描述 F.relu 的  Expr
        exprs = graph.get_function_by_type(F.relu).as_list()
        
        print("replace before: ")
        print(exprs[-1].top_graph)
        
        for id, expr in enumerate(exprs):
            
            # 获取当前 expr 所属的 InternalGraph
            cur_graph = expr.top_graph
            
            # 获取 F.relu 的 输出 TensorNode
            relu_inp_node = expr.inputs[0]
            relu_out_node = expr.outputs[0]
            
            # cur_graph 所有将 relu_out_node 作为输入的 Expr 替换为 relu_inp_node 作为输入
            cur_graph.replace_node({relu_out_node: relu_inp_node})
            
            # 删除 cur_graph 中与 cur_graph.outputs 无关的 Node 和 Expr
            cur_graph.compile()
        
        assert len(graph.get_function_by_type(F.relu).as_list()) == 0
        
        print("replace after: ")
        print(exprs[-1].top_graph)
        '''
        获得如下结果：
        replace before:
        ResNet_layer1_1.Graph (self, x) {
                %37:    conv1 = getattr(self, "conv1") -> (Conv2d)
                %38:    conv1_out = conv1(x, )
                %39:    bn1 = getattr(self, "bn1") -> (BatchNorm2d)
                %40:    bn1_out = bn1(conv1_out, )
                %41:    relu_out = nn.relu(bn1_out, )
                %42:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %43:    conv2_out = conv2(relu_out, )
                %44:    bn2 = getattr(self, "bn2") -> (BatchNorm2d)
                %45:    bn2_out = bn2(conv2_out, )
                %46:    downsample = getattr(self, "downsample") -> (Identity)
                %47:    downsample_out = downsample(x, )
                %48:    iadd_out = bn2_out.__iadd__(downsample_out, )
                %49:    relu_out_1 = nn.relu(iadd_out, )
                return relu_out_1
        }
        replace after:
        ResNet_layer1_1.Graph (self, x) {
                %37:    conv1 = getattr(self, "conv1") -> (Conv2d)
                %38:    conv1_out = conv1(x, )
                %39:    bn1 = getattr(self, "bn1") -> (BatchNorm2d)
                %40:    bn1_out = bn1(conv1_out, )
                %42:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %43:    conv2_out = conv2(bn1_out, )
                %44:    bn2 = getattr(self, "bn2") -> (BatchNorm2d)
                %45:    bn2_out = bn2(conv2_out, )
                %46:    downsample = getattr(self, "downsample") -> (Identity)
                %47:    downsample_out = downsample(x, )
                %48:    iadd_out = bn2_out.__iadd__(downsample_out, )
                return iadd_out
        }
        '''


insert_exprs
------------
向 InternalGraph 中插入 Expr。可用于插入 ``function`` 或 ``Module`` ，并在插入的过程中将这些 ``function`` 或 ``Module`` 解析为 Expr 或 TracedModule。

一般与 ``replace_node`` 和 ``compile`` 一起使用完成图手术。

:py:meth:`.InternalGraph.insert_exprs` ``(expr: Optional[Expr] = None)``
    向 Graph 中插入 Expr

    * ``expr`` 在 :py:attr:`.InternalGraph._exprs` 中 ``expr`` 之后插入新的 ``function`` 或 ``Module``。 

在 ``insert_exprs`` 的作用域里，``TensorNode`` 可以当作 ``Tensor`` 使用， ``ModuleNode`` 可以当作 ``Module``。

.. admonition:: insert_exprs 插入 function
    :class: dropdown

    .. code:: python

        # 向 layer1 中的所有 F.relu 后插入一个 F.neg 函数
        
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        # 获取 layer1 中所有描述 F.relu 的  Expr
        exprs = graph.get_function_by_type(F.relu).as_list()
        
        for id, expr in enumerate(exprs):
            
            # 获取当前 expr 所属的 InternalGraph
            cur_graph = expr.top_graph
            
            print("%d : insert function before"%id)
            print(cur_graph)
            print(expr)
        
            # 获取 F.relu 的 输出 TensorNode
            relu_out_node = expr.outputs[0]
            
            with cur_graph.insert_exprs():
                neg_out_node = F.neg(relu_out_node)
        
            # cur_graph 所有将 relu_out_node 作为输入的 Expr 替换为 neg_out_node 作为输入
            cur_graph.replace_node({relu_out_node: neg_out_node})
            
            # 删除 cur_graph 中与 cur_graph.outputs 无关的 Node 和 Expr
            cur_graph.compile()
        
            print("%d : insert function after"%id)
            print(cur_graph)
        '''
        获得如下结果：
        0 : insert function before
        ResNet_layer1_0.Graph (self, x) {
                ...
                %25:    relu_out = nn.relu(bn1_out, )
                ...
                return relu_out_1
        }
        %25:    relu_out = nn.relu(bn1_out, )
        0 : insert function after
        ResNet_layer1_0.Graph (self, x) {
                ...
                %25:    relu_out = nn.relu(bn1_out, )
                %186:   neg_out = elemwise.neg(relu_out, )
                %26:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %27:    conv2_out = conv2(neg_out, )
                ...
                return relu_out_1
        }
        ...
        '''

.. admonition:: insert_exprs 替换 function
    :class: dropdown

    .. code:: python

        # 将 layer1 中的 所有 F.relu 替换为 F.relu6
        
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        
        # 获取 layer1 中所有描述 F.relu 的  Expr
        exprs = graph.get_function_by_type(F.relu).as_list()
        
        for id, expr in enumerate(exprs):
            
            # 获取当前 expr 所属的 InternalGraph
            cur_graph = expr.top_graph
            
            print("%d : insert function before"%id)
            print(cur_graph)
            print(expr)
        
            # 获取 F.relu 的 输入和输出 TensorNode
            relu_inp_node = expr.inputs[0]
            relu_out_node = expr.outputs[0]
            
            with cur_graph.insert_exprs():
                relu6_out_node = F.relu6(relu_inp_node)
        
            # cur_graph 所有将 relu_out_node 作为输入的 Expr 替换为 relu6_out_node 作为输入
            cur_graph.replace_node({relu_out_node: relu6_out_node})
            
            # 删除 cur_graph 中与 cur_graph.outputs 无关的 Node 和 Expr
            cur_graph.compile()
        
            print("%d : insert function after"%id)
            print(cur_graph)
        '''
        获得如下结果：
        0 : insert function before
        ResNet_layer1_0.Graph (self, x) {
                ...
                %25:    relu_out = nn.relu(bn1_out, )
                %26:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %27:    conv2_out = conv2(relu_out, )
                ...
                return relu_out_1
        }
        %25:    relu_out = nn.relu(bn1_out, )
        0 : insert function after
        ResNet_layer1_0.Graph (self, x) {
                ...
                %24:    bn1_out = bn1(conv1_out, )
                %186:   relu6_out = nn.relu6(bn1_out, )
                %26:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %27:    conv2_out = conv2(relu6_out, )
                ...
                return relu_out_1
        }
        '''

.. admonition:: insert_exprs 插入 Module
    :class: dropdown

    .. code:: python

        import megengine.functional as F
        import megengine.module as M

        class Mod(M.Module):
            def forward(self, x):
                return x

        net = Mod()

        import megengine.traced_module as tm
        inp = F.zeros(shape=(1,))
        # traced_net : TracedModule
        traced_net =  tm.trace_module(net, inp)

        graph = traced_net.graph # graph : InternalGraph
        setattr(traced_net, "mod_0", Mod())

        self, x = graph.inputs
        with graph.insert_exprs():
            mod_out = self.mod_0(x)

        graph.add_output_node(mod_out)
        print(graph)
        '''
        Mod.Graph (self, x) {
            %5:     mod_0 = getattr(self, "mod_0") -> (Module)
            %6:     mod_0_out = mod_0(x, )
            return x, mod_0_out
        }
        '''

.. note::

    由于 ``__setitem__`` 比较特殊，因此在图手术模式下 TensorNode 的赋值结果作为输出时需要特别注意要图手术结果是否符合预期。

    ::
    
        # x_node 是一个 TensorNode , x_node 的 name 为 x_node
        x = x_node
        with graph.insert_exprs():
            x[0] = 1  # 此操作会解析为 setitem_out = x_node.__setietm__(0, 1, ), 此时变量 x 依然对应的是 x_node
            x[0] = 2  # 此操作会解析为 setitem_out_1 = setitem_out.__setietm__(0, 2, ), 此时变量 x 依然对应的是 x_node
        graph.replace_node({* : x}) #此处实际替换的 x 依然为 x_node

        with graph.insert_exprs():
            x[0] = 1  # 此操作会解析为 setitem_out = x_node.__setietm__(0, 1, ), 此时变量 x 依然对应的是 x_node
            x = x * 1 # 此操作会解析为 mul_out = setitem_out.__mul__(1, ), 此时变量 x 对应的是 mul_out
        graph.replace_node({* : x}) #此处实际替换的 x 为 mul_out

compile
-------
该方法会将 InternalGraph 与输出无关的 Expr 删除。

:py:meth:`.InternalGraph.compile` ``()``

常与 ``insert_exprs`` 和 ``replace_node`` 一起使用。

wrap
----
有时不希望插入的函数被解析为 megengine 内置的 function, 此时可以选择用 :py:meth:`~.traced_module.wrap` 函数将自定义的函数当作 megengine 内置函数处理，
即不再 ``trace`` 到函数内部。

:py:meth:`~.traced_module.wrap` ``(func: Callable)``
    将自定义函数注册为内置函数

    * ``func`` 为一个可调用的对象。 

.. admonition:: wrap
    :class: dropdown

    .. code:: python

        @tm.wrap
        def my_relu6(x):
            x = F.minimum(F.maximum(x, 0), 6)
            return x
            
        graph = traced_resnet.layer1.graph # graph : InternalGraph
        exprs = graph.get_function_by_type(F.relu).as_list()
        
        for id, expr in enumerate(exprs):
            cur_graph = expr.top_graph
        
            relu_inp_node = expr.inputs[0]
            relu_out_node = expr.outputs[0]
        
            with cur_graph.insert_exprs():
                my_relu6_out_node = my_relu6(relu_inp_node)
        
            cur_graph.replace_node({relu_out_node: my_relu6_out_node})
            cur_graph.compile()
        
        print("replace relu after")
        print(exprs[-1].top_graph)
        
        '''
        获得如下结果：
        replace relu after
        ResNet_layer1_1.Graph (self, x) {
                %37:    conv1 = getattr(self, "conv1") -> (Conv2d)
                %38:    conv1_out = conv1(x, )
                %39:    bn1 = getattr(self, "bn1") -> (BatchNorm2d)
                %40:    bn1_out = bn1(conv1_out, )
                %188:   my_relu6_out = __main__.my_relu6(bn1_out, )
                %42:    conv2 = getattr(self, "conv2") -> (Conv2d)
                %43:    conv2_out = conv2(my_relu6_out, )
                %44:    bn2 = getattr(self, "bn2") -> (BatchNorm2d)
                %45:    bn2_out = bn2(conv2_out, )
                %46:    downsample = getattr(self, "downsample") -> (Identity)
                %47:    downsample_out = downsample(x, )
                %48:    iadd_out = bn2_out.__iadd__(downsample_out, )
                %189:   my_relu6_out_1 = __main__.my_relu6(iadd_out, )
                return my_relu6_out_1
        }
        '''

TracedModule 常用方法
=====================

flatten
-------
该方法可去除 InternalGraph 的层次结构，即将子 graph 展开, 并返回一个新的 TracedModule。在新的 TracedModule 中，所有的 ``Getattr`` Expr 将被转换为 ``Constant`` Expr。

:py:meth:`.TracedModule.flatten` ``()``
    返回一个新的 TracedModule，其 Graph 无层次结构

拍平后的 InternalGraph 仅包含内置 Module 的 Expr，此时可以直观的得到数据之间的连接关系。

.. admonition:: flatten
    :class: dropdown

    .. code:: python

        flattened_resnet = traced_resnet.flatten() # traced_resnet : TracedModule , flattened_resnet : TracedModule
        print(flattened_resnet.graph)
        
        '''
        获得如下结果：
        ResNet.Graph (self, x) {
                %0:     conv1 = Constant(self.conv1) -> (Module)
                %1:     conv1_out = conv1(x, )
                %2:     bn1 = Constant(self.bn1) -> (Module)
                %3:     bn1_out = bn1(conv1_out, )
                %4:     relu_out = nn.relu(bn1_out, )
                %5:     maxpool = Constant(self.maxpool) -> (Module)
                %6:     maxpool_out = maxpool(relu_out, )
                %7:     layer1_0_conv1 = Constant(self.layer1.0.conv1) -> (Module)
                %8:     layer1_0_conv1_out = layer1_0_conv1(maxpool_out, )
                %9:     layer1_0_bn1 = Constant(self.layer1.0.bn1) -> (Module)
                %10:    layer1_0_bn1_out = layer1_0_bn1(layer1_0_conv1_out, )
                %11:    layer1_0_relu_out = nn.relu(layer1_0_bn1_out, )
                %12:    layer1_0_conv2 = Constant(self.layer1.0.conv2) -> (Module)
                %13:    layer1_0_conv2_out = layer1_0_conv2(layer1_0_relu_out, )
                %14:    layer1_0_bn2 = Constant(self.layer1.0.bn2) -> (Module)
                %15:    layer1_0_bn2_out = layer1_0_bn2(layer1_0_conv2_out, )
                %16:    layer1_0_downsample = Constant(self.layer1.0.downsample) -> (Module)
                %17:    layer1_0_downsample_out = layer1_0_downsample(maxpool_out, )
                %18:    layer1_0_iadd_out = layer1_0_bn2_out.__iadd__(layer1_0_downsample_out, )
                %19:    layer1_0_out = nn.relu(layer1_0_iadd_out, )
                %20:    layer1_1_conv1 = Constant(self.layer1.1.conv1) -> (Module)
                %21:    layer1_1_conv1_out = layer1_1_conv1(layer1_0_out, )
                %22:    layer1_1_bn1 = Constant(self.layer1.1.bn1) -> (Module)
                %23:    layer1_1_bn1_out = layer1_1_bn1(layer1_1_conv1_out, )
                %24:    layer1_1_relu_out = nn.relu(layer1_1_bn1_out, )
                %25:    layer1_1_conv2 = Constant(self.layer1.1.conv2) -> (Module)
                %26:    layer1_1_conv2_out = layer1_1_conv2(layer1_1_relu_out, )
                %27:    layer1_1_bn2 = Constant(self.layer1.1.bn2) -> (Module)
                %28:    layer1_1_bn2_out = layer1_1_bn2(layer1_1_conv2_out, )
                %29:    layer1_1_downsample = Constant(self.layer1.1.downsample) -> (Module)
                %30:    layer1_1_downsample_out = layer1_1_downsample(layer1_0_out, )
                %31:    layer1_1_iadd_out = layer1_1_bn2_out.__iadd__(layer1_1_downsample_out, )
                %32:    layer1_out = nn.relu(layer1_1_iadd_out, )
                %33:    layer2_0_conv1 = Constant(self.layer2.0.conv1) -> (Module)
                %34:    layer2_0_conv1_out = layer2_0_conv1(layer1_out, )
                %35:    layer2_0_bn1 = Constant(self.layer2.0.bn1) -> (Module)
                %36:    layer2_0_bn1_out = layer2_0_bn1(layer2_0_conv1_out, )
                %37:    layer2_0_relu_out = nn.relu(layer2_0_bn1_out, )
                %38:    layer2_0_conv2 = Constant(self.layer2.0.conv2) -> (Module)
                %39:    layer2_0_conv2_out = layer2_0_conv2(layer2_0_relu_out, )
                %40:    layer2_0_bn2 = Constant(self.layer2.0.bn2) -> (Module)
                %41:    layer2_0_bn2_out = layer2_0_bn2(layer2_0_conv2_out, )
                %42:    layer2_0_downsample_0 = Constant(self.layer2.0.downsample.0) -> (Module)
                %43:    layer2_0_downsample_0_out = layer2_0_downsample_0(layer1_out, )
                %44:    layer2_0_downsample_1 = Constant(self.layer2.0.downsample.1) -> (Module)
                %45:    layer2_0_downsample_out = layer2_0_downsample_1(layer2_0_downsample_0_out, )
                %46:    layer2_0_iadd_out = layer2_0_bn2_out.__iadd__(layer2_0_downsample_out, )
                %47:    layer2_0_out = nn.relu(layer2_0_iadd_out, )
                %48:    layer2_1_conv1 = Constant(self.layer2.1.conv1) -> (Module)
                %49:    layer2_1_conv1_out = layer2_1_conv1(layer2_0_out, )
                %50:    layer2_1_bn1 = Constant(self.layer2.1.bn1) -> (Module)
                %51:    layer2_1_bn1_out = layer2_1_bn1(layer2_1_conv1_out, )
                %52:    layer2_1_relu_out = nn.relu(layer2_1_bn1_out, )
                %53:    layer2_1_conv2 = Constant(self.layer2.1.conv2) -> (Module)
                %54:    layer2_1_conv2_out = layer2_1_conv2(layer2_1_relu_out, )
                %55:    layer2_1_bn2 = Constant(self.layer2.1.bn2) -> (Module)
                %56:    layer2_1_bn2_out = layer2_1_bn2(layer2_1_conv2_out, )
                %57:    layer2_1_downsample = Constant(self.layer2.1.downsample) -> (Module)
                %58:    layer2_1_downsample_out = layer2_1_downsample(layer2_0_out, )
                %59:    layer2_1_iadd_out = layer2_1_bn2_out.__iadd__(layer2_1_downsample_out, )
                %60:    layer2_out = nn.relu(layer2_1_iadd_out, )
                %61:    layer3_0_conv1 = Constant(self.layer3.0.conv1) -> (Module)
                %62:    layer3_0_conv1_out = layer3_0_conv1(layer2_out, )
                %63:    layer3_0_bn1 = Constant(self.layer3.0.bn1) -> (Module)
                %64:    layer3_0_bn1_out = layer3_0_bn1(layer3_0_conv1_out, )
                %65:    layer3_0_relu_out = nn.relu(layer3_0_bn1_out, )
                %66:    layer3_0_conv2 = Constant(self.layer3.0.conv2) -> (Module)
                %67:    layer3_0_conv2_out = layer3_0_conv2(layer3_0_relu_out, )
                %68:    layer3_0_bn2 = Constant(self.layer3.0.bn2) -> (Module)
                %69:    layer3_0_bn2_out = layer3_0_bn2(layer3_0_conv2_out, )
                %70:    layer3_0_downsample_0 = Constant(self.layer3.0.downsample.0) -> (Module)
                %71:    layer3_0_downsample_0_out = layer3_0_downsample_0(layer2_out, )
                %72:    layer3_0_downsample_1 = Constant(self.layer3.0.downsample.1) -> (Module)
                %73:    layer3_0_downsample_out = layer3_0_downsample_1(layer3_0_downsample_0_out, )
                %74:    layer3_0_iadd_out = layer3_0_bn2_out.__iadd__(layer3_0_downsample_out, )
                %75:    layer3_0_out = nn.relu(layer3_0_iadd_out, )
                %76:    layer3_1_conv1 = Constant(self.layer3.1.conv1) -> (Module)
                %77:    layer3_1_conv1_out = layer3_1_conv1(layer3_0_out, )
                %78:    layer3_1_bn1 = Constant(self.layer3.1.bn1) -> (Module)
                %79:    layer3_1_bn1_out = layer3_1_bn1(layer3_1_conv1_out, )
                %80:    layer3_1_relu_out = nn.relu(layer3_1_bn1_out, )
                %81:    layer3_1_conv2 = Constant(self.layer3.1.conv2) -> (Module)
                %82:    layer3_1_conv2_out = layer3_1_conv2(layer3_1_relu_out, )
                %83:    layer3_1_bn2 = Constant(self.layer3.1.bn2) -> (Module)
                %84:    layer3_1_bn2_out = layer3_1_bn2(layer3_1_conv2_out, )
                %85:    layer3_1_downsample = Constant(self.layer3.1.downsample) -> (Module)
                %86:    layer3_1_downsample_out = layer3_1_downsample(layer3_0_out, )
                %87:    layer3_1_iadd_out = layer3_1_bn2_out.__iadd__(layer3_1_downsample_out, )
                %88:    layer3_out = nn.relu(layer3_1_iadd_out, )
                %89:    layer4_0_conv1 = Constant(self.layer4.0.conv1) -> (Module)
                %90:    layer4_0_conv1_out = layer4_0_conv1(layer3_out, )
                %91:    layer4_0_bn1 = Constant(self.layer4.0.bn1) -> (Module)
                %92:    layer4_0_bn1_out = layer4_0_bn1(layer4_0_conv1_out, )
                %93:    layer4_0_relu_out = nn.relu(layer4_0_bn1_out, )
                %94:    layer4_0_conv2 = Constant(self.layer4.0.conv2) -> (Module)
                %95:    layer4_0_conv2_out = layer4_0_conv2(layer4_0_relu_out, )
                %96:    layer4_0_bn2 = Constant(self.layer4.0.bn2) -> (Module)
                %97:    layer4_0_bn2_out = layer4_0_bn2(layer4_0_conv2_out, )
                %98:    layer4_0_downsample_0 = Constant(self.layer4.0.downsample.0) -> (Module)
                %99:    layer4_0_downsample_0_out = layer4_0_downsample_0(layer3_out, )
                %100:   layer4_0_downsample_1 = Constant(self.layer4.0.downsample.1) -> (Module)
                %101:   layer4_0_downsample_out = layer4_0_downsample_1(layer4_0_downsample_0_out, )
                %102:   layer4_0_iadd_out = layer4_0_bn2_out.__iadd__(layer4_0_downsample_out, )
                %103:   layer4_0_out = nn.relu(layer4_0_iadd_out, )
                %104:   layer4_1_conv1 = Constant(self.layer4.1.conv1) -> (Module)
                %105:   layer4_1_conv1_out = layer4_1_conv1(layer4_0_out, )
                %106:   layer4_1_bn1 = Constant(self.layer4.1.bn1) -> (Module)
                %107:   layer4_1_bn1_out = layer4_1_bn1(layer4_1_conv1_out, )
                %108:   layer4_1_relu_out = nn.relu(layer4_1_bn1_out, )
                %109:   layer4_1_conv2 = Constant(self.layer4.1.conv2) -> (Module)
                %110:   layer4_1_conv2_out = layer4_1_conv2(layer4_1_relu_out, )
                %111:   layer4_1_bn2 = Constant(self.layer4.1.bn2) -> (Module)
                %112:   layer4_1_bn2_out = layer4_1_bn2(layer4_1_conv2_out, )
                %113:   layer4_1_downsample = Constant(self.layer4.1.downsample) -> (Module)
                %114:   layer4_1_downsample_out = layer4_1_downsample(layer4_0_out, )
                %115:   layer4_1_iadd_out = layer4_1_bn2_out.__iadd__(layer4_1_downsample_out, )
                %116:   layer4_out = nn.relu(layer4_1_iadd_out, )
                %117:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, )
                %118:   flatten_out = tensor.flatten(avg_pool2d_out, 1, )
                %119:   fc = Constant(self.fc) -> (Module)
                %120:   fc_out = fc(flatten_out, )
                return fc_out
        }
        '''

set_watch_points & clear_watch_points
-------------------------------------
查看 TracedModule 执行时 graph 中某个 Node 对应的真正的 Tensor/Module。

:py:meth:`.TracedModule.set_watch_points` ``(nodes : Sequence[Node])``
    设置观察点

    * ``nodes`` 待观察的 Node

:py:meth:`.TracedModule.clear_watch_points` ``()``
    清除观察点

.. admonition:: set_watch_points & clear_watch_points
    :class: dropdown

    .. code:: python

        # 获取 avgpool 的输入 TensorNode 和 输出 TensorNode
        avgpool_inp_node, avgpool_out_node = traced_resnet.graph.get_node_by_id([180,181])
        
        # 将获得到的 TensorNode 传入 set_watch_points
        traced_resnet.set_watch_points([avgpool_inp_node, avgpool_out_node])
        
        # 执行一次 TracedModule
        inp = F.zeros(shape = (1,3,224,224))
        traced_resnet(inp)  # traced_resnet ： TracedModule
        
        # 获取观察到的 Tensor。
        watched_value = traced_resnet.watch_node_value  # watch_node_value ： Dict[TensorNode, Tensor]
        print("avgpool input shape: ", watched_value[avgpool_inp_node].shape)
        print("avgpool output shape: ", watched_value[avgpool_out_node].shape)
        
        '''
        获得如下结果：
        avgpool input shape:  (1, 512, 7, 7)
        avgpool output shape:  (1, 512, 1, 1)
        '''

set_end_points & clear_end_points
---------------------------------
设置模型停止运行的位置，接受一个 ``List[Node]`` 作为输入，当网络生成所有设置的 ``Node`` 后会立即返回，不再继续往下执行。

:py:meth:`.TracedModule.set_end_points` ``(nodes : Sequence[Node])``
    设置结束运行点

    * ``nodes`` 停止运行处的的 ``Node``

:py:meth:`.TracedModule.clear_end_points` ``()``
    清除结束运行点

.. admonition:: set_end_points & clear_end_points
    :class: dropdown

    .. code:: python

        # 获取 avgpool 的输入 TensorNode 和 输出 TensorNode
        avgpool_inp_node, avgpool_out_node = traced_resnet.graph.get_node_by_id([180,181])
        
        # 将 avgpool 的输入和输出设为结束点，当 avgpool 执行后，就立刻返回结果
        traced_resnet.set_end_points([avgpool_inp_node, avgpool_out_node])
        
        inp = F.zeros(shape = (1,3,224,224))
        avgpool_inp, avgpool_out =  traced_resnet(inp)  # traced_resnet ： TracedModule
        
        print("avgpool input shape: ", avgpool_inp.shape)
        print("avgpool output shape: ", avgpool_out.shape)
        
        '''
        获得如下结果：
        avgpool input shape:  (1, 512, 7, 7)
        avgpool output shape:  (1, 512, 1, 1)
        '''

TracedModule 的局限
===================

* 不支持动态控制流，动态控制流是指 ``if`` 语句中的 ``condition`` 随输入的变化而变化，或者是 ``for``, ``while`` 每次运行的语句不一样。当 ``trace`` 到控制流时，仅会记录并解释满足条件的那个分支。

* 不支持全局变量（Tensor），即跨 Module 使用 ``Tensor`` 将会得到不可预知的结果，如下面的例子。

    .. code:: python

        import megengine.module as M
        import megengine as mge

        g_tensor = mge.Tensor([0])

        class Mod(M.Module):
            def forward(self, x):
                x = g_tensor + 1
                return x

* 被 ``trace`` 的 Module 或 function 参数中的非 ``Tensor`` 类型，将会被看作是常量存储在 Expr 的 :py:attr:`~.Expr.const_val` 属性中，并且该值将不会再变化。

* 当被 ``trace`` 的自定义 Module 被调用了多次，并且每次传入参数中的非 ``Tensor`` 数据不一致时，将会被 ``trace`` 出多个 Graph。此时将无法通过 :py:attr:`.TracedModule.graph` 属性访问 Graph，只能通过对应 Moldule 的 ``CallMethod`` Expr 访问，如下面的例子。

    .. code:: python

        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm

        class Mod(M.Module):
            def forward(self, x, b):
                x  = x + b
                return x

        class Net(M.Module):
            def __init__(self, ):
                super().__init__()
                self.mod = Mod()

            def forward(self, x):
                x = self.mod(x, 1)
                x = self.mod(x, 2)
                return x

        net = Net()
        inp = F.zeros(shape=(1, ))

        traced_net = tm.trace_module(net, inp)

        print(traced_net.graph)
        '''
        Net.Graph (self, x) {
                %5:     mod = getattr(self, "mod") -> (Module)
                %6:     mod_out = mod(x, 1, )
                %10:    mod_1 = getattr(self, "mod") -> (Module)
                %11:    mod_1_out = mod_1(mod_out, 2, )
                return mod_1_out
        }
        '''
        # 此时 traced_net.mod 将会被 trace 出 2 个 graph，因此无法直接访问 graph 属性
        try:
            print(traced_net.mod.graph)
        except:
            print("error")

        # 可通过 mod 的 CallMethod Expr 访问对应的 Graph
        print(traced_net.graph.get_expr_by_id(6).as_unique().graph)
        '''
        mod.Graph (self, x) {
                %9:     add_out = x.__add__(1, )
                return add_out
        }
        '''
        print(traced_net.graph.get_expr_by_id(11).as_unique().graph)
        '''
        mod_1.Graph (self, x) {
                %14:    add_out = x.__add__(2, )
                return add_out
        }
        '''
