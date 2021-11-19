.. _design:

=====================
TracedModule 基本概念
=====================

TracedModule 来源于普通的 Module，但它与普通 Module 不同的是其 :py:meth:`.TracedModule.forward` 方法的执行逻辑通过 :py:class:`~.InternalGraph` 来描述。

下面的例子展示了 Module、TracedModule 以及 InternalGraph 之间的关系。

.. code-block:: python

    import megengine.module as M
    import megengine.functional as F
    import megengine as mge
    
    class SimpleModule(M.Module):
        def __init__(self):
            super().__init__()
            self.linear = M.Linear(4, 5)
            self.param = mge.Parameter([1])
    
        def forward(self, x):
            x = x + mge.Tensor([1])
            x = F.relu(x)
            return self.linear(x + self.param)
    
    module = SimpleModule()
    print(module)
    """
    SimpleModule(
    (linear): Linear(in_features=4, out_features=5, bias=True)
    )
    """

    import megengine.traced_module as tm
    inp = F.zeros(shape = [3, 4])
    
    # traced_module : TracedModule
    traced_module = tm.trace_module(module, inp)
    print(traced_module)
    """
    TracedModule(
    (linear): Linear(in_features=4, out_features=5, bias=True)
    )
    """
    
    # graph 描述了 SimpleModule.forward 的执行逻辑，TracedModule.forward 通过解析 graph 执行
    graph = traced_module.graph
    print(graph)
    """
    SimpleModule.Graph (self, x) {
            %5:     const_tensor = Constant(<class 'megengine.tensor.Tensor'>) -> (Tensor)
            %6:     add_out = x.__add__(const_tensor, )
            %7:     relu_out = nn.relu(add_out, )
            %8:     linear = getattr(self, "linear") -> (Linear)
            %9:     param = getattr(self, "param") -> (Tensor)
            %10:    add_out_1 = relu_out.__add__(param, )
            %11:    linear_out = linear(add_out_1, )
            return linear_out
    }
    """

一个普通的 Module 可通过 :py:func:`~.trace_module` 方法将其转换为 TracedModule。
在转换过程中，用户自定义的 Module 将被转换为 TracedModule，内置 Module（如 :py:class:`~.module.Linear`, :py:class:`~.module.Conv2d` 等）不作转换。
转换后的模型仅由 MegEngine 的数据结构构成，可脱离源代码被序列化以及反序列化。

构成 InternalGraph 的基本单元为 :py:class:`~.traced_module.node.Node` 和 :py:class:`~.traced_module.expr.Expr`。

Node
----
通过 Node 描述 一个 :py:class:`~.Tensor` 或 :py:class:`~.Module`。

.. code-block:: python
    
    Class Node:
        expr : Expr # 描述了该 Node 由哪个 Expr 生成
        users : List[Expr] # 描述了该 Node 被哪些 Expr 使用
    
        @property
        def top_graph(self) -> InternalGraph: # 该 Node 所属的 InternalGraph
            ...

Node 的 expr 属性记录了生成该 Node 的 Expr，users 属性记录了将该 Node 作为输入的 Expr。

.. code-block:: python

    graph = traced_module.graph
    """
    SimpleModule.Graph (self, x) {
            %5:     const_tensor = Constant(<class 'megengine.tensor.Tensor'>) -> (Tensor)
            %6:     add_out = x.__add__(const_tensor, )
            %7:     relu_out = nn.relu(add_out, )
            %8:     linear = getattr(self, "linear") -> (Linear)
            %9:     param = getattr(self, "param") -> (Tensor)
            %10:    add_out_1 = relu_out.__add__(param, )
            %11:    linear_out = linear(add_out_1, )
            return linear_out
    }
    """
    linear_out = graph.outputs[0] # InternalGraph have inputs and outputs
    self_node = graph.inputs[0]
    print(linear_out)
    print(linear_out.expr)
    """
    linear_out
    %8:     linear_out = linear(add_out_1, )
    """
    print(self_node)
    print(self_node.users)
    """
    self
    [%5:    linear = getattr(self, "linear") -> (Linear),
    %6:       param = getattr(self, "param") -> (Tensor)]
    """

InternalGraph 中的 Node 有两种：

* :py:class:`~.TensorNode`：描述一个 Tensor，记录了该 Tensor 的 dtype 、shape 和 qparams 等信息；
* :py:class:`~.ModuleNode`：描述一个 Module，记录了该 Module 的类型，以及对应的 Module。

.. code-block:: python

    print("node: {}, type: {}".format(linear_out, type(linear_out)))
    print("shape : {}, dtype : {}".format(linear_out.shape, linear_out.dtype))
    """
    node: linear_out, type: <class 'megengine.traced_module.node.TensorNode'>
    shape : (3, 5), dtype : <class 'numpy.float32'>
    """
    print("node: {}, type: {}".format(self_node, type(self_node)))
    """
    node: self, type: <class 'megengine.traced_module.node.ModuleNode'>
    """
    # ModuleNode 可以通过直接访问 owner 属性获取该 ModuleNode 所对应的 Module
    print(self_node.owner)
    """
    TracedModule(
    (linear): Linear(in_features=4, out_features=5, bias=True)
    )
    """

Expr
-----
通过 Expr 来描述一个 Module.forward 中的某个表达式。
一个 Expr 由表达式的输入 ( :py:attr:`~.Expr.inputs` )、输出 ( :py:attr:`~.Expr.outputs` )、以及由输入到输出的执行逻辑 ( :py:meth:`~.Expr.interpret` ) 构成。

.. code-block:: python

    Class Expr:
        inputs : List[Node] # 输入的 Node
        const_val : List[int,float,...] # 输入的常量
        outputs : List[Node] # 输出的 Node
    
        @property
        def top_graph(self) -> InternalGraph: # 该 Expr 所属的 InternalGraph
            ...
    
        def interpret(self, *args, **kwargs): # 执行逻辑
            ...

Expr 的子类分别有：

* :py:class:`~.Expr.GetAttr`: 获取 TracedModule 的中的某个属性，该 Expr 保存一个 name 字符串（用来描述要获取的属性），接受一个输入（一般为一个 ModuleNode），它的执行逻辑为 outputs = getattr(inputs[0], self.name)。
    
    例如：SimpleModule.forward 中的 self.param 将会被解释为 "%7: param= getattr(self, "param") -> (Tensor)"，self.linear 将会被解释为 ”%7: linear = getattr(self, "linear") -> (Linear)“，这两个 GetAttr 的输入均为 self 这个 ModuleNode。

    .. code-block:: python

        exprs = graph.exprs(recursive=False).as_dict()
        print(exprs[9])
        print("inputs: {}, outputs: {}".format(exprs[9].inputs, exprs[9].outputs))
        """
        %9:     param = getattr(self, "param") -> (Tensor)
        inputs: [self], outputs: [param]
        """

* :py:class:`~.Expr.CallMethod`: 调用变量（Module，Tensor 等）的一个方法，该 Expr 保存一个 method 字符串（用来描述调用变量的哪个方法），接受多个输入（第一个输入为变量本身，即 self）。
  它的执行逻辑为 otuputs = getattr(inputs[0], selfmethod)(\*inputs[1:]) 。

    例如：SimpleModule.forward 中的 x = x + self.param  将会被解释为 "%9: add_out_1 = relu_out.__add__(param, )"，这个 expr 是指调用了 x 的 "__add__" 方法，输入为 x 和 self.param。

    .. code-block:: python

        exprs = graph.exprs(recursive=False).as_dict()
        print(exprs[10])
        print("inputs: {}, outputs: {}".format(exprs[10].inputs, exprs[10].outputs))
        """
        %10:    add_out_1 = relu_out.__add__(param, )
        inputs: [relu_out, param], outputs: [add_out_1]
        """

* :py:class:`~.Expr.CallFunction`: 调用 megengine 内置的某个函数，该 Expr 保存一个 func (可调用的函数)，接受多个输入。它的执行逻辑为 outputs = self.func(\*inputs) 。

    例如：SimpleModule.forward 中的 x = F.relu(x) ，将会被解释为 relu_out = nn.relu(add_out, ), 代表调用了 nn.relu 这个 function，其输入为 add_out。

    .. code-block:: python

        exprs = graph.exprs(recursive=False).as_dict()
        print(exprs[7])
        print("inputs: {}, outputs: {}".format(exprs[7].inputs, exprs[7].outputs))
        """
        %7:     relu_out = nn.relu(add_out, )
        inputs: [add_out], outputs: [relu_out]
        """

* :py:class:`~.Expr.Constant`: 产生一个常量，该 Expr 会记录一个不会改变的参数（int, float, Module, Tensor 等），不接受输入，它的执行逻辑为 outputs = self.value。

    例如：SimpleModule.forward 中的 mge.Tensor([1]) 将会被解释为 ”%5: const_tensor = Constant(<class 'megengine.tensor.Tensor'>) -> (Tensor)“，表示一个生成固定 Tensor 的 Expr。

    .. code-block:: python

        exprs = graph.exprs(recursive=False).as_dict()
        print(exprs[5])
        print("inputs: {}, outputs: {}".format(exprs[5].inputs, exprs[5].outputs))
        """
        %5:     const_tensor = Constant(<class 'megengine.tensor.Tensor'>) -> (Tensor)
        inputs: [], outputs: [const_tensor]
        """

* :py:class:`~.Expr.Input`: 表示 Module.forward 的输入，仅仅是一个占位符的作用。真正推理的时候会将其替换为真正的 Tensor。

**所有的 Node 在实际执行推理的时候（interpret）都会被替换为实际的 Tensor 或者 Module。**

InternalGraph
-------------
将 Module.foward 中的每一条语句都解释为由 Node 和 Expr 组成的执行序列就构成了最终的 InternalGraph。

.. code-block:: python

    Class InternalGraph:
        _exprs : List[Expr]
        _inputs : List[Node]
        _outputs : List[Node]
    
        def interpret(self, *inputs):
            ...

InternalGraph 包含以下三个属性：

* :py:attr:`~.InternalGraph._exprs`: 按执行顺序排列的 Expr 列表
* :py:attr:`~.InternalGraph._inputs`: 该 graph 的输入 Node
* :py:attr:`~.InternalGraph._outputs`: 该 graph 的输出 Node

在解析 Module.forward 的过程中，会将 forward 里的每一个执行语句描述为 Expr，并按执行次序依次添加到 _exprs 属性里。在真正推理时，只需要遍历 _exprs 并依次 interpret 即可得到与执行原 Module 的 foward 一样的结果。

执行方式如下：保存一个 {Node: Tensor/Module} 的字典，这样每个 Expr 都可以通过自己的 inputs 记录的 Node 找到推理时真正想要的 Tensor/Module。

.. code-block:: python

    def interpret(self, *inputs):
        node2value = {}
        for n, v in zip(self._inputs, inputs):
            node2value[n] = v
        for expr in self._exprs: # 按顺序遍历 _epxrs 并执行
            values = expr.interpret(*list(node2value[i] for i in expr.inputs))
            if values is not None:
                for n, v in zip(expr.outputs, values):
                    node2value[n] = v
        return list(node2value[i] for i in self._outputs)
