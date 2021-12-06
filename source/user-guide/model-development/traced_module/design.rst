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

首先创建一个 ``SimpleModule`` 实例：

>>> module = SimpleModule()
>>> print(module)
SimpleModule(
  (linear): Linear(in_features=4, out_features=5, bias=True)
)

调用 :py:func:`~.trace_module` 将普通 Module 转换为 TracedModule:

>>> import megengine.traced_module as tm
>>> inp = F.zeros(shape = [3, 4])
>>> traced_module = tm.trace_module(module, inp)
>>> print(traced_module)
TracedModule(
  (linear): Linear(in_features=4, out_features=5, bias=True)
)

traced_module 拥有 :py:attr:`~.TracedModule.graph` 属性，graph 描述了 SimpleModule.forward 的执行逻辑：

>>> graph = traced_module.graph
>>> print(graph)    
SimpleModule.Graph (self, x) {
        %2:     const_tensor = Constant() -> (Tensor)
        %3:     add_out = x.__add__(const_tensor, )
        %4:     relu_out = nn.relu(add_out, )
        %5:     linear = getattr(self, "linear") -> (Linear)
        %6:     param = getattr(self, "param") -> (Tensor)
        %7:     add_out_1 = relu_out.__add__(param, )
        %8:     linear_out = linear(add_out_1, )
        return linear_out
}

我们可以看到 ``SimpleModule.Graph(self, x)`` 与 ``SimpleModule.forward(self, x)`` 的代码逻辑一致。

一个普通的 Module 可通过 :py:func:`~.trace_module` 方法将其转换为 TracedModule。
在转换过程中，用户自定义的 Module 将被转换为 TracedModule，内置 Module（如 :py:class:`~.module.Linear`, :py:class:`~.module.Conv2d` 等）不作转换。

graph 是 TracedModule 中最重要的属性，其实际是一个 :py:class:`~.traced_module.InternalGraph`, 
构成 InternalGraph 的基本单元为 :py:class:`~.traced_module.node.Node` 和 :py:class:`~.traced_module.expr.Expr`。

Node
----
**Node 的常用属性以及方法的使用例子请参考** :ref:`node-expr-method`。

通过 :py:class:`~.traced_module.node.Node` 来描述 ``forward`` 中的 :py:class:`~.Tensor` 或 :py:class:`~.Module`。

.. code-block:: python
    
    Class Node:
        expr : Expr # 描述了该 Node 由哪个 Expr 生成
        users : List[Expr] # 描述了该 Node 被哪些 Expr 使用
    
        @property
        def top_graph(self) -> InternalGraph: ... # 该 Node 所属的 InternalGraph
        @property
        def name(self) -> str: ... # 该 Node 的名字
        @property
        def qualname(self) -> str: ... # 生成该 Node 的 Module 的名字

Node 的 **expr** 属性记录了生成该 Node 的 Expr, 例如 ``SimpleModule`` 的输出是调用 ``linear`` 这个子 module 生成的。

>>> out_node = graph.outputs[0]
>>> print(out_node)
linear_out
>>> out_node.expr
%8:     linear_out = linear(add_out_1, )

Node 的 **users** 属性记录了该 Node 的被哪些 Expr 作为输入, 例如 ``SimpleModule`` 的中的输入 ``x`` 是 ``x = x + 1`` 这个的输入。

>>> inp_node = graph.inputs[1]
>>> print(inp_node)
x
>>> inp_node.users
[%3:    add_out = x.__add__(const_tensor, )]

Node 的 **name** 属性是该 Node 的名字，该名字在其所属的 graph 中是唯一的。

Node 的 **qualname** 属性记录了该 Node 是由哪个 Module 中所生成的，可以从 qualname 得到该 Module 的名字。
例如，``out_node.qualname`` 为 'SimpleModule.linear.[out]', 表示 ``out_node`` 是 SimpleModule 中 linear 这个子 module 的输出。

>>> out_node = graph.outputs[0]
>>> out_node.qualname
'SimpleModule.linear.[out]'

InternalGraph 中的 Node 有两种：

* :py:class:`~.TensorNode`：描述一个 Tensor，记录了该 Tensor 的 dtype 、shape 和 qparams 等信息

    >>> x = graph.inputs[1]
    >>> type(x)
    <class 'megengine.traced_module.node.TensorNode'>
    >>> x.shape
    (3, 4)
    >>> x.dtype
    numpy.float32

* :py:class:`~.ModuleNode`：描述一个 Module，记录了该 Module 的类型，以及对应的 Module

    >>> self = graph.inputs[0]
    >>> type(self)
    <class 'megengine.traced_module.node.ModuleNode'>
    >>> x.owner # 通过 owner 属性访问该 ModuleNode 所对应的 Module
    TracedModule(
    (linear): Linear(in_features=4, out_features=5, bias=True)
    )

Expr
-----
**Expr 的常用属性以及方法的使用例子请参考** :ref:`node-expr-method`。

通过 Expr 来描述一个 ``forward`` 中的某个表达式。
一个 Expr 由表达式的输入 ( :py:attr:`~.traced_module.Expr.inputs` )、
输出 ( :py:attr:`~.traced_module.Expr.outputs` )、
以及由输入到输出的执行逻辑 ( :py:meth:`~.traced_module.Expr.interpret` ) 构成。

.. code-block:: python

    Class Expr:
        inputs : List[Node] # 输入的 Node
        const_val : List[int,float,...] # 输入的常量
        outputs : List[Node] # 输出的 Node
    
        @property
        def top_graph(self) -> InternalGraph:... # 该 Expr 所属的 InternalGraph
    
        def interpret(self, *args, **kwargs):... # 根据输入执行该 expr

Expr 的子类分别有：

* :py:class:`~.Expr.GetAttr`: 获取 TracedModule 的中的某个属性，该 Expr 保存一个 name 字符串（用来描述要获取的属性），
  接受一个 ModuleNode 作为输入，它的执行逻辑为 ``outputs = getattr(inputs[0], name)``。
    
  例如：``SimpleModule.forward`` 中的 self.param 将会被解释为  ``%6: param = getattr(self, "param") -> (Tensor)``，
  self.linear 将会被解释为 ``%5: linear = getattr(self, "linear") -> (Linear)``，这两个 GetAttr 的输入均为 ``self`` 这个 ModuleNode。
    
  >>> exprs = graph.exprs(recursive=False).aslist()
  >>> exprs[6]
  %6:    param = getattr(self, "param") -> (Tensor)
  >>> exprs[6].inputs
  [self]
  >>> exprs[6].outputs
  [param]

* :py:class:`~.Expr.CallMethod`: 调用变量（Module 或 Tensor）的一个方法，该 Expr 保存一个 method 字符串（用来描述调用变量的哪个方法），
  接受多个输入（第一个输入为变量本身，即 self）。它的执行逻辑为 ``otuputs = getattr(inputs[0], method)(\*inputs[1:])``。

  例如：``SimpleModule.forward`` 中的 x = x + self.param  将会被解释为 ``%7: add_out_1 = relu_out.__add__(param, )``，
  这个 expr 是指调用了 x 的 ``__add__`` 方法，输入为 x 和 param。

  >>> exprs = graph.exprs(recursive=False).as_dict()
  >>> exprs[7]
  %7:     add_out_1 = relu_out.__add__(param, )
  >>> exprs[7].inputs
  [relu_out, param]
  >>> exprs[7].outputs
  [add_out_1]

* :py:class:`~.Expr.CallFunction`: 调用 megengine 内置的某个函数，该 Expr 保存一个 func，接受多个输入。
  它的执行逻辑为 ``outputs = func(\*inputs)`` 。

  例如：``SimpleModule.forward`` 中的 x = F.relu(x) ，将会被解释为 ``%4: relu_out = nn.relu(add_out, )``, 
  表示调用了 `nn.relu` 这个 function，其输入为 add_out。

  >>> exprs = graph.exprs(recursive=False).as_dict()
  >>> exprs[4]
  %4:    relu_out = nn.relu(add_out, )
  >>> exprs[4].inputs
  [add_out]
  >>> exprs[4].outputs
  [relu_out]

* :py:class:`~.Expr.Constant`: 产生一个常量，该 Expr 会记录一个不会改变的 value（Module 或 Tensor），不接受输入，它的执行逻辑为 ``outputs = value``。

  例如：``SimpleModule.forward`` 中的 ``mge.Tensor([1])`` 将会被解释为 ``%2: const_tensor = Constant() -> (Tensor)``， 表示一个生成常量 Tensor。

  >>> exprs = graph.exprs(recursive=False).as_dict()
  >>> exprs[2]
  %4:    relu_out = nn.relu(add_out, )
  >>> exprs[2].inputs
  []
  >>> exprs[2].outputs
  [const_tensor]

* :py:class:`~.Expr.Input`: 表示 Module.forward 的输入，仅仅是一个占位符的作用。真正推理的时候会将其替换为真正的 Tensor。

**所有的 Node 在实际执行推理的时候（interpret）都会被替换为实际的 Tensor 或者 Module。**

InternalGraph
-------------
**InternalGraph 的常用属性以及方法的使用例子请参考** :ref:`api-example`。

将 Module.foward 中的每一条语句都解释为由 Expr 组成的执行序列就构成了最终的 InternalGraph。

.. code-block:: python

    Class InternalGraph:
        _exprs : List[Expr]
    
        def interpret(self, *inputs):...

        @property
        def inputs(self):...

        @property
        def outputs(self):...

InternalGraph 包含以下三个属性：

* :py:attr:`~.InternalGraph._exprs`: 按执行顺序排列的 Expr 列表

    >>> graph._exprs
    [%2:    const_tensor = Constant(<class 'megengine.tensor.Tensor'>) -> (Tensor),
     %3:    add_out = x.__add__(const_tensor, ),
     %4:    relu_out = nn.relu(add_out, ),
     %5:    linear = getattr(self, "linear") -> (Linear),
     %6:    param = getattr(self, "param") -> (Tensor),
     %7:    add_out_1 = relu_out.__add__(param, ),
     %8:    linear_out = linear(add_out_1, )]

* :py:attr:`~.InternalGraph.inputs`: 该 graph 的输入 Node

    >>> graph.inputs
    [self, x]

* :py:attr:`~.InternalGraph.outputs`: 该 graph 的输出 Node

    >>> graph.outputs
    [linear_out]

在解析 Module.forward 的过程中，会将 forward 里的每一个执行语句描述为 Expr，并按执行次序依次添加到 _exprs 属性里。

在真正推理时，只需要遍历 _exprs 并依次 interpret 即可得到与执行原 Module 一样的结果。