.. _imperative-runtime:

==================
Imperative Runtime
==================

.. mermaid::

    flowchart TD

    tensor[Tensor Wrapper]
    tensor -- "创建、计算、删除" --> TI[Tensor Interpreter];
    tensor -- 记录求导关系 --> autograd
    tensor -- Trace --> tracer

    gm[Grad Manager] -- "创建 / backward" --> autograd
    autograd --> proxygraph
    autograd --> tensor
    functional -- apply --> tensor
    用户 -- "new / .numpy / del / _reset"  --> tensor

    TI -- "计算、Shape推导" --> proxygraph
    TI --> CompNode+kernel
    TI <--> DTR

    module --> functional;
    optimizer;
    quantization

Imperative Runtime 是为了动态训练单独设计的一套新接口，其设计基本原则包含：

1. 与 graph runtime 的计算行为尽可能复用相同的计算代码，确保训推一致性
2. Pythonic 一切资源完全与 python 对象深度绑定

各类模块：

* module / optimizer 等：Python 模块
* functional: 各类计算函数，底层基本是直接调用 `apply(OpDef, args)`
* Tensor Wrapper: C++ 模块，从 Python 层可以直接看到的 tensor 类型，提供计算、自动微分、trace 等功能
* Tensor Interpreter:

    * 一切计算的入口，提供 `put tensor`, `apply(OpDef, tensor)`, `get tensor` 三大类功能
    * 所有计算操作均为异步，因此除可被外界观测到的 `put` 和 `get` 外，其他操作均可被透明的调整顺序或优化
    * 底层计算部分直接调用 kernel，部分通过 proxygraph 调用 graph runtime 实现

* DTR: 动态重计算模块，负责 Tensor Interpreter 的 drop 指令，确保记录计算过程，确保被 drop 掉的 tensor 在被需要时重新计算得到
* autograd: 自动微分机制，负责记录 Tensor Wrapper 的计算过程并通过 refcount 确保依赖的 tensor 不被释放
* tracer: 在 trace 模式下记录全部的计算过程，从而生成静态图
* proxygraph: 一系列桥接机制的统称，通过建立临时的计算图实现复用 graph runtime 中的计算、shape 推导的能力；其中的 graph 与用户实际计算无关，可随时清空。

