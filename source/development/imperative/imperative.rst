.. _imperative:

======================
为什么需要 Imperative
======================

当我们谈论 MegEngine 时，我们在谈论什么
----------------------------------------

众所周知，\ ``MegEngine``
是由旷视自研、训练推理一体化、动静态合一的工业级深度学习框架。

由于历史原因，在各个地方可能还会看到这些名称：\ ``MegDL``\ 、\ ``MegBrain``\ 、\ ``MegSkull``\ 、\ ``MegHair``\ ，现在你可以认为它们都是
``MegEngine`` 的一部分。简单说，\ ``MegDL`` 包含了
``MegBrain``\ 、\ ``MegSkull`` 和 ``MegHair``\ ，\ ``MegDL`` 是
``MegEngine`` 的前身。

现在，当我们谈论 ``MegEngine``
时，通常包含三个层次：\ ``Imperative``\ 、\ ``MegBrain`` 和
``MegDNN``\ 。它们的角色定位分别是：

-  ``Imperative``\ ：负责处理动态图运行时（\ ``Imperative Runtime``\ ）
-  ``MegBrain``\ ：负责处理静态图运行时（\ ``Graph Runtime``\ ）
-  ``MegDNN``\ ：\ ``MegEngine`` 的底层计算引擎

MegDNN 与 MegBrain
------------------

``MegDNN`` 在 ``MegEngine``
中扮演的角色是\ **底层计算引擎**\ 。\ ``MegDNN``
是一个跨平台的底层算子库，训练和推理过程中的所有计算操作最终都需要落到一个
``MegDNN`` 的算子上进行，一个 ``MegDNN``
算子可能会根据场景（如张量尺寸等）有一或多个不同的实现（也叫
``kernel``\ ）。

作为一个跨平台的计算库，\ ``MegDNN`` 为我们提供丰富的与 ``Tensor``
相关的计算原语，比如
``Convolution``\ 、\ ``Pooling``\ 、\ ``MatrixMul``\ 、\ ``Transpose``
等。目前 ``MegDNN``
支持的平台有：\ ``x86``\ 、\ ``arm``\ 、\ ``CUDA``\ 、\ ``RoCM``\ 、\ ``OpenCL``\ 、\ ``Hexagon``
等。

.. mermaid::

    flowchart TD
    subgraph Python Interface
        mgeall[MegEngine autograd/ functional / Optimizer] --> pymge
        pymge[MegEngine Python / Imperative Runtime]
    end

    subgraph C++ Interface
        cpplite[MegEngine Lite]
        lar[Load And Run]
    end

    cg{Computing Graph}
    tensorinterpreter{Tensor Interpreter}
    cpplite -- "包装、简化" -->  cg
    lar --> cg
    pymge --> tensorinterpreter{Tensor Interpreter};
    tensorinterpreter -- "复用 Shape 推导、计算" --> cg;
    cg <-- "序列化/反序列化" --> serig[Serialization Manager];
    cg --> |compile| comps[Computing Sequence]
    algochooser[Algo Chooser]

    subgraph dnn
        shape_deduce[Shape 推导器]
        megdnn_kernel
    end

    ray[MegRay]
    cg --> shape_deduce
    tensorinterpreter --> shape_deduce
    comps --> algochooser -- "执行计算" --> megdnn_kernel[MegDNN kernel];
    tensorinterpreter -- "计算" --> algochooser

    tensorinterpreter -- "分配存储、同步" --> compnode
    comps -- "分配存储、同步" --> compnode[CompNode]
    comps --> ray;
    tensorinterpreter --> ray;

    subgraph Hardware
       x86
       Arm
       CUDA
       RoCM
    end

    subgraph NPU
       TensorCore
    end

    comps --> externCOpr[Extern C Opr / RuntimeOpr] --> NPU;
    compnode -- "抽象内存分配、同步机制" --> Hardware
      ray -- 抽象通信算子 --> Hardware
    megdnn_kernel -- "抽象计算" --> Hardware

为了确保训练推理一致性， ``Imperative`` 复用了 ``MegBrain``
的计算代码，因此我们需要了解 ``MegBrain`` 做了什么。

``MegBrain`` 负责处理静态图的运行时，主要提供 ``C++`` 的训练和推理接口。从下面的
``MegEngine`` 整体架构图可以看出，\ ``Imperative`` 通过
``Tensor Interpreter`` 复用了许多 ``MegBrain`` 的代码。比如 ``shape``
推导、计算、求导、\ ``Trace`` 等。

在 ``MegBrain`` 中，一个 ``Computing Graph`` 由 ``SymbolVar`` 以及许多
``op`` 组成。\ ``SymbolVar`` 是在 ``MegBrain`` 层面 ``Tensor``
的表示，可以理解为传递给 ``op`` 进行计算的数据。

因为 ``Computing Graph`` 中已经有非常多的算子的实现，因此
``Tensor Interpreter`` 中较多复用 ``MegBrain`` 的
``op``\ 。原因有：1）重写算子代价高，且容易写错；2）若 ``Imperative``
的实现和 ``MegBrain`` 的实现不一致的话，容易导致训练推理不一致。

那么，\ ``Imperative`` 的 ``Tensor Interpreter`` 如何复用 ``MegBrain``
的代码呢？对于动态训练来说，如果一个 ``op`` 没有在 ``Imperative`` 找到
``native`` 的实现，就会为这个 ``op``
建一张计算图，然后立即跑这张计算图。通过这种方式将动态图“转换”为静态图。

Imperative 登场
----------------

因为 ``MegEngine`` 是动静合一的深度学习框架，\ ``MegBrain``
解决了静态图的训练和推理问题，还需要有一个“组件”负责处理动态图的训练和推理，于是便有了
``Imperative``\ ，也就是说，\ ``Imperative Runtime``
是为了动态训练而单独设计的一套新接口。

上面提到 ``Imperative`` 复用了很多 ``MegBrain`` 的部分。\ ``Imperative``
自身包含的模块主要有：\ ``Module``\ 、\ ``Optimizer``\ 、\ ``Functional``\ 、\ ``Interpreter``\ 、\ ``DTR``\ 、\ ``Tracer``\ 等（之后会详细介绍）。

``Imperative`` 的设计基本原则包括：

1. 与 ``graph runtime``
   的计算行为尽可能复用相同的计算代码，确保训练推理一致性。
2. ``Pythonic``\ ：一切资源完全与 ``Python`` 对象深度绑定。

Imperative 和 MegDNN / MegBrain 的关系
--------------------------------------

简单来说，\ ``MegDNN`` 负责 ``MegEngine`` 中所有的计算相关的动作，无论是
``MegBrain`` 还是 ``Imperative`` 的 ``op``\ ，最终都需要通过调用
``MegDNN kernel`` 来完成计算。

既然 ``MegDNN``
包揽了计算的活儿，那么在训练推理过程中那些与计算无关的工作，自然就落到了
``MegBrain`` 和 ``Imperative`` 的头上。这些工作包括：求导、内存分配、对
``Tensor`` 的 ``shape`` 进行推导、图优化、编译等。

``MegEngine`` 整体上是有两部分 ``Runtime``
以及底层的一些公共组件组成的。这两部分的 ``Runtime`` 分别叫做
``Graph Runtime``\ （对应 ``MegBrain``\ ） 和 ``Imperative Runtime``\ 。

``Graph Runtime`` 负责静态图部分，主要提供 ``C++`` 推理接口。

``Imperative Runtime`` 负责动态图部分，主要为动态训练提供 ``Python``
接口。
