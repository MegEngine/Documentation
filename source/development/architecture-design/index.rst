.. _architecture-design:

==================
MegEngine 架构设计
==================

.. toctree::
   :hidden:
   :maxdepth: 1

   megdnn
   graph-runtime
   imperative-runtime

------------------
MegEngine 整体架构
------------------

MegEngine 整体由两部分 runtime 加上底层的公共组件组成，其中静态图部分（又称 graph runtime）主要提供 C++ 推理接口；动态图部分（又称 imperative runtime）主要提供 Python 接口供动态训练使用。

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
       TRT
        Cambricon
        DSA
    end

    comps --> externCOpr[Extern C Opr / RuntimeOpr] --> NPU;
    compnode -- "抽象内存分配、同步机制" --> Hardware
      ray -- 抽象通信算子 --> Hardware
    megdnn_kernel -- "抽象计算" --> Hardware


* 接口层

    * MegEngine Imperative Runtime: 动态解释执行接口
    * MegEngine Lite: C++ 静态图推理接口
    * Load and run: 一份用于调试性能的工具（可看做一种推理代码的样例）

* 核心模块层
    * Computing Graph: 一张以 OpNode 和 VarNode 依次相连的 DAG，用以表达全部计算依赖，是静态图的模式的核心。模块内部包含了图优化、静态推导、自动求导的各类功能。通过 compile 可以产生 Computing Sequence 以用于实际执行
    * Computing Sequence: 一个带有依赖关系的执行序列，是 Computing Graph 的一种拓扑排序结果，其中包含了内存分配策略等资源信息，可以通过 execute 执行其中的全部 Op
    * Tensor Interpreter: Tensor 解释器，用于解释执行动态模式下的计算操作；其中部分操作是通过构建一张临时的 Computing Graph 来复用原有操作，另一部分通过直接调用底层实现（以获得更高的性能）

* 工具模块
     * Shape 推导器: 用于静态推导 shape
     * Algo Chooser: 同一 Op 不同 kernel 的选择器，用以挑选在当前参数下最快的 kernel，是 Fastrun 机制的核心
     * Serialization Manager: 对 Computing Graph 进行序列化 / 反序列化，提供无限向后兼容性 (backward compatible)

* 硬件抽象层（HAL）

    * MegDNN kernel: 包含各类平台下的计算算子实现（部分简单算子直接在 megengine src 目录下实现，未包含在 dnn 中）
    * Extern C Opr / Runtime Opr: 用于包装 DSA / TRT 等子图，对上层抽象为一个 Op
    * CompNode: 对硬件的基本操作进行抽象，包括 执行计算、同步机制、内存分配、跨设备拷贝 等原语。一个 CompNode 对应一个 GPU stream 或 CPU 线程，部分硬件上实现了内存池以进一步提高性能
    * MegRay: 对训练场景下的集合通讯、点对点通信进行了设备无关的抽象，底层对应了 nccl / rccl / ucx / 自研方案 等不同实现

* 硬件层

------------------
Graph Runtime 详解
------------------

.. mermaid::

    flowchart TB
    cg{Computing Graph}
    serig[Serialization Manager] <-- "序列化/反序列化 .mge 文件" -->  cg

    cg --> |求导| autograd[静态求导器]
    autograd --> |插入对应反向算子| cg

    subgraph compile [Compile 阶段]
        cg <--"生成静态内存分配策略" --> memplaner[静态内存分配器]
        cg <-- "静态推导 Shape" --> static_inference[静态推导器]
        cg <-- "图优化" --> gopt[Graph Optimizer]
    end
    compile -->|compile 返回| comps[Computing Sequence]

Computing Graph 的主要使用流程

* 静态求导器: 基于链式法则，对给定的目标进行求导，并将求导后的反向算子重新插入到原有的 Computing Graph 中
* 静态内存分配器：
* 静态推导器 (Static Infernece Manager)：基于网络的静态信息，对 Shape 和部分网络中的 value 进行静态推导，其中部分推导可以允许失败（因为可能需要运行时信息）
* compile 操作：经过上述


-----------------------
Imperative Runtime 架构
-----------------------

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
