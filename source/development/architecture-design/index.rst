.. _architecture-design:

==================
MegEngine 架构设计
==================

.. toctree::
   :hidden:
   :maxdepth: 1

   graph-runtime
   imperative-runtime
   megdnn

MegEngine 整体由两部分 Runtime 加上底层的公共组件组成：

* 其中静态图部分（又称 :ref:`graph-runtime` ）主要提供 C++ 推理接口；
* 动态图部分（又称 :ref:`imperative-runtime` ）主要提供 Python 接口供动态训练使用；

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
