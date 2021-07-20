.. _graph-runtime:

=============
Graph Runtime
=============

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

