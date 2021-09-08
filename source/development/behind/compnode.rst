.. _compnode:

========
CompNode
========

原理解释
--------

CompNode 是 Computing Node 的缩写，是 MegEngine 对计算设备的抽象。MegEngine 在 CompNode 上抽象了两种语义：

* 计算：一个 Op 在该设备上进行了计算操作
* 存储：一个 Tensor 存储在该设备分配的内存中

CompNode 属性往往可以通过输入 Tensor 确定，y = F(x) 即可让 MegEngine  推导出 F 这个 Op 以及 y 这个 Tensor 的 CompNode 与 x 是一致的。在多输入 CompNode 不同或没有 Tensor 做为输入的情况下，需要手工指定一个 Op 的 CompNode，例如 ``concat(x_on_cn1, x_on_cn2)`` 或 ``zeros(10)`` 。相关的代码定义在： :src:`src/core/include/megbrain/comp_node.h` - ``mgb::CompNode``.

基础用法
--------

CompNode 的表达方式为三段，分别是为 Device Type、Device Number 和 Stream ID，格式为 ``Device Type[Device Number][:Stream ID]`` 。

Device Type 包含 CPU、GPU、ROCm 等（GPU 实际上特指 CUDA GPU，这是一个历史遗留问题）。Stream 在不同的设备上会映射到不同的概念上，例如在 CPU 上代表 thread，在 GPU 上代表 cuda stream。多个拥有相同 Stream ID 的任务一定是串行执行的，否则将有可能并行执行（具体取决于该设备的并行能力）。因此我们可以通过配置 Stream ID 来控制两个 Op 或网络是否允许并行执行。

Device Number 和 Stream ID 可以不填写，默认为 0。以下是部分使用场景及其释义。

cpu
  第 1 个 CPU，thread1

cpu0
  第 1 个 CPU，thread1

cpu0:0
  第 1 个 CPU，thread1

gpu1:3
  第 2 个 GPU，Stream3

特殊用法
--------

除了基础用法，MegEngine 中含允许一些特殊用法。

xpu
  xpu 表示当前系统中最快的计算设备

cpu:default
  一般 CPU CompNode 中有两个线程，一个用来发射计算任务，另外一个则执行计算任务。cpu:default 创建的 CompNode 中只有一个线程，该线程直接 inplace 执行计算任务。

multithread2:0
  其中 multithread 表示创建多线程的 CompNode，2 代表线程数量，0 代表有 2 个线程的 CompNode 的索引下标

multithread:default:2
  default 代表 inplace 执行计算任务，2 代表线程数量

逻辑设备与物理设备
-------------------

MegEngine 引入了逻辑设备（logical device) 和物理设备（physical device）的概念，并在内部维护设备映射关系，即 logical device -> physical device。当你指定、创建一个 CompNode 时，传入的字符串就是 logical device，在实际运行前将会解析为 physical device。

在这个映射过程中，可能发生的事情有

* 按照设备性能为逻辑设备 xpu 指定物理设备，将会以 GPU > ROCm > CPU 的顺序选择
* 设备限制或用户指定不使用多线程时，将非 cpu:default 的逻辑设备都映射到物理设备 cpu0:1023
