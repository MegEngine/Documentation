.. _deployment:

======================
模型部署总览与流程建议
======================

当使用 MegEngine 完成模型的训练过程后，为了让模型可以实现它的价值，我们需要对模型进行“部署”，
即在特定的硬件设备和系统环境限制下，使用模型进行推理。

根据最终部署设备的不同，我们可能将会经历不同的部署路线：

.. list-table::
   :header-rows: 1


   * - 计算硬件
     - 举例
     - 适用场景

   * - 有 Python 环境的设备
     - GPU 服务器
     - 希望尽量简单，不在意 Python 性能上的限制
   * - C / C++ 环境的设备
     - 任何设备，尤其是嵌入式芯片、TEE 环境等
     - 希望性能尽量高、资源占用低，能接受编译 C++ 库的复杂性
   * - NPU
     - Atlas / RockChip / 寒武纪等芯片
     - 需要利用 NPU 的算力，接受稍复杂的转换步骤

在下面这张流程图中，可以了解到不同部署路线中的几个基本步骤：

.. mermaid::

   graph LR

   training_code[训练代码] ==> |tm.trace_module| tm_file[.tm 文件]
   training_code .-> |dump| mge_file
   tm_file ==> |dump| mge_file[.mge 文件]

   mge_file ==> |load| litepy[Lite Python 运行时]
   mge_file ==> |load| lite[Lite C++ 运行时]

   tm_file -- mge_convert --> otherformat[其他格式: ONNX/TFLite/Caffe] -- NPU 厂商转换器 --> NPU
   tm_file -- mge_convert 自带 NPU 转换器 --> NPU

.. note::

   为了更好的选择模型部署，需要了解到以下几点：

   * 最推荐的路线为训练代码 -> ``.tm`` 文件 -> ``.mge`` 文件 -> Lite 执行；
   * 如果你的团队中存在研究员 / 工程人员的分工，建议以 ``.tm`` 文件做为分界面 —— 
     研究员负责交付 ``.tm`` 模型（永久存档），工程人员负责后续的部署流程；
   * 如果你独立负责完整的训练到部署过程，且不在意长期存档模型。
     为了快捷，可以直接从训练代码生成 ``.mge`` 文件（即上述虚线），结果是等价的。

.. seealso::

   * :ref:`tracedmodule-quick-start`
   * :ref:`lite-quick-start-python`
   * :ref:`lite-quick-start-cpp`
   * `mgeconvert <https://github.com/megengine/mgeconvert>`_ : 适用于 MegEngine 的各种转换器。

