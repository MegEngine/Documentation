.. _load-and-run:

使用 MegEngine Load and run 进行模型推理
=====================================================

.. note::

   ``Load and run`` （简称 LAR ） 是 MegEngine 提供与之相关的配套的推理工具，主要提供了以下功能：

   * 测试各类模型的推理性能，获取模型推理结果以及推理相关信息。
   * 测试验证不同模型优化方法的效果（load and run 有很多丰富的配置选项,可以执行 ``./load_and_run --help`` 查看详细的帮助文档）。


工具提供了大量的配置选项，这些选项可以灵活的适配各种推理所需的场景，能够在不同设备上运行 MegEngine 模型，给出推理结果以及相应的性能分析结果，是基于 MegEngine 的模型研究和单模型的推理部署的很重要的生产工具。

有关 Load and run 工具的各种使用细节主要包括以下几个部分：

.. toctree::
   :maxdepth: 1

   build-lar

   lar-model-support

   lar-basic-usage

   lar-options-list

   lar-inference-optimize

   lar-profile-model

   lar-accuracy-analysis

   lar-debug

   load-and-run-py


.. note::

   * 目前发布的版本我们开放了对 CPU（x86, x64, ARM, ARMv8.2）和 GPU（CUDA）平台的支持。
   * 为了方便使用，Megengine 提供了 load and run :ref:`Python 版本 <load-and-run-py>` 。


