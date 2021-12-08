.. _user-guide:

========
用户指南
========
.. toctree::
   :hidden:
   :maxdepth: 1

   about/index
   install/index
   transfer-from/index
   faq/index

.. toctree::
   :caption: 模型开发（基础篇）
   :hidden:
   :maxdepth: 1

   model-development/tensor/index
   model-development/functional/index
   model-development/data/index
   model-development/module/index
   model-development/autodiff/index
   model-development/optimizer/index
   model-development/serialization/index
   model-development/hub/index

.. toctree::
   :caption: 模型开发（进阶篇）
   :hidden:
   :maxdepth: 1

   model-development/recomputation.rst
   model-development/distributed/index
   model-development/quantization/index
   model-development/amp/index
   model-development/profiler/index
   model-development/jit/index

.. toctree::
   :caption: 推理部署篇
   :hidden:
   :maxdepth: 1
   
   deployment/index
   deployment/traced_module/index
   deployment/lite/index
   deployment/lite_interface/index
   deployment/lite_advance/index

.. toctree::
   :caption: 工具与插件篇
   :hidden:
   :maxdepth: 1

   tools/stats
   tools/runtimeopr
   tools/load-and-run
   tools/load-and-run-py
   tools/customop

用户指南是一份较为通用的 MegEngine 操作手册，
其中的内容按照机器学习基础流程，结合用户的实际使用情景组织而成。
我们会根据社区的反馈，不断丰富指南内容，改进指南质量。

.. admonition:: 使用说明
   :class: warning

   * 初次接触 MegEngine 的用户应该从 :ref:`getting-started` 环节开始，
     我们假定用户指南的读者对 :ref:`deep-learning` 中提到的概念已经有所了解，进入框架日常使用阶段。
   * 对于有经验的用户，完全可以选择性地浏览自己感兴趣的主题，不同主题之间没有强烈的依赖关系。
     但我们也尽可能地将用户指南内容组织得有序，满足习惯顺序阅读用户的需求。

.. panels::
   :container: +full-width text-center
   :card:

   模型开发（基础篇）📕
   ^^^^^^^^^^^^^^^^^^^^
   基础篇将拓展延伸 :ref:`deep-learning` 中提到的 MegEngine 各个常见子包的使用情景。
   ---
   模型开发（进阶篇）📗
   ^^^^^^^^^^^^^^^^^^^^
   进阶篇将介绍更多特性，帮助你全面地掌握 MegEngine 的使用，发挥其全部实力。
   ---
   推理部署篇 📘
   ^^^^^^^^^^^^^
   介绍如何将训练好的模型部署到实际的生产环境，进行高效推理。
   ---
   工具与插件篇 📙
   ^^^^^^^^^^^^^^^
   了解 MegEngine 内提供各种工具与插件。

你可能正带着特定的问题来阅读用户指南 🤔 ，可以试着从下面找到解决方案 ⬇️  ⬇️  ⬇️

MegEngine Cookbook
------------------

.. dropdown:: :fa:`eye,mr-1` 如何在 MegEngine 上利用多卡进行分布式训练？

   请参考 :ref:`distributed-guide` 。

.. dropdown:: :fa:`eye,mr-1` 如何在 MegEngine 上使用量化技术？

   请参考 :ref:`quantization-guide` 。

.. dropdown:: :fa:`eye,mr-1` 如何在使用 MegEngine 训练模型时节省显存？

   请参考 :ref:`recomputation-guide` 。

.. dropdown:: :fa:`eye,mr-1` 如何在使用调试、优化、验证模型的性能？

   * 通过 Profiler, 我们可以对模型进行性能分析，请参考 :ref:`profiler-guide` ；
   * MegEngine 中提供了一些统计工具，请参考 :ref:`stats` ；
   * 如果你希望在不同的平台测试验证模型性能，请参考 :ref:`load-and-run` 。

.. dropdown:: :fa:`eye,mr-1` 如何对 MegEngine 的功能进行拓展？

   * 如果你希望为 MegEngine 添加新的算子，请参考 :ref:`add-an-operator` 。
   * 如果你希望参与到 MegEngine 的开发中来，请参考 :ref:`development`  

寻求更多支持
------------

对于一些非常见问题（即使用情景比较冷门时），其解决方案可能不会被整理到 MegEngine 用户指南中。
你可以通过 `MegEngine 官方论坛 <https://discuss.megengine.org.cn/>`_ 或者
`GitHub Issues <https://github.com/MegEngine/MegEngine/issues>`_ 进行交流讨论。
当某一类问题被提及的频率明显上升了，我们会开始考虑将其和对应的解决方案放入用户指南作为必要的参考。

虽然我们的社区生态中也提供了在线即时聊天渠道，但在必要的时候我们需要花些时间将讨论的过程和解决方案沉淀下来。
正所谓 “前人栽树，后人乘凉。” 为了让我们的交流和讨论能够为后来者提供参考信息，
请尽可能地使用能被互联网检索到的历史记录形式进行交流讨论。
我们相信通过这种方式，可以巩固你在 MegEngine 中进行实践所得到的经验，并放大它的价值。

我们也欢迎你通过自己的社交平台账号分享与 MegEngine 有关的故事，让更多人看见 MegEngine~
