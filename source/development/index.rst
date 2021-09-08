.. _development:

==========
开发者指南
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   architecture-design/index
   behind/index
   roadmap/index
   meps
   environment/index
   workflow
   how-to/index
   debugging-tools
   benchmark
   governance
   contribute-to-docs/index

.. admonition:: 开发者指南的定位
   :class: note

   * MegEngine 核心开发者的最频繁使用的信息查询手册；
   * 帮助 MegEngine 用户完成向核心开发者过渡的最佳材料；
   * 只要你是一名经验丰富的软件开发人员，欢迎你探索 MegEngine 世界。

.. warning::

   开发者指南文档版本与 MegEngine 源代码文档版本是对应的（我们仍需要几个版本来建设它）。

如何寻找想要的内容
------------------

这里的内容并不仅限于 MegEngine 开发者阅读，MegEngine 用户也可以将其作为茶余饭后的消遣。

:ref:`architecture-design`
  提供 MegEngine 架构设计的全局视角描述，目的是帮助开发者快速理清楚 MegEngine 源代码架构。
  但需注意，一些简明直观的示意图可以帮助开发者快速掌握要领，但也隐去了许多细节，因此可能产生误导。
  了解细节的更好做法是阅读《幕后揭秘》或《增强提案》，最好做法是借助这些材料阅读 MegEngine 源代码。

  架构设计像是一本地图或者是手册，适合为新手留下第一印象，也方便老手们回顾那些忽然生疏的概念。

:ref:`megengine-behind-the-scenes`
  通过一系列的文章为你介绍 MegEngine 背后的一些工作机制，适合作为 MegEngine 用户指南的进阶材料。
  文风更加倾向于技术博客，以解释清楚为目的，相较于架构设计文章，平均阅读用时更长，但细节更加丰富。
  开发者可按照默认顺序阅读，也可以挑选感兴趣的主题深入阅读，再决定是否要了解 MegEngine 源代码。
  这些材料能够大大减少开发者熟悉 MegEngine 源代码所需花费的时间，但注意使用对应版本的文档。

  幕后揭秘的职责不在于讲解源代码实现，更多地是讲解结构、流程，加深对 MegEngine 机制的理解。

:ref:`roadmap`
  顾名思义，如果你想了解 MegEngine 接下来一段时间的研发方向，请戳这里。

:ref:`meps`
  记录 MegEngine 开发过程中的设计与讨论，更多细节请参考 :ref:`mep-0001` 。

.. note::

   * 不同的材料中通常会进行相互引用，以满足人们对不同类型材料的阅读需求，也方便跳转；
   * 如果说用户指南能帮助一个人成为 Effective MegEnginer, 
     那么开发者指南的目的在于帮助整个社区了解 MegEngine Internals, 锻炼内功，拓展视野。

For developers
~~~~~~~~~~~~~~

看来你已经决定加入我们，成为 MegEngine 开发者的一员了。以下材料会有所帮助：

:ref:`dev-environment`
  拥有类似甚至完全一样的开发环境设置，会更加方便与其它开发者进行交流讨论。

:ref:`workflow`
  学习如何使用 Git 进行版本控制，以及了解 MegEngine 团队的 Git 协作流程。

:ref:`how-to-series`
  整理了一些高频的开发需求情景和解决方案，适合新人上手练习，或作为参考指南。

:ref:`debugging-tools`
  实现新功能很有乐趣，但也有可能引入新的 Bug，学会一些调试技巧会很有帮助。

:ref:`benchmark`
  东西好不好用，实现高不高效，测试了才知道。

帮助我们改进文档
----------------
*“滴水穿石非一日之功，冰冻三尺非一日之寒。”*

文档需要投入许多时间和精力维护，欢迎你加入文档建设，参考 :ref:`contribute-to-docs` 。
