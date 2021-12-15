.. _traced_module-guide:
.. currentmodule:: megengine

===============================
使用 TracedModule 发版
===============================

TracedModule 简介
=====================

**TracedModule 本质是一个 Module**，
它由一个普通的 Module 通过 :py:func:`~.trace_module` 方法转换得到，
仅依赖于 MegEngine 的数据结构，可脱离用户源代码被训练、序列化、反序列化以及图手术。

TracedMdoule 与普通 Module 的区别在于它有如下特性:

* 拥有 graph 属性，graph 描述了 TracedModule 计算过程，可通过修改 graph 来改变计算过程
* 可序列化到文件中，即使在没有模型源码的环境中也可被正确 load（普通 :py:class:`~.Module` 的序列化文件脱离模型源码后无法 load 回来）

TracedModule 模型开发流程
=========================

.. image:: ../../../_static/images/tracemodule-flow.jpg
   :align: center

1. 模型训练阶段

   * 使用 MegEngine 来训练模型，模型基于 Module 构建（可直接将模型转为 TracedModule 训练）
   * 将模型（Module 或 TracedModule）转为 TracedModule，并通过 pickle 将模型序列化到文件中

2. 模型发版阶段

   * 在脱离模型源码的环境中 load 序列化后的 TracedModule
   * 如果需要对模型进行图手术、量化等操作，可直接对模型进行修改，此时模型依然在动态图模式下运行，所见即所得
   * 当真正确定最终要发版的模型后，可直接通过 :ref:`trace <trace>` & :ref:`dump <dump>` 转为 c++ 模型在第一方部署，
     或基于 `mgeconvert <https://github.com/MegEngine/mgeconvert>`__ 工具转换为其它框架的模型用于部署


文档目录
========

.. toctree::
   :maxdepth: 3

   quick-start
   design
   api-example
   graphsurgeon-example
