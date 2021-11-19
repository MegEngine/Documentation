.. _tracemodule:
.. currentmodule:: megengine

=====================
TracedModule 发版方案
=====================

.. toctree::
   :hidden:
   :maxdepth: 1

   design
   api-example
   graphsurgeon-example

TracedModule 是什么
=====================

TracedModule 是基于 MegEngine 的发版方案。

TracedModule 由一个普通的 Module 通过 :py:func:`~.trace_module` 方法转换得到，仅由 MegEngine 的数据结构而构成，可脱离用户源代码被训练、序列化以及反序列化、以及图手术。

相比于普通的 Module, 它有如下特性:

* 拥有 graph 属性，graph 描述了 TracedModule 计算过程，可通过修改 graph 来改变计算过程（即 :py:meth:`.TracedModule.forward` 通过解析 :py:attr:`.TracedMoudle.graph` 执行）；
* 可序列化（pickle）到文件中，即使在没有模型源码的环境中也可被正确 load（普通 Module 的序列化文件脱离模型源码后无法 load 回来）。

TracedModule 发版方案的优点:

*  graph 基于 MegEngine 内置的 Module 和 function 构建，OP 粒度粗；
*  图手术直观，可直接查看 graph ，了解修改后的 graph 是否与预期一致；
*  基于动态图，调试方便，所见即所得。

TracedModule 发版流程
======================

.. image:: ../../../_static/images/tracemodule-flow.jpg
   :align: center

step1: 模型训练阶段

*  使用 MegEngine 来训练模型，模型基于 Module 构建。此时可直接将模型（Module） 转为 TracedModule 训练。
*  将模型（Module 或 TracedModule）转为 TracedModule，并通过 pickle 将模型序列化到文件中。

step2: 模型发版阶段

*  在无模型源码的环境中 pickle.load 第一步中序列化的 TracedModule 模型。
*  如果需要对模型进行图手术、后量化等操作，可直接对模型进行修改，此时模型依然在动态图模式下运行，所见即所得。
*  当真正确定最终要发版的模型后，可直接通过 :ref:`trace <trace>` & :ref:`dump <dump>` 转为 c++ 模型进行部署。转换为其他框架模型的转换工具正在开发中，敬请期待。
