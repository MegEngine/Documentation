.. _jit-guide:

===============
即时编译（JIT）
===============

我们在 :ref:`megengine-basics` 这篇教程中曾提到过这样一个概念：

*“在深度学习领域，任何复杂的深度神经网络模型本质上都可以用一个计算图表示出来。”*

.. figure:: ../../../_static/images/computing_graph.png

   MegEngine 中 Tensor 为数据节点, Operator 为计算节点

>>> y = w * x + b
>>> p = w * x
>>> y = p + b

在默认情况下，MegEngine 中的指令将像 Python 之类的解释型语言一样，动态地解释执行。
我们将这种执行模式称为 “动态图” 模式，此时完整的计算图信息其实并不存在；
而与之相对的是 “静态图” 模式，在执行之前能够拿到完整的计算图结构，
能够根据全局图信息进行一定的优化，加快执行速度。

在 MegEngine 中，通过使用即时编译技术（JIT），将动态图编译成静态图，并支持序列化。

.. toctree::
   :maxdepth: 1

   trace
   dump
   xla

接下来的内容将对相关概念和原理进行一定的解释，不了解这些细节并不影响基本使用。

.. _dynamic-and-static-graph:

动态图 Vs. 静态图
-----------------

.. panels::
   :container: +full-width
   :card:

   动态图
   ^^^^^^
   MegEngine 默认使用 **动态计算图** ，其核心特点是计算图的构建和计算同时发生（Define by run）。

   * **原理：** 在计算图中定义一个 :py:class:`~megengine.Tensor` 时，其值就已经被计算且确定了。
   * **优点：** 这种模式在调试模型时较为方便，能够实时得到中间结果的值。
   * **缺点：** 但是由于所有节点都需要被保存，这就导致我们难以对整个计算图进行优化。

   借助即时编译技术，MegEngine 中的动态图可通过 :class:`~.jit.trace` 接口转换成静态图。

   ---
   静态图
   ^^^^^^
   MegEngine 也支持 **静态计算图** 模式，将计算图的构建和实际计算分开（Define and run）。

   * **原理：** 在构建阶段，MegEngine 根据完整的计算流程对原始的计算图进行优化和调整，
     得到更省内存和计算量更少的计算图，这个过程称之为 “编译” 。编译之后图的结构不再改变，也就是所谓的 “静态” 。
     在计算阶段，MegEngine 根据输入数据执行编译好的计算图得到计算结果。
   * **优点：** 静态图相比起动态图，对全局的信息掌握更丰富，可做的优化也会更多。
   * **缺点：** 但中间过程对于用户来说是个黑盒，无法像动态图一样随时拿到中间计算结果。

什么是即时编译
--------------

即时编译（Just-in-time compilation）是源自编译（Compiling）中的概念。

以传统的 C/C++ 语言为例，我们写完代码之后，
一般会通过编译器编译生成可执行文件，然后再执行该可执行文件获得执行结果。
如果我们将从源代码编译生成可执行文件袋过称为 build 阶段，
将执行可执行文件叫做 runtime 阶段的话，JIT 是没有 build 阶段的，只存在于 runtime 阶段。
JIT 一般被用在解释执行的语言如 Python 中，JIT 会在代码执行的过程中检测热点函数（HotSpot），
随后对热点函数进行重编译，下次运行时遇到热点函数则直接执行编译结果即可。这样做可以显著加快代码执行的速度。

.. seealso::

   维基百科： `Just-in-time compilation 
   <https://en.wikipedia.org/wiki/Just-in-time_compilation>`_

.. _tracing-optim-example:

静态图编译优化举例
------------------

下面我们举例说明静态图编译过程中可能进行的内存和计算优化：

.. image:: ../../../_static/images/op_fuse.png
   :align: center

在上图左侧的计算图中，为了存储 ``x``, ``w``, ``p``,  ``b``, ``y`` 五个变量，
动态图需要 40 个字节（假设每个变量占用 8 字节的内存）。
在静态图中，由于我们只需要知道结果 ``y``, 可以让 ``y`` 复用中间变量 ``p`` 的内存，
实现 “原地”（Inplace）修改。这样，静态图所占用的内存就减少为 32 个字节。

.. seealso::

   更多相关解释可参考 MegEngine 官方博客 《 `JIT in MegEngine 
   <https://megengine.org.cn/blog/jit-in-megengine>`_ 》


