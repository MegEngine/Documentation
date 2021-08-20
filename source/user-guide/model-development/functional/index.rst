.. _functional-guide:
.. currentmodule:: megengine.functional

==========================
使用 Functional 操作与计算 
==========================

.. note::

   * 除了 :ref:`Tensor <tensor-guide>` 内置的方法，我们还可以调用 Functional 包中实现的有关操作与计算接口。
   * 想要快速浏览或检索所有相关 API 和例程，请参考 :py:mod:`megengine.functional` 页面。

.. warning::

   大部分时候时候，我们会将对 Tensor 的操作和运算区分为两个概念：

   * 对单个 Tensor 进行操作（Manipulation），通常其属性（如形状、设备、类型等）发生了变化；
   * Tensor 之间可以进行运算（Operation），通常基于 Tensor 内元素的数值进行了计算。

   而有些时候它们概念一致，彼此之间可以互相指代（取决于上下文）。
   但可以确定的一点是，不论如何称呼这些行为，它们都可以从 :py:mod:`~.functional` 包中能找到。

区别于 API 参考中的分类方式，本文档中将 Tensor 的常见操作/运算分为以下类型进行介绍：

.. toctree::
   :maxdepth: 2 

   creation
   manipulation
   general-operations
