.. py:module:: megengine.tensor
.. currentmodule:: megengine

==============
张量（Tensor）
==============

.. autosummary::
   :toctree: api
   :nosignatures:

   tensor.Tensor
   tensor.tensor
   tensor.Parameter 

张量的类型与形状
================
.. autosummary::
   :toctree: api
   :nosignatures:

   Tensor.dtype
   Tensor.astype
   Tensor.shape
   Tensor.reshape
   Tensor.ndim
   Tensor.size
   Tensor.flatten
   Tensor.transpose
   Tensor.T

转为其它数据类型
================
.. autosummary::
   :toctree: api
   :nosignatures:

   Tensor.numpy
   Tensor.tolist
   Tensor.item

归约计算（Reduction）
=====================
.. autosummary::
   :toctree: api
   :nosignatures:

   Tensor.sum
   Tensor.mean
   Tensor.prod
   Tensor.min
   Tensor.max

求导时视作常量
==============
.. autosummary::
   :toctree: api
   :nosignatures:

   Tensor.detach

查询/改变所在设备
=================
.. autosummary::
   :toctree: api
   :nosignatures:

   Tensor.device
   Tensor.to

\*对张量进行计算
================

.. seealso::
   
   - 创建张量的函数例如 :func:`~.ones`, :func:`~.arange` ，可在 :py:mod:`.functional.tensor` 中找到。
   - 处理张量的函数例如 :func:`~.transpose`, :func:`~.reshape` ，可在 :py:mod:`.functional.tensor` 中找到。

