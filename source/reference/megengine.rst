.. py:module:: megengine
.. currentmodule:: megengine

=========
megengine
=========

.. py:module:: megengine.tensor
.. currentmodule:: megengine

.. code-block:: python3

   import megengine as mge  # NOT as torch - Even if we look like twins.

.. note::

   ``MGE`` 或 ``mge`` 是 ``MegEngine`` 的官方缩写，我们建议相关衍生库将其作为前缀。

.. warning::

   不要尝试 [ ``import megengine as torch`` ] 这是行不通的！ ( /ω＼)

   * MegEngine 的部分 API 设计借鉴了 PyTorch, 我们相信这对开发者而言会更加友好；
   * 但二者的底层设计完全不同，MegEngine is just MegEngine.

Tensor
------
.. code-block:: python3

   from megengine import Tensor
   from megengine import tensor  # tensor is a alias of Tensor

MegEngine 中提供了一种名为 “张量” （:class:`Tensor` ）的数据结构，
区别于物理学中的定义，其概念与 NumPy_ 中的 :class:`~numpy.ndarray` 更加相似，
即张量是一类多维数组，其中每个元素的数据类型和空间大小一致，仅数据值有所不同。

.. note::

   与 NumPy 的区别之处在于，MegEngine 还支持利用 GPU 设备进行更加高效计算。
   当 GPU 和 CPU 设备都可用时，MegEngine 将优先使用 GPU 作为默认计算设备，无需用户进行手动设定。
   
   * 如果有查看/改变默认计算设备的需求，请参考 :mod:`megengine.device` 中提供的接口。
   * 通过 :meth:`.Tensor.to` 和 :func:`.functional.copy` 可将 Tensor 拷贝到指定设备。

.. _Numpy: https://numpy.org
 
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Tensor
   Parameter



.. note::

   对于 Tensor 的各种操作，绝大部分都实现在 :mod:`megengine.functional` 中。

Tensor 数据类型
~~~~~~~~~~~~~~~

MegEngine 中使用 :class:`numpy.dtype` 来表示数据类型，目前支持的数据类型如下：

========================  ======================================================
数据类型                  numpy.dtype              
========================  ======================================================
32-bit floating-point     :any:`numpy.float32` a alias of :class:`numpy.single`
16-bit floating-point     :any:`numpy.float16` a alias of :class:`numpy.half`
8-bit integer (unsigned)  :any:`numpy.uint8`
8-bit integer (signed)    :any:`numpy.int8`
16-bit integer (signed)   :any:`numpy.int16`
32-bit integer (signed)   :any:`numpy.int32`
Boolean                   :any:`numpy.bool8` a alias of :class:`numpy.bool_`
========================  ======================================================

.. note::

   单精度浮点 ``float32`` 是 MegEngine 中默认的（也是最通用的）Tensor 数据类型。

.. warning::

   并不是所有的已有算子都支持上述任意数据类型的计算，还有一些情况尚未实现。


