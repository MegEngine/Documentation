.. _tensor-dtype:

Tensor 数据类型
---------------

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

要获取指定数据类型的 Tensor, 可以使用 :meth:`~.Tensor.astype` 方法进行转化。

.. note::

   单精度浮点 ``float32`` 是 MegEngine 中默认的（也是最通用的）Tensor 数据类型。

.. note::

   我们会在 :mod:`megengine.quantization` 模块中提到对量化数据类型的支持。

.. warning::

   并不是所有的已有算子都支持上述任意数据类型的计算，还有一些情况尚未实现。

