.. _tensor-dtype:

===============
Tensor 数据类型
===============

:attr:`~.Tensor.dtype` 是 Tensor 的一种基础属性，表示数据类型，单个 Tensor 内的元素数据类型完全一致。

已经支持的类型
--------------

MegEngine 中使用 :class:`numpy.dtype` 来表示数据类型，目前支持的数据类型如下：

.. list-table::
   :header-rows: 1

   * - 数据类型
     - numpy.dtype
     - 等效字符串
   * - 32-bit floating-point
     - :any:`numpy.float32` / :class:`numpy.single`  
     - float32
   * - 16-bit floating-point     
     - :any:`numpy.float16` / :class:`numpy.half`
     - float16
   * - 8-bit integer (unsigned) 
     - :any:`numpy.uint8`                                     
     - uint8
   * - 8-bit integer (signed)    
     - :any:`numpy.int8`                                      
     - int8
   * - 16-bit integer (signed)   
     - :any:`numpy.int16`                                     
     - int16
   * - 32-bit integer (signed)   
     - :any:`numpy.int32`                                     
     - int32
   * - Boolean                   
     - :any:`numpy.bool8` / :class:`numpy.bool_`     
     - bool / bool\_ / bool8

.. warning::

   并不是所有的已有算子都支持上述任意数据类型之间的计算，还有一些情况尚未实现。

.. note::

   单精度浮点 ``float32`` 是 MegEngine 中默认的（也是最通用的）Tensor 数据类型。

.. note::

   我们会在 :mod:`megengine.quantization` 模块中提到对量化数据类型的支持。


dtype 作为参数
--------------

``dtype`` 同时也可以作为一些 API 中的参数，用来指定返回 Tensor 的数据类型：

>>> megengine.Tensor([1, 2, 3], dtype="float32")
Tensor([1. 2. 3.], device=xpux:0)

要获取指定数据类型的 Tensor, 还可以使用 :meth:`~.Tensor.astype` 方法对已经存在的 Tensor 进行转化。

>>> megengine.Tensor([1, 2, 3]).astype("float32")
Tensor([1. 2. 3.], device=xpux:0)


