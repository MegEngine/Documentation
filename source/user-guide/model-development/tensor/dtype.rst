.. _tensor-dtype:

===============
Tensor 数据类型
===============

.. seealso::

   在计算机科学中，数据类型负责告诉编译器或解释器程序员打算如何使用数据。
   参考 `Data type <https://en.wikipedia.org/wiki/Data_type>`_ WIKI.

   MegEngine 中借助 :class:`numpy.dtype` 来表示基础数据类型，参考如下：

   * NumPy 中有着专门实现的 :class:`numpy.dtype`, 参考其对
     `Data type objects <https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes>`_ 
     的解释；
   * NumPy 官方 `Data types <https://numpy.org/doc/stable/user/basics.types.html>`_ 
     文档中对数组类型和转换规则进行了解释。

   根据 :ref:`mep-0003` ，MegEngine 将参考《数组 API 标准》中对 
   `数据类型 <https://data-apis.org/array-api/latest/API_specification/data_types.html>`_ 的规格定义。

上面提到的数据类型（Data type, :attr:`~.Tensor.dtype` ）是 Tensor 的一种基础属性，
单个 Tensor 内的元素的数据类型完全一致，每个元素占据的内存空间也完全相同。
Tensor 数据类型可以在创建时指定，也可以从已经存在的 Tensor 中指定进行转化，此时 :ref:`dtype-argument` 。
``float32`` 是 MegEngine 中最经常用到的 Tensor 数据类型。

>>> a = megengine.functional.arange(5)
>>> a.dtype
numpy.float32

已经支持的类型
--------------

目前支持的数据类型如下：

.. list-table::
   :header-rows: 1

   * - 数据类型
     - numpy.dtype
     - 等效字符串
     - 数值区间
   * - 单精度浮点
     - :any:`numpy.float32` / :class:`numpy.single`  
     - ``float32``
     - 参考 IEEE 754-2019
   * - 半精度浮点
     - :any:`numpy.float16` / :class:`numpy.half`
     - ``float16``
     - 参考 IEEE 754-2019
   * - 无符号 8 位整型
     - :any:`numpy.uint8`
     - ``uint8``
     - :math:`[0, 2^{8}-1]`
   * - 有符号 8 位整型  
     - :any:`numpy.int8`
     - ``int8``
     - :math:`[-2^{7}, 2^{7}-1]`
   * - 有符号 16 位整型
     - :any:`numpy.int16`
     - ``int16``
     - :math:`[−2^{15}, 2^{15}-1]`
   * - 有符号 32 位整型
     - :any:`numpy.int32`
     - ``int32``
     - :math:`[−2^{31}, 2^{31}-1]`
   * - 布尔型
     - :any:`numpy.bool8` / :class:`numpy.bool_`
     - ``bool``
     - ``True`` 或者 ``False``

.. warning::

   并不是所有的已有算子都支持上述任意数据类型之间的计算（仅保证 ``float32`` 类型全部可用）。

.. note::

   我们会在 :mod:`megengine.quantization` 模块中提到对量化数据类型的支持。

.. _dtype-argument:

dtype 作为参数使用
------------------

:class:`~.Tensor` 初始化时以及调用 :ref:`创建 Tensor <tensor-creation>` 函数时可接受 ``dtype`` 参数，用来指定数据类型：

>>> megengine.Tensor([1, 2, 3], dtype="float32")
Tensor([1. 2. 3.], device=xpux:0)

>>> megengine.functional.arange(5, dtype="float32")
Tensor([0. 1. 2. 3. 4.], device=xpux:0)

如果使用已经存在的数据来创建 Tensor 而不指定 ``dtype``, 则 Tensor 的数据类型将根据输入的类型推导：

>>> megengine.Tensor([1, 2, 3])
Tensor([1 2 3], device=xpux:0)

>>> megengine.Tensor([1, 2, 3]).dtype
int32

基本的推导规则为：

* Python scalar bool -> MegEngine Tensor bool
* Python scalar int -> MegEngine Tensor int32
* Python scalar float -> MegEngine Tensor float32
* Numpy array dtype -> MegEngine Tensor dtype （保持一致，前提是类型支持）

.. warning::

   如果使用不支持类型的 NumPy 数组作为输入创建 MegEngine Tensor, 可能会出现非预期行为。
   因此最好在做类似转换时每次都指定 ``dtype`` 参数，或先转换 NumPy 数组为支持的数据类型。

另外还可以使用 :meth:`~.Tensor.astype` 方法得到转换数据类型后的 Tensor（原 Tensor 不变）：

>>> megengine.Tensor([1, 2, 3]).astype("float32")
Tensor([1. 2. 3.], device=xpux:0)

.. _dtype-promotion:

类型提升规则
------------

.. note::

   根据 :ref:`mep-0003`, 类型提升规则应当参考《数组 API 标准》
   中的 `相关规定 <https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`_ ：

   .. image:: ../../../_static/images/dtype_promotion_lattice.png
      :align: center

   多个不同数据类型的 Tensor 或 Python 标量作为操作数参与运算时，
   所返回的结果类型由上图展示的关系决定——
   沿着箭头方向提升，汇合至最近的数据类型，将其作为返回类型。
   
   * 决定类型提升的关键是参与运算的数据的类型，而不是它们的值；
   * 图中的虚线表示 Python 标量的行为在溢出时未定义；
   * 布尔型、整数型和浮点型 ``dtypes`` 之间未连接，表明混合类型提升未定义。

   在 MegEngine 中，由于尚未支持《标准》中的所有类型，当前提升规则如下图所示：

   .. image:: ../../../_static/images/dtype_promotion_megengine.png
      :align: center

   * 遵循 **类型优先** 的原则，存在 bool -> int -> float 的混合类型提升规则；
   * 当 Python 标量类型与 Tensor 进行混合运算时，转换成 Tensor 数据类型；
   * 布尔型 ``dtype`` 与其它类型之间未连接，表明相关混合类型提升未定义。

.. note::

   这里讨论的类型提升规则主要适用于 :ref:`element-wise-operations` 的情况。

举例如下， ``uint8`` 和 ``int8`` 类型 Tensor 运算会返回 ``int16`` 类型 Tensor:

>>> a = megengine.Tensor([1], dtype="int8")  # int8 -> int16
>>> b = megengine.Tensor([1], dtype="uint8")  # uint8 -> int16
>>> (a + b).dtype
numpy.int16

``int16`` 和 ``float32`` 类型 Tensor 运算会返回 ``float32`` 类型 Tensor:

>>> a = megengine.Tensor([1], dtype="int16")  # int16 -> int32 -> float16 -> float32
>>> b = megengine.Tensor([1], dtype="float32")
>>> (a + b).dtype
numpy.float32

Python 标量和 Tensor 混合运算时，在种类一致时，会将 Python 标量转为相应的 Tensor 数据类型：

>>> a = megengine.Tensor([1], dtype="int16")
>>> b = 1  # int -> a.dtype: int16
>>> (a + b).dtype
numpy.int16

注意，如果此时 Python 标量是 ``float`` 类型，而 Tensor 为 ``int``, 则按照类型优先原则提升：

>>> a = megengine.Tensor([1], dtype="int16")
>>> b = 1.0  # Python float -> float32
>>> (a + b).dtype
numpy.float32

此时 Python 标量按照 :ref:`上一小节 <dtype-argument>` 的推导规则转为了 ``float32`` Tensor.
