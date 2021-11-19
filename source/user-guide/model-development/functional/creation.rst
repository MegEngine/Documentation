.. _tensor-creation:

===================
如何创建一个 Tensor
===================

.. note::

   * 在 MegEngine 中创建 Tensor 与 `NumPy 创建数组
     <https://numpy.org/doc/stable/user/basics.creation.html>`_ 的途径类似；
   * 由于 ndarray 是 Python 数据科学社区中较为通用支持的格式
     （例如 SciPy、Pandas、OpenCV 等库都对 ndarray 提供了良好的支持），
     如果存在着 NumPy 已经实现但 MegEngine 尚未支持的创建途径，可以先创建 NumPy ndarray,
     再将其转换成 MegEngine Tensor. 也是下面提到的最后一种方法。

创建 Tensor 的常见途径如下：

* :ref:`create-tensor-from-python-structure` （例如 :py:class:`list`, :py:class:`tuple` ）；
* :ref:`create-tensor-with-intrinsic-functions` （例如 :py:func:`~.arange`, :py:func:`ones`, :py:func:`zeros` 等）；
* :ref:`create-tensor-from-random-package` （可从 :py:func:`~.random.normal`, :py:func:`~.random.uniform` 等分布中采样）；
* :ref:`create-tensor-through-manipulation` （例如 :py:func:`~.split`, :py:func:`stack` 等）；
* :ref:`create-tensor-from-ndarray`.

.. warning::

   任何通过已有数据创建 Tensor 的方式都是通过拷贝创建的，和原始数据不会共享内存。

.. _create-tensor-from-python-structure:

将 Python 序列转换为 Tensor
---------------------------

可以使用 Python 序列（例如列表和元组）定义 MegEngine Tensor.

列表 :py:class:`list` 和元组 :py:class:`tuple` 分别使用 ``[...]`` 和 ``(...)`` 定义，可以用来定义 Tensor 如何创建：

* 由数字组成的列表将创建 1 维 Tensor;
* 由列表组成的列表将创建 2 维 Tensor;
* 同理，更进一步的列表嵌套将创建更加高维的 Tensor.

>>> a1D = megengine.Tensor([1, 2, 3, 4])
>>> a2D = megengine.Tensor([[1, 2], [3, 4]])
>>> a3D = megengine.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

.. warning::

   * 这种写法其实调用了 :py:class:`~.megengine.Tensor` 类的构造函数，传入了 ``data`` 参数；
   * ``megengine.tensor`` 是 ``megengine.Tensor`` 的别名，**二者本质上没有任何区别。**

.. seealso::

   将 Tensor 转化为 Python 内置数据类型可以使用 :py:meth:`~.Tensor.item` 或 :py:meth:`~.Tensor.tolist` .

默认的数据类型
~~~~~~~~~~~~~~
.. seealso::

   当你使用 :py:class:`megengine.Tensor` 来定义新的 Tensor 时，需要考虑到其中每个元素的 :ref:`数据类型 <tensor-dtype>` 。

默认行为是以 32 位有符号整数 ``int32`` 或浮点数 ``float32`` 来创建 Tensor.

>>> megengine.Tensor([1, 2, 3, 4]).dtype
numpy.int32

>>> megengine.Tensor([1., 2., 3., 4.]).dtype
numpy.float32

如果你希望得到的 Tensor 是某种数据类型，则需要指定在创建 Tensor 时显式指定 dtype.

创建时指定数据类型
~~~~~~~~~~~~~~~~~~

数据类型是可以被显式指定的，但显式指定 ``dtype`` 有可能产生非预期的溢出，例如：

>>> a = megengine.Tensor([127, 128, 129], dtype="int8")
>>> a
Tensor([ 127 -128 -127], dtype=int8, device=xpux:0)

一个 8 位有符号整数表示从 -128 到 127 的整数。将 int8 Tensor 赋值给此范围之外的整数会导致溢出。

如果使用不匹配的数据类型执行计算，可能会得到非预期的结果，例如：

>>> a = megengine.Tensor([2, 3, 4], dtype="uint8")
>>> b = megengine.Tensor([5, 6, 7], dtype="uint8")
>>> a - b
Tensor([253 253 253], dtype=uint8, device=xpux:0)

你可能希望得到的结果是 ``[-3, -3, -3]``, 但在 ``uint8`` 数据类型下，这些值将表示为 ``253``.

不同数据类型之间的计算
~~~~~~~~~~~~~~~~~~~~~~

注意上面两个 Tensor 即 ``a`` 和 ``b`` 有着相同的 ``dtype: uint8``, 因此得到的 Tensor 的数据类型也会相同。
如果你在两个不同 ``dtype`` 的 Tensor 之间进行计算，MegEngine 将进行类型提升来满足计算要求：

>>> a - b.astype("int8")
Tensor([-3 -3 -3], dtype=int16, device=xpux:0)

注意到数据类型为 ``uint8`` 的 ``a`` 与数据类型为 ``int8`` 的 ``b`` 进行计算，
最终得到了一个数据类型为 ``int16`` 的 Tensor.

.. _create-tensor-with-intrinsic-functions:

使用内置函数创建 Tensor
-----------------------

.. note::

   * MegEngine 的 :py:mod:`functional` 子包内置了多个创建 Tensor 的函数（位于 :ref:`creation` ）；
   * 使用这些函数创建的 Tensor 默认数据类型为 ``float32``.

根据它们所创建的 Tensor 的维数，这些函数大致上可以分为三类：

* :ref:`create-1d-tensor`
* :ref:`create-2d-tensor`
* :ref:`create-nd-tensor`

.. _create-1d-tensor:

创建 1 维 Tensor
~~~~~~~~~~~~~~~~

创建 1 维 Tensor 的函数如 :py:func:`~.arange` 和 :py:func:`~.linspace`
通常需要至少两个输入，即 ``start`` 和 ``stop``.

:py:func:`~.arange` 将创建具有规律递增值的 Tensor, 下面显示了一些用法：

>>> megengine.functional.arange(10)
Tensor([0. 1. 2. 3. 4. 5. 6. 7. 8. 9.], device=xpux:0)

>>> megengine.functional.arange(2, 10, dtype="float")
Tensor([2. 3. 4. 5. 6. 7. 8. 9.], device=xpux:0)

>>> megengine.functional.arange(2, 3, 0.1)
Tensor([2.  2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9], device=xpux:0)

使用该函数得到的 Tensor 中的值不包括终点 ``stop``, 即范围为 ``[start, stop)``.

:py:func:`~.linspace` 将创建具有指定数量元素的 Tensor, 并在指定的开始值和结束值之间等距间隔。例如：

>>> megengine.functional.linspace(1., 4., 6)
Tensor([1.  1.6 2.2 2.8 3.4 4. ], device=xpux:0)

使用这个函数的好处是可以保证 Tensor 中元素的数量、值的起点和终点。

.. note::

   在 NumPy 中使用 :py:func:`~.arange` 的最佳实践是使用整型的 ``start``, ``stop`` 和 ``step`` 值。
   这是由于机器表示浮点数时存在着舍入误差，向 ``arange`` 传入非整数值时有可能得到非预期结果：

   >>> np.arange(7.8, 8.4, 0.05)
   array([7.8 , 7.85, 7.9 , 7.95, 8.  , 8.05, 8.1 , 8.15, 8.2 , 8.25, 8.3 ,
       8.35, 8.4 ])

   在 NumPy 中由于浮点误差的累积，最终的结果中将会看到 ``8.4`` 这个值。

   而在 MegEngine 中，``arange`` 内部调用了 ``linspace`` 进行实现，此时得到的结果与 NumPy 不同：

   >>> megengine.functional.arange(7.8, 8.4, 0.05)
   Tensor([7.8  7.85 7.9  7.95 8.   8.05 8.1  8.15 8.2  8.25 8.3  8.35], device=xpux:0)

.. _create-2d-tensor:

创建 2 维 Tensor
~~~~~~~~~~~~~~~~

创建 2 维 Tensor 的函数通常以表示为二维数组的特殊矩阵的属性来定义。

例如 :py:func:`~.eye` 定义了一个 2 维单位矩阵，行索引和列索引相等的元素为 1, 其余为 0, 如下所示：

>>> megengine.functional.eye(3)
Tensor([[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]], device=xpux:0)

>>> megengine.functional.eye(3, 5)
Tensor([[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]], device=xpux:0)

.. _create-nd-tensor:

创建 n 维 Tensor
~~~~~~~~~~~~~~~~

此类函数如 :py:func:`~.ones`, :py:func:`~.zeros` 通常可以根据给定的形状创建 Tensor.

>>> megengine.functional.zeros((2, 3))
Tensor([[0. 0. 0.]
 [0. 0. 0.]], device=xpux:0)

>>> megengine.functional.zeros((2, 3, 2))
Tensor([[[0. 0.]
  [0. 0.]
  [0. 0.]]
 [[0. 0.]
  [0. 0.]
  [0. 0.]]], device=xpux:0)

.. seealso::

   * 本质上它们都是通过调用 :py:func:`~.full` 来实现创建满足给定形状和值的 Tensor;
   * 使用 :py:func:`~.zeros_like`, :py:func:`~.ones_like`, :py:func:`~.full_like` 根据输入 Tensor 形状进行创建。

.. _create-tensor-from-random-package:

使用 random 子包随机生成
------------------------
例如使用 :py:func:`~.random.normal` 可以从服从正态分布的随机变量中采样：

>>> a = megengine.random.normal(100, 1, (5,))
Tensor([ 99.8308 101.949  100.2816 101.8977  99.9773], device=xpux:0)

使用 :py:func:`~.random.uniform` 可以从服从均匀分布的随机变量中采样：

>>> megengine.random.uniform(10, 20, (5,))
Tensor([12.557  17.8996 10.0152 18.2324 11.2644], device=xpux:0)

.. seealso::

   * Python 的 ``random`` 标准库文档 -- `Generate pseudo-random numbers <https://docs.python.org/3/library/random.html>`_
   * NumPy 的随机采样官方文档 ——  `Random sampling <https://numpy.org/doc/stable/reference/random/index.html>`_
   * MegEngine 所有随机数生成相关 API 都列举在 :py:mod:`~.random` 页面。

.. _create-tensor-through-manipulation:

基于现有的 Tensor 进行操作
--------------------------

.. note::

   使用 :py:func:`~.functional.copy` 函数可以拷贝一个 Tensor.

.. seealso::

   更多具体内容请参考 :ref:`tensor-manipulation` 页面。

.. _create-tensor-from-ndarray:

将 NumPy ndarray 转化为 MegEngine Tensor
----------------------------------------

我们也能够通过 :py:class:`~.megengine.Tensor`, 将 ndarray 作为输入数据传入，得到对应的 Tensor.

>>> a = np.array([1, 2, 3])
>>> a.dtype
dtype('int64')

>>> b = megengine.Tensor(a)
>>> Tensor([1 2 3], dtype=int32, device=xpux:0)
Tensor([1 2 3], dtype=int32, device=xpux:0)

通过 Tensor 的 :py:meth:`~.Tensor.numpy` 方法，我们可以得到 Tensor 转为 ndarray 后的结果：

>>> b.numpy()
array([1, 2, 3], dtype=int32)

.. seealso::

   相关注意事项如数据类型等，与 :ref:`create-tensor-from-python-structure` 一致。
