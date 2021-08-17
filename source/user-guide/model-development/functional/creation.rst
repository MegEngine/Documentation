.. _tensor-creation:

===========
创建 Tensor
===========

创建 Tensor 的方式有很多种，常见操作如下：

* 如果想要使用已经存在的数据创建 Tensor, 可以将其传入 :class:`~.Tensor` 的构造函数：
  支持传入标量（Scalar）、Python :class:`list` 以及 NumPy :class:`~numpy.ndarray` （当然也包括 Tensor 自己）；
* 对应地，Tensor 也支持通过 :meth:`~.Tensor.item` , :meth:`~.Tensor.tolist` 和 :meth:`~.Tensor.numpy` 变成其它数据结构；
* 我们也可以使用 :ref:`creation` 和 :py:mod:`~.random` 中的 API 来无中生有想要的 Tensor 对象。

使用已经存在的数据
------------------
一种常见的思路是，经过一系列流程，首先将输入的数据转化成 NumPy 支持的 ndarray 格式。

假设数据已被处理成 ndarray 类型的 ``data``, 我们则可以像下面的代码一样将其转换为 MegEngine Tensor:

>>> data = np.array([1, 2, 3])
>>> o1 = megengine.Tensor(data)
>>> o2 = megengine.tensor(data)
>>> print(o1)
>>> print(o2)
Tensor([1 2 3], dtype=int32, device=xpux:0)
Tensor([1 2 3], dtype=int32, device=xpux:0)

在这个过程中，MegEngine 会根据传入数据的类型自动进行推导，比如这里得到的是 ``int32`` 类型的数据。

.. warning::

   * MegEngine 基于已有数据生成的 Tensor 都是经过拷贝创建的，与原始数据的内存并不共享。
   * ``tensor`` 只是 ``Tensor`` 的别名， **二者在使用上没有任何的区别。**

使用相关算子创建
----------------

使用一些 :py:mod:`~.functional` 模块中提供的接口，可以用来生成 Tensor. 比如：

>>> megengine.functional.eye(3)
Tensor([[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]], device=xpux:0)

.. seealso::

   更多接口比如 :py:func:`~.ones`, :py:func:`~.full`, ... 均可在 :ref:`creation` 中找到。

使用一些 :py:mod:`~.random` 模块中提供的接口，可以随机生成符合要求的 Tensor. 比如：

>>> megengine.random.normal(100, 10, (3, 3))
Tensor([[114.2197  90.2169  99.4554]
 [ 90.5455  93.9307  92.4804]
 [116.8908 105.406  108.6316]], device=xpux:0)

.. seealso::

   更多接口比如 :py:func:`~.uniform`, :py:func:`~.beta` ... 均可在 :py:mod:`~.random` 中找到。

.. note::

   * 使用 :py:func:`.random.seed` 可以设置随机数种子；
   * 你也可以使用 NumPy 创建 ndarray 数据，然后转化成 MegEngine Tensor.

