.. _tensor-creation:

===========
Tensor 创建
===========

创建 Tensor 的方式有很多种，常见操作如下：

* 如果想要使用已经存在的数据创建 Tensor, 可以将其传入 :class:`~.Tensor` 的构造函数：
  支持传入标量（Scalar）、Python :class:`list` 以及 NumPy :class:`~numpy.ndarray` （当然也包括 Tensor 自己）；
* 对应地，Tensor 也支持通过 :meth:`~.Tensor.item` , :meth:`~.Tensor.tolist` 和 :meth:`~.Tensor.numpy` 变成其它类型。

.. note::

   Tensor 中重载了 Python 中常见的运算符，支持直接进行 ``+`` ``-`` ``*`` ``/`` 等操作。
   但请注意，与 Tensor 相关的绝大多数计算接口，都封装在 :mod:`megengine.functional` 模块中。

.. note::

   更多时候，我们希望能够将现实中的数据（或数据集）在 MegEngine 中进行处理。
   一种普遍的方法是将其转化为 :class:`~.numpy.ndarray` 后传入 :class:`~.Tensor` 来创建实例；
   还可以利用 MegEngine 中封装好的接口，方便进行一些预处理操作，参考 :mod:`megengine.data` 子包。

