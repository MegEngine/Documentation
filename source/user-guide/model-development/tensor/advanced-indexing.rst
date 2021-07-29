.. _tensor-advanced-indexing:

===============
Tensor 高级索引
===============

.. warning::

   不能将 NumPy 中存在的一些概念和设计直接应用于 MegEngine. 

.. seealso::

   在 MegEngine 中，想要 :ref:`access-tensor-element` ，可以使用标准的 ``x[obj]`` 语法。
   看上去一切都和 NumPy 很相似，后者的官方文档中也对 :class:`~.numpy.ndarray` 的各种索引方式都
   `进行了解释 <https://numpy.org/doc/stable/reference/arrays.indexing.html>`_ 。
   但 MegEngine 的 Tensor 实现和 NumPy 还是略有不同，如果不清楚某些细节，可能无法对一些现象做出解释。


举例：切片索引得到的对象
------------------------

.. panels::
   :container: +full-width
   :card:

   MegEngine 
   ^^^^^^^^^
   >>> x = tensor([1., 2., 3.])
   >>> y = x[:]
   >>> y[1] = 6
   >>> x
   Tensor([1. 2. 3.], device=xpux:0) 
   ---
   NumPy
   ^^^^^
   >>> x = array([1., 2., 3.])
   >>> y = x[:]
   >>> y[1] = 6
   >>> x
   array([1., 6., 3.])

出现这种情况的原因是，在 NumPy 中使用切片索引时，得到的是原数组的视图（View）。
改变视图中的元素，原始数组中的元素也会发生变化 —— 这是很多 NumPy 用户初学时容易困扰的地方。
**而在 MegEngine 中没有视图的概念，** 通过切片得到的子 Tensor 和原 Tensor 占用的是不同的内存区域。

.. _slice-will-not-reduce-dim:

使用切片不会改变形状
~~~~~~~~~~~~~~~~~~~~

.. warning::

   切片的作用是从整体中取出一部分，但不会产生降低维度的行为。

>>> b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> b[1:2]
[[4, 5, 6]]
>>> b[1:2][0:1]
[[4, 5, 6]]

以上面的代码为例，整个过程中，切片得到的都是一个 ``ndim=2`` 的 Tensor.

* 执行 ``b[1:2]`` 得到的结果是 ``[[4, 5, 6]]`` 而不是 ``[4, 5, 6]``.
* 对 ``[[4, 5, 6]]`` 进行 ``[0:1]`` 切片，得到的还是 ``[[4, 5, 6]]``.

错误的理解思路可能是这样的：

* 执行 ``b[1:2]`` 得到的结果是 ``[4, 5, 6]``. —— 错！切片不会降维！
* 对 ``[4, 5, 6]`` 进行 ``[0:1]`` 切片，得到的是 ``4``. —— 同理，错！不会降维！

.. seealso::

   如果你希望切片操作后能去掉冗余的维度，可以使用 :func:`~.squeeze` .

使用列表索引特定元素
--------------------

实际上除了切片索引，我们还可以使用由多个整数组成的列表进行索引：

.. panels::
   :container: +full-width
   :card:

   MegEngine 
   ^^^^^^^^^
   >>> x = tensor([1., 2., 3.])
   >>> y = x[[0, 1, 2]]
   >>> y[1] = 6
   >>> x
   Tensor([1. 2. 3.], device=xpux:0) 
   ---
   NumPy
   ^^^^^
   >>> x = array([1., 2., 3.])
   >>> y = x[[0, 1, 2]]
   >>> y[1] = 6
   >>> x
   array([1., 2., 3.])

此时 NumPy 将不会生成原始数组的视图，与 MegEngine 的逻辑一致。

注意语法细节，一些用户容易将列表索引写成如下形式：

>>> x = tensor([1., 2., 3.])
>>> y = x[0, 1, 2]
IndexError: too many indices for tensor: tensor is 1-dimensional, but 3 were indexed

实际上这是对 Tensor 的 n 个维度分别进行索引的语法。

.. _multi-dim-indexing:

在多个维度进行索引
------------------

在多个维度进行切片
------------------
