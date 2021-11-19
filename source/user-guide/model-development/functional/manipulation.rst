.. _tensor-manipulation:

======================
如何对 Tensor 进行操作
======================

这个文档将展示如何使用 :py:mod:`functional` 中的接口对已经存在的 Tensor 进行操作。

* :ref:`reshaping-tensor` （例如 :py:func:`~.reshape`, :py:func:`~.flatten` 和 :py:func:`~.transpose` ）
* :ref:`squeeze-and-unsqueeze` （使用 :py:func:`~.expand_dims` 增加维度，使用 :py:func:`~.squeeze` 降低维度） 
* :ref:`tensor-broadcasting` （使用 :py:func:`~.broadcast_to` 按照规则对 Tensor 进行扩充 ）
* :ref:`tensor-manipulation-miscs` （Tensor 的多变一与一变多等操作）
* 更多 API 请直接参考 :ref:`manipulation` .

这些操作的相同点是：没有基于 Tensor 的数值进行计算，往往改变的是 Tensor 的形状。

.. note::

   * 由于 :ref:`tensor-shape` 蕴含着太多核心的信息，因此我们操作 Tensor 时需要额外关注形状的变化；
   * 一些对 Tensor 的操作需要指定沿着轴进行，如果不清楚轴的概念，请参考 :ref:`tensor-axes` 。

.. seealso::

   * 使用 :py:func:`~.functional.copy` 或 :py:meth:`.Tensor.to` 可以得到位于指定 :ref:`所在设备 <tensor-device>` 的 Tensor;
   * 使用 :py:meth:`.Tensor.astype` 可以改变 Tensor 的 :ref:`数据类型 <tensor-dtype>` ；
   * 使用 :py:meth:`.Tensor.item`, :py:meth:`.Tensor.tolist`, :py:meth:`.Tensor.numpy` 可以将 Tensor 转化成其它数据结构。

   更多内置方法请参考 :py:class:`Tensor` API 文档。

.. warning::

   MegEngine 中的计算都是非原地（inplace）进行的，返回新 Tensor, 但原始的 Tensor 并没有改变。

   >>> a = megengine.functional.arange(6)
   >>> a.reshape((2, 3))
   Tensor([[0. 1. 2.]
     [3. 4. 5.]], device=cpux:0)
   
   可以发现调用 :py:meth:`.Tensor.reshape` 将得到 ``a`` 改变形状后的 Tensor, 但 ``a`` 自身没有变化：

   >>> a
   Tensor([0. 1. 2. 3. 4. 5.], device=cpux:0)

.. warning::

   一些操作的背后并没有实际产生数据拷贝，想要了解底层逻辑，可以参考 :ref:`tensor-layout` 。

.. _reshaping-tensor:

通过重塑来改变形状
------------------

重塑的特点是不会改变原 Tensor 中的数据，但会得到一个给定新形状的 Tensor.

:py:func:`~.reshape` 可能是最重要的对 Tensor 的操作，设我们现在有这样一个 Tensor:

>>> a = megengine.functional.arange(12)
>>> a
Tensor([ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.], device=cpux:0)
>>> a.shape
(12,)
>>> a.size
12

通过 :py:func:`~.reshape` 我们可以在不改变 ``size`` 属性的情况下改变 ``shape``:  

>>> megengine.functional.reshape(a, (3, 4))
Tensor([[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]], device=xpux:0)

``reshape`` 的目标形状中支持存在一个值为 ``-1`` 的轴，其真实值将根据 ``size`` 自动地推导出来：

>>> a = megengine.functional.ones((2, 3, 4))
>>> a.shape
(2, 3, 4)

>>> megengine.functional.reshape(a, (-1, 4)).shape
(6, 4)

>>> megengine.functional.reshape(a, (2, -1)).shape
(2, 12)

>>> megengine.functional.reshape(a, (2, -1, 2)).shape
(2, 6, 2)

:py:func:`~.flatten` 操作将对起点轴 ``start_axis`` 到终点轴 ``end_axis`` 的子张量进行展平，等同于特定形式的 ``reshape``:

>>> a = megengine.functional.ones((2, 3, 4))
>>> a.shape
(2, 3, 4)

>>> megengine.functional.flatten(a, 1, 2).shape
(2, 12)

而 :py:func:`~.transpose` 将根据给定的模版 ``pattern`` 来改变形状：

>>> a = megengine.functional.ones((2, 3, 4))
>>> a.shape
(2, 3, 4)

>>> megengine.functional.transpose(a, (1, 2, 0)).shape
(3, 4, 2)

上面的例程中将原本排序为 ``(0, 1, 2)`` 的轴变为了 ``(1, 2, 0)`` 顺序。

.. seealso::

   * 更多使用细节说明请参考对应的 API 文档；
   * 这类 API 在 :py:class:`Tensor` 中都提供了对应的内置方法实现 ——
       :py:meth:`.Tensor.reshape` / :py:meth:`.Tensor.flatten` / :py:meth:`.Tensor.transpose`

.. _squeeze-and-unsqueeze:

对 Tensor 进行升维和降维
------------------------

改变 Tensor 形状的另一个方法是增加它的维度，或者删除冗余的维度。
一些时候，我们用挤压（Squeeze）来称呼删除维度的操作，用解压（Unsqueeze）来称呼添加维度的操作。
显然，升维或者降维都不会改变一个 Tensor 中元素的个数，这与重塑 Tensor 的特点比较相似。

使用 :py:func:`~.expand_dims` 增加维度时，需要指定所在轴的位置：

>>> a = megengine.Tensor([1, 2]) 
>>> b = megengine.functional.expand_dims(a, axis=0)
>>> print(b, b.shape, b.ndim)
Tensor([[1 2]], dtype=int32, device=xpux:0) (1, 2) 2

>>> b = megengine.functional.expand_dims(a, axis=1)
>>> print(b, b.shape, b.ndim)
Tensor([[1]
 [2]], dtype=int32, device=xpux:0) (2, 1) 2

如上所示，对于一个 1 维 Tensor, 在 ``axis=0`` 的位置增加维度，则原 Tensor 中的元素将位于 ``axis=1`` 维度；
在 ``axis=1`` 的位置增加维度，则原 Tensor 中的元素将位于 ``axis=0`` 维度。

.. note::

   虽然使用 :py:func:`~.reshape` 可以达到一样的效果：

   >>> a = megengine.Tensor([1, 2])
   >>> b = megengine.functional.reshape(a, (1, -1))
   >>> print(b, b.shape, b.ndim)
   Tensor([[1 2]], dtype=int32, device=xpux:0) (1, 2) 2

   但我们应当尽可能使用语义明确的接口。

增加新的维度
~~~~~~~~~~~~

增加维度的逻辑很简单，新 Tensor 从 ``axis=0`` 开始判断，如果该维度是 :py:func:`~.expand_dims` 得到的，则该新增维度的轴长度为 1.
如果该维度并不在需要新增维度的位置上，则按照原 Tensor 的形状逐个维度进行填充。举例如下：

>>> a = megengine.functional.ones((2, 3))
>>> b = megengine.functional.expand_dims(a, axis=1)
>>> b.shape
(2, 1, 3)

对于 2 维 Tensor ``a``, 我们想要在 ``axis=1`` 的位置添加一个维度。从 ``axis=0`` 开始排列新 Tensor, 由于 ``0`` 并不在增加维度的范围内，
因此新 Tensor 的 ``axis=0`` 维度将由 ``a`` 的第 0 维进行填充（即例子中长度为 2 的维度）；
接下来排列新 Tensor 中 ``axis=1`` 的位置，该位置是新增的维度，因此对应位置轴长度为 1.
新增维度后，原 Tensor 中后续的维度（这里是长度为 2 的维度）将直接接在当前新 Tensor 维度后面，最终得到形状 ``(2, 1, 3)`` 的 Tensor.

:py:func:`~.expand_dims` 还支持一次性新增多个维度，规则与上面描述的一致：

>>> a = megengine.functional.ones((2, 3))
>>> b = megengine.functional.expand_dims(a, axis=(1, 2))
>>> b.shape
(2, 1, 1, 3)

>>> a = megengine.functional.ones((2, 3))
>>> b = megengine.functional.expand_dims(a, axis=(1, 3))
>>> b.shape
(2, 1, 3, 1)

.. warning::

   使用 :py:func:`~.expand_dims` 新增维度时要注意范围，不能超出原 Tensor 的维数 ``ndim`` 与新增维数之和：

   >>> a = megengine.functional.ones((2, 3))
   >>> b = megengine.functional.expand_dims(a, axis=3)
   >>> b.shape
   extra message: invalid axis 3 for ndim 3 

   在上面的例子中，原 Tensor 维数 ``ndim`` 为 2, 如果新增一个维度，最终新 Tensor 的维数应该是 3.
   新增的轴应该满足 ``0 <= axis <= 2``, 上面给出的 3 已经超出了所能表达的维度范围。

去除冗余的维度
~~~~~~~~~~~~~~

与 :py:func:`~.expand_dims` 相反的操作是 :py:func:`~.squeeze`, 能够去掉 Tensor 中轴长度为 1 的维度：

>>> a = megengine.Tensor([[1, 2]])
>>> b = megengine.functional.squeeze(a)
>>> b
Tensor([1 2], dtype=int32, device=xpux:0)

默认 :py:func:`~.squeeze` 将移除掉所有轴长度为 1 的维度，也可以通过 ``axis`` 指定性移除：

>>> a = megengine.functional.ones((1, 2, 1, 3))
>>> megengine.functional.squeeze(a, axis=0).shape
(2, 1, 3)

>>> megengine.functional.squeeze(a, axis=2).shape
(1, 2, 3)

同样地，:py:func:`~.squeeze` 支持一次性去除多个指定的维度： 

>>> megengine.functional.squeeze(a, axis=(0, 2)).shape
(2, 3)

.. warning::

   使用 :py:func:`~.squeeze` 新增维度时要注意轴长度，只能去掉轴长度为 1 的冗余维度。

.. _tensor-broadcasting:

对 Tensor 进行广播
------------------

.. seealso::

   * MegEngine 的广播机制与 NumPy `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ 一致，这里使用一样的代码与配图；
   * 在进行 :ref:`element-wise-operations` 时，会尝试进行广播使得输入 Tensor 的形状一致；
   * 我们可以使用 :py:func:`~.broadcast_to` 将 Tensor 广播至指定的形状。

.. seealso::

   * 本小节示例图形来自 NumPy 文档，生成它们的源代码参考自：
     `astroML <http://www.astroml.org/book_figures/appendix/fig_broadcast_visual.html>`_
   * 类似的扩充 Tensor 形状的 API 有：:py:func:`~.repeat` / :py:func:`~.tile`

术语 “广播” 描述了 MegEngine 在算术运算期间如何处理具有不同形状的 Tensor 对象。
出于某些原因，在运算时需要让较小的 Tensor 基于较大的 Tensor 进行广播，使得它们具有兼容的形状。
广播机制可以避免制作不必要的数据拷贝，使得一些算法的实现变得更加高效（参考 :ref:`tensor-layout` ）。

Tensor 之间经常进行逐个元素的运算，在最简单的情况下，两个 Tensor 具有完全相同的形状：

>>> a = megengine.Tensor([1.0, 2.0, 3.0])
>>> b = megengine.Tensor([2.0, 2.0, 2.0])
>>> a * b
Tensor([2. 4. 6.], device=xpux:0)

当两个 Tensor 的形状不一致，但满足广播的条件时，则会先广播至一样的形状，再进行运算。

最简单的广播示例发生在将 Tensor 和标量的元素级别运算中：

>>> a = megengine.Tensor([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
Tensor([2. 4. 6.], device=xpux:0)

结果等同于前面的例子，其中 ``b`` 是一个标量。我们可以想象成，在算术运算期间，
标量 ``b`` 被 *拉伸（Stretch）*  成一个形状与 ``a`` 相同的 Tensor,
此时 ``b`` 中的新元素只是原始标量元素的副本。
但这里的拉伸只是概念上的类比，MegEngine 的 Tensor 内部存在着一些机制，
可以统一使用原始的标量值，而无需发生实际的数据拷贝；
这些机制也使得广播行为更具有内存和计算效率。

.. figure:: ../../../_static/images/theory.broadcast_1.gif
   :align: center

   在最简单的广播示例中，标量 ``b`` 被拉伸为与 ``a`` 形状相同的数组，因此这些形状兼容逐个元素的乘法。

我们可以通过使用 :py:func:`~.broadcast_to` 来人为地对 Tensor 进行广播，同样以 ``b`` 为例：

>>> b = megengine.Tensor(2.0)
>>> broadcast_b = megengine.functional.broadcast_to(b, (1, 3))
>>> broadcast_b
Tensor([[2. 2. 2.]], device=xpux:0)

.. warning::

   MegEngine 中要求 :py:mod:`functional` API 输入数据为 Tensor,
   因此这里的传给 ``broadcast_to`` 的标量实际上是一个 0 维 Tensor.
   在使用 ``*`` 等算术运算时没有这样的限制，因此无需将输入提前转化为 Tensor.

.. _broadcasting-rule:

广播机制与规则
~~~~~~~~~~~~~~

.. note::

   在对两个 Tensor 进行运算时，MegEngine 从它们的形状最右边的元素开始逐个向左比较。
   当两个维度兼容时（指对应轴长度相等，或者其中一个值为 1），则能够进行相应的运算。

   如果无法满足这些条件，则会抛出异常，表明 Tensor 之间的形状不兼容：

   .. code-block:: shell

      ValueError: operands could not be broadcast together

.. warning::

   广播规则并没有要求进行运算的 Tensor 具有相同的维数 ``ndim``.

   例如，如果你有一个 256 x 256 x 3 的 RGB 值 Tensor, 
   并且想用不同的值缩放图像中的每种颜色，则可以将图像乘以具有 3 个值的一维 Tensor.

   .. list-table::

      * - Image
        - (3d array)
        - 256 x
        - 256 x
        - 3
      * - Scale
        - (1d array)
        - 
        - 
        - 3
      * - Result
        - (3d array)
        - 256 x
        - 256 x
        - 3
   
   >>> image = megengine.random.normal(0, 1, (256, 256, 3))
   >>> scale = megengine.random.normal(0, 1, (          3,))
   >>> result = image * scale
   >>> print(image.shape, scale.shape, result.shape)
   (256, 256, 3) (3,) (256, 256, 3)

   在下面这个例子中，Tensor ``a`` 和 ``b`` 都具有长度为 1 的轴，这些轴在广播操作中扩展为更大的尺寸。

   .. list-table::

      * - A
        - (4d array)
        - 8 x
        - 1 x
        - 6
        - 1
      * - B
        - (3d array)
        - 
        - 7 x
        - 1 x
        - 5
      * - Result
        - (4d array)
        - 8 x
        - 7 x
        - 6 x
        - 5

   >>> a = megengine.random.normal(0, 1, (8, 1, 6, 1))
   >>> b = megengine.random.normal(0, 1, (   7, 1, 5))
   >>> result = a * b
   >>> print(a.shape, b.shape, result.shape)
   (8, 1, 6, 1) (7, 1, 5) (8, 7, 6, 5)

更多广播可视化例子
~~~~~~~~~~~~~~~~~~

下面这个例子展示了 2 维 Tensor 和 1 维 Tensor 之间的加法运算：

>>> a = megengine.Tensor([[ 0.0,  0.0,  0.0],
>>> ...                   [10.0, 10.0, 10.0],
>>> ...                   [20.0, 20.0, 20.0],
>>> ...                   [30.0, 30.0, 30.0]])
>>> b = megengine.Tensor([1.0, 2.0, 3.0])
>>> a + b
Tensor([[ 1.  2.  3.]
 [11. 12. 13.]
 [21. 22. 23.]
 [31. 32. 33.]], device=xpux:0)

如下图所示，``b`` 经过广播后将添加到 ``a`` 的每一行。

 .. figure:: ../../../_static/images/theory.broadcast_2.gif
    :align: center

    如果 1 维 Tensor 元素的数量与 2 维 Tensor 的列数相匹配，则它们相加会导致广播。
    
 .. figure:: ../../../_static/images/theory.broadcast_3.gif
    :align: center

    如果两个 Tensor 的尾部维度不兼容时，广播失败，无法进行元素级别的运算。

.. _tensor-manipulation-miscs:

对 Tensor 进行拼接、切分
------------------------

另一类常见的 Tensor 操作是将多个 Tensor 拼成一个 Tensor, 或者是将一个 Tensor 拆成多个 Tensor.

:py:func:`~.concat`
  沿着已经存在的轴连接 Tensor 序列。

:py:func:`~.stack`
  沿着新的轴连接 Tensor 序列。

:py:func:`~.split`
  将 Tensor 切分成多个相同大小的子 Tensor.

更多的接口和详细使用说明请参考 :ref:`manipulation` API 文档。


