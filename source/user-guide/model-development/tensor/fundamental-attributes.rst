.. _tensor-fundamental-attributes:

========================
Rank, Axes 与 Shape 属性
========================

.. note::

   秩（Rank），轴（Axes）和形状（Shape）是 Tensor 数据结构最基本的属性。

   * 我们需要密切关注这三个属性之间的关系，它们也会影响到我们 :ref:`索引元素 <access-tensor-element>` 的具体方式。
   * 如果你对于这些基本属性的概念不是很清楚，将影响你对于 :ref:`tensor-manipulation` 背后行为逻辑的理解。

.. _tensor-rank:

Tensor 的秩
-----------

Tensor 的秩（Rank）指 Tensor 的维数（维度的数量，the number of dimensions）。

.. warning::

   * 也有人使用阶（Order）和度（Degree）来指代 Tensor 维度的数量，此时概念等同于秩。
   * 如果你学习过线性代数，可能接触了矩阵的秩的定义，例如 :py:func:`numpy.linalg.matrix_rank`.
     你可能也见过有人用非常复杂的概念对张量的秩进行了严谨的数学描述... 或许已被不同的说法搞得云里雾里。
     但是在深度学习领域，让我们保持简单，**秩可以表示一个 Tensor 维度的数量，仅此而已。**

如果我们说这儿有一个秩为 2 的（rank-2）Tensor，这等同于下面这些表述：

* 我们有一个矩阵（Matrix）
* 我们有一个 2 维数组（2d-array）
* 我们有一个 2 维张量（2d-tensor）

但在 MegEngine 中并没有为 Tensor 设计 ``rank`` 这个属性，而是使用了字面上更容易理解的 :py:attr:`~.Tensor.ndim`,
即 the number of dimensions 的缩写。这也是 NumPy 中用来表示多维数组 ``ndarray`` 维度的数量所设计的属性。

.. code-block:: python

   >>> x = megengine.tensor([1, 2, 3])
   >>> x.ndim
   1

   >>> x = megengine.tensor([[1, 2], [3, 4]])
   >>> x.ndim
   2

当你听到别人提到某个 Tensor 的秩时，在深度学习的语义下，你应该意识到他/她此时指代的是维数。
一种可能性是，这是一名使用过 TensorFlow 框架的用户，已经习惯了用秩来描述维度的数量。

好了，接下来我们能够忘记 ``rank`` 这种说法，用 ``ndim`` 进行交流了。

.. note::

   * 在 MegEngine 中为 Tensor 提供了 :py:attr:`~.Tensor.ndim` 属性，用来表示维度的数量。
   * Tensor 的 ``ndim`` 属性的值对应我们常说的 “一个 n 维张量” 时中的 ``n``.
     它告诉我们想要从当前这个 Tensor 中访问特定元素时所需索引（Indices）的个数。
     （参考 :ref:`access-tensor-element`）

.. seealso::

   NumPy 中对于维数的定义请参考 :py:attr:`numpy.ndarray.ndim`.


Tensor 的轴
-----------


