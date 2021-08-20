.. _tensor-layout:

===============
Tensor 内存布局
===============
.. warning::

   * 这一部分内容属于底层细节，在绝大多数情景下用户不需要了解这些背后的设计。
     如果你希望成为 MegEngine 的核心开发者，了解底层细节将很有帮助，更多内容请参考开发者指南；
   * 相关的代码实现在： :src:`dnn/include/megdnn/basic_types.h` - ``megdnn::TensorLayout``.

.. seealso::

   NumPy 对 ndarray 内存布局的解释：
   `Internal memory layout of an ndarray
   <https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray>`_

Tensor 值如何存储在内存中
-------------------------

一个 :py:class:`~.Tensor` 类的实例由一维连续的计算机内存段组成。

结合 :ref:`tensor-indexing` 机制，可以将值映射到内存块中对应元素的位置，
而索引可以变化的范围由 Tensor 的 :ref:`形状 <tensor-shape>` 属性指定。
每个元素占用多少个字节以及如何解释这些字节由 Tensor 的 :ref:`数据类型 <tensor-dtype>` 属性指定。

一段内存本质上是连续的，有许多不同的方案可以将 N 维 Tensor 数组的项排列在一维块中。
根据排列顺序的区别，又可以分为行主序和列主序两种风格，下面我们以最简单的 2 维情况进行举例：

.. panels::
   :container: +full-width
   :card:

   .. image:: ../../../_static/images/Row_and_column_major_order.svg 

   上图分别使用行主序和列主序进行索引：

   * 其中 :math:`a_{11} \ldots a_{33}` 代表九个元素各自的值；
   * 偏移量和索引之间有着明显的关系。
   +++
   图片来自 `Row- and column-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_
   ---
   这个 2 维 Tensor 中的元素实际上可以由一维连续的内存块分别进行映射：

   .. list-table::
      :header-rows: 1

      * - Offset 
        - Access
        - Value
      * - 0
        - a[0][0]
        - a11
      * - 1
        - a[0][1]
        - a12
      * - 2
        - a[0][2]
        - a13
      * - 3
        - a[1][0]
        - a21
      * - 4
        - a[1][1]
        - a22
      * - 5
        - a[1][2]
        - a23
      * - 6
        - a[2][0]
        - a31
      * - 7
        - a[2][1]
        - a32
      * - 8
        - a[2][2]
        - a33

   +++
   这里以 C 风格所用的行主序进行举例。

MegEngine 和 NumPy 一样灵活，支持任何跨步索引方案，这里需要提到一个概念：步幅（Strides）。

.. _tensor-strides:

Tensor 的步幅
-------------

.. seealso::

   NumPy 的 ndarray 具有 :py:attr:`~numpy.ndarray.strides` 属性（MegEngine 中也存在着这一概念，但没有提供接口）。

.. note::

   Tensor 的步幅 ``strides`` 是一个元组，告诉我们遍历 Tensor 元素时要在每个维度中步进（step）的字节数；
   或者可以理解成在某个轴上索引元素时，单位刻度代表的内存范围，
   即必须在内存中跳过多少字节才能沿某个轴移动到下一个位置。
   这个属性通常不需要由用户进行修改。

以 2 维情况为例
~~~~~~~~~~~~~~~

想象有这样一个由 32 位（4 字节）整型元素组成的 Tensor:

>>> x = megengine.tensor([[0, 1, 2, 3, 4],
...                       [5, 6, 7, 8, 9]], dtype="int32")

该 Tensor 中的元素一个接一个地存储在内存中（称为连续内存块），占据 40 个字节。
我们必须跳过 4 个字节才能移动到下一列，但必须跳过 20 个字节才能到达下一行的相同位置。
因此，``x`` 的步幅为 ``(20, 4)``.

我们用 :math:`s^{\text {row }}` 表示行主序得到的步幅，则有 :math:`s_0^{\text {row }} = 4 \times 5 = 20`, :math:`s_1^{\text {row }} = 4`.

借助 :math:`s^{\text {row }}` 来计算，对应地 ``x[1][2]`` （对应值为 :math:`7` ）位置元素的字节偏移量为 :math:`1 \times 20 + 2 \times 4 = 28` .

推广到一般情况
~~~~~~~~~~~~~~

更一般的情况，对于形状为 ``shape`` 的一个 N 维 Tensor, 其步幅 :math:`s^{\text {row }}` 计算公式如下：

.. math::

   s_{k}^{\text {row }}=\text { itemsize } \prod_{j=k+1}^{N-1} d_{j}

其中 :math:`\text {itemsize}` 取决于 ``dtype``, 而 :math:`d_{j}=\text { self.shape }[j]` .

索引为 :math:`T[n_0, n_1, \ldots , n_{N-1}]` 元素的字节偏移量为：

.. math::

    n_{\text {offset }}=\sum_{k=0}^{N-1} s_{k} n_{k}

步幅概念的用途
~~~~~~~~~~~~~~

.. seealso::

   对于一些改变形状的 Tensor 操作，我们可以通过修改步幅来避免实际进行内存的拷贝。


