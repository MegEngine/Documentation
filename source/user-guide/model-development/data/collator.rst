.. _data-collator-guide:

==========================
使用 Collator 定义合并策略
==========================

.. note::

   在使用 ``DataLoader`` 获取批数据的整个流程中， ``Collator`` 负责合并样本，最终得到批数据。

   >>> dataloader = DataLoader(dataset, collator=...)

   通常用户不用实现自己的 ``Collator``, 使用默认合并策略就可以处理大部分批数据合并的情况。
   但遇到一些默认合并策略难以处理的情景时，用户可以使用自己实现的 ``Collator``.
   参考 :ref:`custom-collator`.

.. warning::

   ``Collator`` 仅适用于 Map-style 的数据集，因为 Iterable-style 数据集的批数据必然是逐个合并的。

默认合并策略
------------

经过前面的处理流程后， ``Collator`` 通常会接收到一个列表：

* 如果你的 ``Dataset`` 子类的 ``__getitem__`` 方法返回的是单个元素，则 ``Collator`` 得到一个普通列表；
* 如果你的 ``Dataset`` 子类的 ``__getitem__`` 方法返回的是一个元组，则 ``Collator`` 得到一个元组列表。

MegEngine 中使用 :py:class:`~.Collator` 作为默认实现，通过调用 ``apply`` 方法来将列表数据合并成批数据：

>>> from megengine.data import Collator
>>> collator = Collator()

其实现逻辑中使用 :py:func:`numpy.stack` 函数来将列表中包含的所有样例在第一个维度（ ``axis=0`` ）合并。

.. seealso::

   MegEngine 中也提供了类似的 :py:func:`~.functional.stack` 函数，不过它仅适用于 Tensor 数据。

.. warning::

   默认的 ``Collator`` 支持 NumPy ndarrays, Numbers, Unicode strings, bytes, dicts 或 lists 数据类型。
   要求输入必须包含至少一种上述数据类型，否则用户需要使用自己定义的 ``Collator``.

Collator 效果示范
~~~~~~~~~~~~~~~~~
如果此时每个样本是形状为 :math:`(C, H, W)` 的图片 ``image``, 且在 ``Sampler`` 中指定了 ``batch_size`` 为 :math:`N`.
那么 ``Collator`` 的主要目的就是将获得的该样本列表合并成一个形状为 :math:`(N, C, H, W)` 的批样本结构。

我们可以模拟得到这样一个 ``image_list`` 数据，并借助 ``Collator`` 得到 ``batch_image``:

>>> N, C, H, W = 5, 3, 32, 32
>>> image_list = []
>>> for i in range(N):
...     image_list.append(np.random.random((C, H, W)))
>>> print(len(image_list), image_list[0].shape)
5 (3, 32, 32)

>>> batch_image = collator.apply(image_list)
>>> batch_image.shape
(5, 3, 32, 32)

如果样本带有标签，则 ``Collator`` 就需要将由 ``(image, label)`` 元组构成的列表合并，
形成一个大的 ``(batch_image, bacth_label)`` 元组。这也是我们对 ``DataLoader`` 进行迭代时通常会获得的东西。

在下面的示例代码中，``sample_list`` 中每个元素都是一个元组（假设所有的标签都用整型 ``1`` 来表示）：

>>> sample_list = []
>>> for i in range(N):
...     sample_list.append((np.random.random((C, H, W)), 1))
>>> type(sample_list[0])
tuple
>>> print(sample_list[0][0].shape, type(sample_list[0][1]))
(3, 32, 32) <class 'int'>

MegEngine 提供的默认 ``Collator`` 也能够很好地处理这种情况：

>>> batch_image, batch_label = collator.apply(sample_list)
>>> print(batch_image.shape, batch_label.shape)
(5, 3, 32, 32) (5,)

.. warning::

   需要注意的是，此时 ``batch_label`` 已经被转换成了 ndarray 数据结构。

.. _custom-collator:

自定义 Collator
---------------

当默认的 ``stack`` 合并策略无法满足我们的需求时，我们则需要考虑自定义 ``Collator``:

* 需要继承 ``Collator`` 类，并在子类中实现 ``apply`` 方法；
* 我们实现的 ``apply`` 方法将被 ``DataLoader`` 调用。

.. seealso::

   * :models:`official/vision/keypoints/dataset.py#L167` - ``HeatmapCollator``
   * :models:`official/vision/detection/tools/utils.py#L125` - ``DetectionPadCollator``
