.. _data-sampler-guide:

=========================
使用 Sampler 定义抽样规则
=========================

经过 :ref:`dataset-guide` 这一步骤后，``DataLoader`` 便可以知道如何从数据集中加载数据到内存。
但除了要能够获取数据集中的单个样本数据外，每批数据的生成也有着一定的要求，比如每批数据的规模大小、以及对抽样规则的要求等，
都需要有对应的配置，使用 ``Sampler`` 可以对每批数据的抽样规则进行自定义。
在其它小节我们还会介绍如何 :ref:`data-collator-guide` ，但本小节我们主要介绍 ``Sampler`` 的概念和使用。

准确来说，抽样器的职责是决定数据的获取顺序，方便为 ``DataLoader`` 提供一个可供迭代的多批数据的索引。

>>> dataloader = DataLoader(dataset, sampler=RandomSampler)

在 MegEngine 中，:py:class:`Sampler` 是所有抽样器的抽象基类，在大部分情况下用户无需对抽样器进行自定义实现，
因为在 MegEngine 中已经实现了常见的各种抽样器，比如上面示例代码中的 ``RandomSampler`` 抽样器。

.. note::

   由于 :ref:`dataset-type` 可以分为 Map-style 和 Iterable 两种，因此 ``Sampler`` 也可分为两类：

   * :ref:`MapSampler <map-sampler-guide>` : 适用于 Map-style 数据集的抽样器：

     * 根据抽样方式划分： :ref:`sequential-sampler-guide` （默认方式） / :ref:`random-sampler-guide` / :ref:`replacement-sampler-guide`
     * 我们还可以使用 :ref:`Infinite <infinite-sampler-guide>` 封装上述类来实现 :ref:`infinite-sampler-guide` ；
     * 如果你想实现自己的 ``MapSampler``, 则需要自行继承该类，并实现 ``sample`` 方法。

   * :ref:`StreamSampler <stream-sampler-guide>` : 适用于 Iterable-style 数据集的抽样器。

.. _map-sampler-guide:

如何使用 MapSampler
-------------------
:py:class:`MapSampler` 类签名如下：

.. class:: MapSampler(dataset, batch_size=1, drop_last=False,
           num_samples=None, world_size=None, rank=None, seed=None) 
   :noindex:

其中 ``dataset`` 用来获取数据集信息， ``batch_size`` 参数用于指定批数据的规模，
``drop_last`` 参数用于设置是否丢掉最后一批不完整的数据，
而 ``num_samples``, ``world_size``, ``rank`` 和 ``seed`` 这些参数用于分布式训练情景。

.. warning::

   ``MapSampler`` 不会真正地将数据读入内存且最终返回经过抽样后的数据，因为会带来比较大的内存开销。
   实际上它根据 ``Dataset`` 中实现的 ``__len__`` 协议来获取样本容量，形成 ``[0, 1, ...]`` 整数索引列表，
   并按照子类实现的 ``sample`` 方法对该整数索引列表进行抽样，
   返回一个可供迭代的列表，里面存放的是抽样得到的各批数据所对应的索引。
   只有在迭代 ``DataLoader`` 时才会根据这些索引加载数据。

下面我们通过 MegEngine 中提供的最常见的几类抽样器，来展示相关概念。

首先随机生成一个形状为 ``(N, C, H, W)`` 的图片数据集，分别对应样本容量、通道数、高度和宽度。

.. code-block::

   import numpy as np
   from megengine.data.dataset import ArrayDataset 

   image_data = np.random.random((100, 3, 32, 32)) # (N, C, H, W)
   image_dataset = ArrayDataset(image_data)

如果你不清楚上面代码的作用，请参考 :ref:`dataset-guide` 。

.. _sequential-sampler-guide:

顺序抽样
~~~~~~~~

使用 :py:class:`~.SequentialSampler` 可对数据集进行顺序抽样：

>>> from megengine.data import SequentialSampler
>>> sampler = SequentialSampler(image_dataset, batch_size=10)
>>> print(len(list(sampler)))
10

如上所示，对含有 100 个样本的数据集，以 10 作为 ``batch_size`` 抽样，可得到 10 批顺序索引。

我们可以将每一批索引的值打印出来：

>>> for batch_id, indices in enumerate(sampler):
...     print(batch_id, indices)
0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
1 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
2 [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
3 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
4 [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
5 [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
6 [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
7 [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
8 [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
9 [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

如果将 ``batch_size`` 修改为 30, 则会得到 4 批顺序索引，最后一批长度为 10: 

>>> sampler = SequentialSampler(image_dataset, batch_size=30)
>>> for batch_id, indices in enumerate(sampler):
...     print(batch_id, len(indices))
0 30
1 30
2 30
3 10

我们可以通过设置 ``drop_last=True`` 丢掉最后一批不完整的索引：

>>> sampler = SequentialSampler(image_dataset, 30, drop_last=True)
>>> for batch_id, indices in enumerate(sampler):
...     print(batch_id, len(indices))
0 30
1 30
2 30

.. note::

   默认情况下，如果用户没有为 ``MapDataset`` 的 ``DataLoader`` 配置抽样器，则会采用如下配置：

   >>> SequentialSampler(dataset, batch_size=1, drop_last=False)

   显然，``batch_size`` 为 1 时等同于逐个遍历数据集中的每个样本。

.. _random-sampler-guide:

无放回随机抽样
~~~~~~~~~~~~~~

使用 :py:class:`~.RandomSampler` 可对数据集进行无放回随机抽样：

>>> from megengine.data import RandomSampler
>>> sampler = RandomSampler(image_dataset, batch_size=10)
>>> for batch_id, indices in enumerate(sampler):
...     print(batch_id, indices)
0 [78, 20, 74, 6, 45, 65, 99, 67, 88, 57]
1 [81, 0, 94, 98, 71, 30, 66, 10, 85, 56]
2 [51, 87, 62, 42, 7, 75, 11, 12, 39, 95]
3 [73, 15, 77, 72, 89, 13, 55, 26, 49, 33]
4 [9, 8, 64, 3, 37, 2, 70, 29, 34, 47]
5 [22, 18, 93, 4, 40, 92, 79, 36, 84, 25]
6 [83, 90, 68, 58, 50, 48, 32, 54, 35, 1]
7 [14, 44, 17, 63, 60, 97, 96, 23, 52, 38]
8 [80, 59, 53, 19, 46, 43, 24, 61, 16, 5]
9 [86, 82, 31, 76, 28, 91, 27, 21, 69, 41]

.. seealso::

   无放回随机抽样也叫简单随机抽样，参考 
   `Simple random sample <https://en.wikipedia.org/wiki/Simple_random_sample>`_

.. _replacement-sampler-guide:

有放回随机抽样
~~~~~~~~~~~~~~
使用 :py:class:`~.ReplacementSampler` 可对数据集进行有放回随机抽样：

>>> from megengine.data import ReplacementSampler 
>>> sampler = ReplacementSampler(image_dataset, batch_size=10)
>>> for batch_id, indices in enumerate(sampler):
...     print(batch_id, indices)
0 [58, 29, 42, 79, 91, 73, 86, 46, 85, 23]
1 [42, 33, 61, 8, 22, 10, 98, 56, 59, 96]
2 [38, 72, 26, 0, 40, 33, 30, 59, 1, 25]
3 [71, 95, 89, 88, 29, 97, 97, 46, 42, 0]
4 [42, 22, 28, 82, 49, 52, 88, 68, 46, 66]
5 [47, 62, 26, 17, 68, 31, 70, 69, 26, 4]
6 [43, 18, 17, 91, 99, 96, 91, 7, 24, 39]
7 [50, 55, 86, 65, 93, 38, 39, 4, 6, 60]
8 [92, 82, 61, 36, 67, 56, 24, 18, 70, 60]
9 [91, 63, 95, 99, 19, 47, 9, 9, 68, 37]

.. _infinite-sampler-guide:

无限抽样
~~~~~~~~

通常数据集在给定 ``batch_size`` 的情况下，只能划分为有限个 ``batch``.
这意味着抽样所能得到的数据批数是有限的，想要重复利用数据，
最常见的做法是循环多个周期 ``epochs`` 来反复遍历数据集：

>>> for epoch in epochs:
>>>     for batch_data in dataloader:

这里的 ``epochs`` 是机器学习算法中一个比较常见的超参数。

但在一些情况下，我们希望能够直接从数据集中无限进行抽样，
因此 MegEngine 提供了 :py:class:`~.Infinite` 包装类：

>>> from megengine.data import Infinite
>>> sampler = Infinite(SequentialSampler(image_dataset, batch_size=10))
>>> sample_queue = iter(sampler)
>>> for step in range(20):
...     indice = next(sample_queue)
...     print(step, indice)
0 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
1 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
2 [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
3 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
4 [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
5 [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
6 [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
7 [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
8 [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
9 [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
10 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
11 [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
12 [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
13 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
14 [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
15 [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
16 [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
17 [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
18 [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
19 [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

``Infinite`` 可以对已有的各类 ``MapSampler`` 进行包装，进而得到一个可无限迭代的批索引列表。

它的实现原理是：当发现当前的批索引列表无法再进行迭代时，表明已经完成一次数据遍历，
此时它会立刻再次调用原来的抽样器形成一个新的批索引列表，以供下一次 ``next`` 调用。

.. seealso::

   可以在官方 `ResNet 
   <https://github.com/MegEngine/Models/blob/master/official/vision/classification/resnet/train.py>`_
   训练代码中找到 ``DataLoader`` 通过无限采样器加载 ImageNet 数据的示例。

自定义 MapSampler 示例
----------------------

.. seealso::

   * :models:`official/vision/detection/tools/utils.py#L67` - ``GroupedRandomSampler``
   * :models:`official/vision/detection/tools/utils.py#L106` - ``InferenceSampler``
   
.. _stream-sampler-guide:

如何使用 StreamSampler
----------------------

这一部分的内容等待添加中...
