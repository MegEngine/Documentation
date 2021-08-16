.. _dataset-guide:

=======================
使用 Dataset 定义数据集
=======================
这个世界上的数据集五花八门，并且总是以各种格式分布在不同地方，
或许一些数据集的存储和组织形式已经成为了参考标准，但并不是所有数据一开始都用 MegEngine 所支持格式的进行存储，
很多时候需要我们写脚本借助一些库或框架对原始数据进行处理，并且转换成 MegEngine 中可用的数据集对象。

在 :ref:`tensor-creation` 小节中，我们提到了 ndarray 是 Python 数据科学社区中较为通用支持的格式，
因此 MegEngine 中的数据集相关操作均以 NumPy 的 :py:class:`~.numpy.ndarray` 作为处理对象，
同时不会产生 Tensor 类型的输出结果，因此在后续如果进行 Tensor 计算时，则需要 :ref:`create-tensor-from-ndarray` 。

.. note::

   * :py:class:`~.DataLoader` 初始化时必须提供 ``dataset`` 参数，通过传入一个数据集对象，表明将要加载的数据；
   * MegEngine 中可以 :ref:`megengine-dataset` （如 :py:class:`~.PascalVOC`,  :py:class:`~.ImageNet` 等）
     替用户完成一些主流数据集的获取、切分等工作。但一些时候这些实现不能满足需求，或者我们需要使用自己采集和标注好的的数据集，
     因此在使用 ``DataLoader`` 之前，通常需要将要用到的数据集人为地封装。

.. _dataset-type:

数据集类型
----------

根据样本的访问方式， MegEngine 中的数据集类型可分为 Map-style 和 Iterable-style 两种：

.. list-table:: 

   * - 类型 
     - :ref:`map-style-dataset`
     - :ref:`iterable-style-dataset`
   * - 抽象基类 [1]_
     - :py:class:`~.dataset.Dataset` / :py:class:`~.dataset.ArrayDataset`
     - :py:class:`~.dataset.StreamDataset`
   * - 访问方式
     - 支持随机
     - 顺序迭代
   * - 适用情景
     - Numpy 数组、字典、磁盘文件 [2]_
     - 生成器、来自网络的流数据

.. [1] 这些类不可直接实例化使用，对应类型的数据集必须自定义子类进行继承，并实现必需的协议。
.. [2] 通常应该尽可能使用 Map-style 的数据集。它提供了查询数据集的大小的方法，更容易打乱，并能够轻松地并行加载。
       但 ``MapDataset`` 不太适合输入数据作为流的一部分到达的情况，例如音频或视频源；
       或者每个数据点可能是文件的一个子集，该文件太大而无法保存在内存中，因此需要在训练期间进行增量加载。
       虽然这些情况可以通过往 Map-style 数据集中加入更复杂的逻辑或在输入时进行额外的预处理来解决，
       但现在有一个更自然的解决方案，即使用 Iterable-style 的 ``StreamDataset`` 作为输入。

.. _map-style-dataset:

Map-style
~~~~~~~~~

:py:class:`~.dataset.Dataset` （也可被称为 ``MapDataset`` ）
  MegEngine 中所有数据集的抽象基类。
  对应数据集类型为 Map-style, 即表示从索引/键到数据样例的映射，具有随机访问能力。
  例如使用 ``dataset[idx]`` 可以从磁盘上的文件夹中读取到第 ``idx`` 个图像及其相应的标签。
  使用时需要实现 ``__getitem__()`` 和 ``__len__()`` 协议。

  下面的代码展示了如何生成一个由 0 到 5 这五个数组成的数据集（不带标签）：

  .. code-block:: python

     from megengine.data.dataset import Dataset

     class CustomMapDataset(Dataset):
         def __init__(self, data):
             self.data = data

         def __getitem__(self, idx):
             return self.data[idx]

         def __len__(self):
             return len(self.data)

  >>> data = list(range(0, 5))
  >>> map_dataset = CustomMapDataset(data)
  >>> print(len(map_dataset))
  >>> print(map_dataset[2])
  5
  2

  .. warning::

     * 请注意，为了避免在加载大型数据集时试图一次性将数据加载到内存而导致 OOM（Out Of Memory），
       我们建议将实际的数据读取操作实现在 ``__getitem__`` 方法中，而不是 ``__init__`` 方法中，
       后者仅记录映射关系中的索引/键内容（可能是文件名或路径组成的列表），这可以极大程度地减少内存占用。
       具体的例子可参考 :ref:`load-image-data-example` ；
     * 但情况并不总是唯一的。如果我们的数据集规模比较小，可以常驻在内存中，
       那么就可以考虑在初始化对象时就加载好整个数据集，减少反复从硬盘或其它位置读取数据到内存的次数。
       例如在不同的 Epoch 中，同一个样本会被用来训练多次，此时从内存中直接读取会更加高效。

:py:class:`~.dataset.ArrayDataset`
  对 ``Dataset`` 类的进一步封装，适用于 NumPy ndarray 数据，无需实现 ``__getitem__()`` 和 ``__len__()`` 协议。

  下面的代码展示了如何生成随机一个具有 100 个样本，每张样本为 32 x 32 像素的 RGB 图片的数据集（标签为随机值）
  这也是我们在处理图像时经常遇到的 ``(N, C, H, W)`` 格式：

  .. code-block:: python

     import numpy as np
     from megengine.data.dataset import ArrayDataset

     data = np.random.random((100, 3, 32, 32))
     target = np.random.random((100, 1))
     dataset = ArrayDataset(data, target)

  >>> print(len(dataset))
  >>> print(type(dataset[0]), len(dataset[0])) 
  >>> print(dataset[0][0].shape)
  100
  <class 'tuple'> 2
  (3, 32, 32)

.. _iterable-style-dataset:
	
Iterable-style
~~~~~~~~~~~~~~

:py:class:`~.dataset.StreamDataset` （也可被称为 ``IterableDataset`` ）
  Iterable-style 数据集，适用于流式数据，即迭代式地访问数据，
  例如使用 ``iter(dataset)`` 可以返回从数据库、远程服务器甚至实时生成的日志中读取的数据流，
  ``DataLoader`` 将使用 ``next`` 对数据进行迭代。

  这种类型的数据集特别适用于：

  * 随机读取成本过高，或者不提供随机访问途径的情况；
  * 批量大小实际取决于获取的数据的情况。

  使用时需要实现 ``__iter__()`` 协议。

  下面的代码展示了如何生成一个由 0 到 5 这五个数组成的数据集（不带标签）：

  .. code-block:: python

     from megengine.data.dataset import StreamDataset

     class CustomIterableDataset(StreamDataset):
         def __init__(self, data):
             self.data = data

         def __iter__(self):
             return iter(self.data)

  >>> data = list(range(0, 5))
  >>> iter_dataset = CustomIterableDataset(data)
  >>> it = iter(iter_dataset)
  >>> print(type(it))
  list_iterator
  >>> print(next(it))
  0
  >>> print(next(it))
  1

  显然，流式数据集不支持获取长度以及通过索引值进行随机访问：

  >>> iterable_dataset[0]
  AssertionError: can not get item from StreamDataset by index

  >>> len(iterable_dataset)
  AssertionError: StreamDataset does not have length

在 DataLoader 中的区别
~~~~~~~~~~~~~~~~~~~~~~
.. panels::
   :container: +full-width
   :card:

   Map-style
   ^^^^^^^^^
   >>> for data in map_dataset:
   ...     print(data)
   0
   1
   2
   3
   4
   ---
   Iterable-style
   ^^^^^^^^^^^^^^
   >>> for data in iter_dataset:
   ...     print(data)
   0
   1
   2
   3
   4

根据上面的例子可以发现，使用相同的原始 List 数据来生成两种类型的数据集并迭代访问，
Map-style 和 Iterable-style 的数据集都能够返回相同的结果，那么区别在哪里呢？
从高层视角看，每次 ``DataLoader`` 从 Map-style 数据集中返回批数据时，
它都会先对数据索引进行采样得到一批索引 ``idx``, 并使用 ``map_dataset[idx]`` 获得批数据.
相反，对于 Iterable-style 数据集，``DataLoader`` 不断地调用 ``next(it)`` 来顺序获取下一个数据，
直到它获取到一个完整的批次。这也是为什么我们说 Iterable-style 数据集更适合将数据提供给顺序模型。

.. seealso::

   参考 :ref:`data-sampler-guide` ，了解如何从样本容量为 N 的数据集中得到长度为 B 的一批索引。

.. _megengine-dataset:

使用已经实现的数据集接口
------------------------

在 :py:mod:`~.data.dataset` 子模块中，除了提供了一些抽象基类待用户自定义子类进行实现，
还提供了一些基于主流数据集封装好的接口，比如常被用于教学和练习用途的 :py:class:`~.MNIST` 数据集：

>>> from megengine.data.dataset import MNIST
>>> train_set = MNIST(root="path/to/data/", train=True, download=False)
>>> test_set = MNIST(root="path/to/data/", train=False, download=False)

借助于封装好的接口，我们可以快速的获取 MNIST 数据集的训练集 ``train_set`` 和测试集 ``test_set`` ，
其中 ``download`` 参数可以控制是否要从数据集官方提供的地址进行下载。更多细节请参考 API 文档。

.. warning::

   这些数据集都是从它们自己的官方站点进行下载的，MegEngine 不提供镜像或加速服务。

.. note::

   * 一些数据集由于许可协议中的规定将不提供原始数据的下载接口（如 :py:class:`~.ImageNet` ），需手动下载；
   * 下载速度受到网络环境和带宽的影响，用户也可以选择使用其它的脚本或工具下载原始数据集；
   * 这些数据集接口源码是非常不错的参考，对于帮助用户学习如何设计数据集接口会很有帮助。

.. _how-to-add-datasets:

如何添加新的数据集
------------------

目前 MegEngine 中提供了一些常见的主流数据集接口，也欢迎用户为我们提供更多的接口实现。

但目前我们还没有提供明确的设计规格和要求，因此建议用户先尝试与官方维护人员进行交流。
