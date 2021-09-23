.. _data-transform-guide:

===========================
使用 Transform 定义数据变换
===========================
.. note::

   对输入数据进行变换（Transformation）是十分常见的操作，尤其是在计算机视觉领域。

   在 :py:mod:`megengine.data.transform` 中提供的各种数据变换都是基于 :py:class:`~.Transform` 抽象类实现的，其中：

   * ``apply`` 抽象方法可用于单个的数据样本， **需要在子类中实现** （ :ref:`下面有举例 <custom-transform-guide>` ）；
   * 各种变换操作可以通过 :py:class:`~.Compose` 进行组合，这样使用起来更加方便。

   我们能够很方便地在 ``DataLoader`` 加载数据时进行相应地变换操作。例如：

   >>> dataloader = DataLoader(dataset, transform=Compose([Resize(32), ToMode('CHW')]))

   更多 API 请参考 :py:mod:`megengine.data.transform` 模块。

.. note::

   借助数据变换，我们可以达成各种目标，包括但不限于：

   * 通过 ``Resize`` 操作，使得输入数据的形状满足模型对形状的要求；
   * 实现 `数据增广 <https://megengine.org.cn/doc/stable/zh/getting-started/beginner/neural-network-traning-tricks.html#%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%B9%BF>`_
     （Data augmentation），更多的数据往往能提升模型的性能...

.. seealso::

   * MegEngine 中提供了大量的 :py:class:`VisionTransform` 实现，用户也可参考 API 文档进行拓展；
   * 一些数据变换的实现参考自 `torchvision <https://pytorch.org/vision/stable/index.html>`_
     以及 `OpenMMLab <https://github.com/open-mmlab>`_ .
   * MegEngine 中也提供了 :py:class:`TorchTransformCompose` 实现，方便使用 ``torchvision`` 中的实现。

.. warning::

   * :ref:`transform-vs-functional`
   * :ref:`when-preprocess-data`

.. _custom-transform-guide:

举例：伪变换和自定义变换
------------------------

:class:`~.PseudoTransform` 的实现非常简单，它没有对输入进行任何处理，而是直接返回：

.. code-block:: python

   class PseudoTransform(Transform):
       def apply(self, input: Tuple):
           return input

我们构造一个数据 ``data`` 进行测试：

>>> data = np.arange(9).reshape(3, 3)
>>> data
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

>>> from megengine.data.transform import PseudoTransform
>>> PseudoTransform().apply(data)
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

如果我们要实现自定义的变换操作，只需要自己实现相应的 ``apply`` 逻辑。

比如我们实现一个 ``AddOneTransform``:

>>> from megengine.data.transform import Transform
>>> class AddOneTransform(Transform):
...     def apply(self, input):
...         return input + 1
>>> AddOneTransform().apply(data)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

可以使用 ``Compose`` 对数据变换进行组合：

>>> from megengine.data.transform import Compose
>>> composed_transform = Compose([AddOneTransform(), AddOneTransform()])
>>> composed_transform.apply(data)
array([[ 2,  3,  4],
       [ 5,  6,  7],
       [ 8,  9, 10]])

.. warning::

   我们这里给出的例子比较简单，实际上 ``apply`` 方法支持 Tuple 类型的输入，
   代码逻辑中完全可以处理更加一些复杂的样本结构，可以参考 :py:class:`VisionTransform` 的实现。

.. seealso::

   可以在官方 `ResNet 
   <https://github.com/MegEngine/Models/blob/master/official/vision/classification/resnet/train.py>`_
   训练代码中找到 ``DataLoader`` 通过组合数据变换对数据进行预处理的例子。

.. _transform-vs-functional:

与 Functional 的区别
--------------------

用户不应当将 ``megengine.data.transform`` 与 ``megengine.functional`` 中的接口搞混淆：

* ``megengine.data.transform`` 可以看作是一个独立的子库，可以对 NumPy 的 ndarray 数据进行各种处理；
* ``megengine.functional`` 中的实现都是围绕着 MegEngine 的 Tensor 数据结构进行的。

从流程上看，用户可以将原始数据转换成 ndarray 作为输入，经过 ``megengine.data.transform`` 做一些处理。
如果需要参与模型训练，得到的结果需要人为地转换成 Tensor 才能够被用于 ``megengine.functional`` 中的接口。

.. _when-preprocess-data:

数据预处理应该在何时发生
------------------------

当我们从 ``DataLoader`` 中获取批数据时，如果定义了 ``Transform``, 则会在每次加载完样本后立即对其进行变换。
数据变换操作也是有计算开销的，且该流程通常在 CPU 设备上进行，以及有些操作会调用类似 ``OpenCV`` 的库。
如果我们对每个样本进行多次加载（比如训练多个周期），那么变换操作也会被执行多次，这可能会带来额外的开销。
因此在有些时候，我们会选择将预处理操作在更早的流程中进行，即直接对原始数据先进行一次预处理操作，
这样在 ``DataLoader`` 中获取的输入便已经是经过预处理的数据了，这样可以尽可能地减少 ``Transform`` 操作。

用户应当考虑到，原始数据相关的 I/O 和处理也有可能成为模型训练整体流程中的瓶颈。
