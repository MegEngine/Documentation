.. _megengine-for-pytorch-users:

===========================
MegEngine for PyTorch users
===========================

.. note::

   在这个页面，会给出 MegEngine 与 PyTorch 设计和实现上的一些区别。
   如果你在使用过程中发现有其它需要关注的差异点，欢迎在这里补充。

Tensor 数据结构相关
-------------------

.. admonition:: MegEngine Tensor 默认不需要 ``.to(device)`` 操作
   :class: warning

   MegEngine 在设备处理上兼容性良好，默认会将 Tensor 放在算力最高的 :ref:`设备 <tensor-device>` 上。


.. admonition:: MegEngine Tensor 类中没有实现所有操作方法
   :class: warning

   Pytorch 的 Tensor 类中提供了许多操作/计算方法，而在 MegEngine 中这些方法被统一实现在 functional 模块中，
   意味着类似 ``functional.add()`` 等操作并不一定存在着对应的 ``Tensor.add()`` 实现，这是设计上的历史决定。
   当你要对一个 Tensor 进行各种操作时，应当想到使用 :mod:`~.functional` 模块中的接口。

.. admonition:: MegEngine Tensor 默认不计算梯度
   :class: warning

   Pytorch 中默认所有 Tensor 都需要被求导，因此提供了 :py:class:`torch.no_grad` 来禁用梯度计算。
   而在 MegEngine 中 Tensor 默认不需要被求导，需要通过 :py:meth:`megengine.autodiff.GradManager.attach` 来进行绑定，
   被绑定后的 Tensor 可以通过 :py:meth:`megengine.Tensor.detach` 来解除绑定。

Data & DataLoader 相关
----------------------

.. admonition:: DataLoader 中的数据在供给之前，通常不用转 Tensor
   :class: warning

   * MegEngine 中的 :mod:`~.data` 模块默认对 NumPy ndarray 格式的数据进行处理，可视作是独立的模块；
   * 推荐的数据处理流程是：原始数据格式 ->  ndarray （ DataLoader ）
     -> 在每个 Batch 数据中将 DataLoader 供给的 ndarray 数据转 Tensor 格式 -> 后续 Tensor 计算；
   * 如果你选择在 DataLoader 中就将数据处理成 Tensor 格式，在多 GPU + 多 ``worker``
     进程读取的情况下可能会导致 CUDA fork 的初始化错误，更详细的解释请参考 :ref:`data-guide` 。

.. admonition:: Sampler 默认支持分布式情景，不需要像 Torch 一样使用 DDP
   :class: warning

   更多说明，请参考 :ref:`distributed-guide` 中的相关介绍。

