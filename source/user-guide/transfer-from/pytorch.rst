.. _megengine-for-pytorch-users:

===========================
MegEngine for PyTorch users
===========================

.. warning::

   Pytorch 的 Tensor 类中提供了许多操作/计算方法，而在 MegEngine 中这些方法被统一实现在 functional 模块中，
   意味着类似 ``functional.add()`` 等操作并不一定存在着对应的 ``Tensor.add()`` 实现，这是设计上的历史决定。

.. warning::

   Pytorch 中默认所有 Tensor 都需要被求导，因此提供了 :py:class:`torch.no_grad` 来禁用梯度计算。
   而在 MegEngine 中 Tensor 默认不需要被求导，需要通过 :py:meth:`megengine.autodiff.GradManager.attach` 来进行绑定，
   被绑定后的 Tensor 可以通过 :py:meth:`megengine.Tensor.detach` 来解除绑定。


