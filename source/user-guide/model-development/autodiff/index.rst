.. _autodiff-guide:

=======================
Autodiff 基本原理与使用
=======================

在深度学习领域，任何复杂的深度神经网络模型本质上都可以用一个计算图表示出来。

计算图中存在前向计算和反向计算的过程。在训练模型参数的过程中，需要根据 `反向传播 <https://en.wikipedia.org/wiki/Backpropagation>`_ （Backpropagation）算法，使用链式法则逐层计算参数关于损失函数的梯度。
作为一类深度学习框架，MegEngine 实现了自动微分机制，能够在反向传播的过程中自动地计算、维护 Tensor 的梯度信息。
这一机制也可以被用于其它科学计算情景。

在这一小节，我们将介绍 Tensor 的梯度（Gradient）属性，以及如何借助 :py:mod:`autodiff` 模块来完成相应工作。

梯度与梯度管理器
----------------

我们以下面这个最简单的一次运算 :math:`y=w*x+b` 作为举例：

>>> from megengine import Tensor
>>> x = Tensor([3.])
>>> w = Tensor([2.])
>>> b = Tensor([-1.])
>>> y = w * x + b

Tensor 具有 :attr:`~.Tensor.grad` （即梯度）属性，用来记录梯度信息，可在需要用到梯度参与计算的情景下使用。

在默认情况下，MegEngine 中的 Tensor 计算是不记录梯度信息的：

>>> print(x.grad)
None

如果希望对 Tensor 的梯度进行管理，需要用到 :py:class:`~.GradManager`, 其通过反向模式进行自动微分：

>>> from megengine.autodiff import GradManager
>>> with GradManager() as gm:
...      gm.attach(x)
...      y = w * x + b
...      gm.backward(y)  # dy/dx = w
>>> x.grad
Tensor([2.], device=xpux:0)  

在上面的代码中， ``with`` 语句中的操作历史都会被梯度管理器记录下来。
我们使用 :py:meth:`~.GradManager.attach` 方法来绑定需要被跟踪的 Tensor（例子中为 ``x`` ），然后执行计算；
使用 :py:meth:`~.GradManager.backward` 方法来计算所有已绑定的 Tensor 对于给定的 ``y`` 的梯度，
并将其累加到对应 Tensor 的 ``grad`` 属性，并在这个过程中释放资源。

.. seealso::

   * 可以通过 :py:meth:`.Tensor.detach` 方法来返回一个解绑后的 Tensor.
   * 可以通过 :py:meth:`~.GradManager.attached_tensors` 接口来查询当前梯度管理器中绑定的 Tensor.

   我们还可以使用 :py:meth:`~.GradManager.record` 和 :py:meth:`~.GradManager.release` 来代替 ``with`` 语义，
   更多用法说明可参考 :py:class:`~.GradManager` API 文档。

神经网络编程示例
----------------

在训练神经网络时，我们可以借助梯度管理器来进行反向传播计算，得到参数的梯度信息：

.. code-block:: python

   gm = GradManager()
   gm.attach(model.parameters())

   for data in dataset:
       with gm:
           loss = model(data)
           gm.backward(loss)

更完整的使用示例可以在 MegEngine 文档的新手入门教程中找到。
