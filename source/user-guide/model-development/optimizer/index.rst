.. _optimizer-guide:

=======================
使用 Optimizer 优化参数
=======================

MegEngine 的 :py:mod:`optimizer` 模块中实现了大量的优化算法，
提供了包括 :py:class:`~.SGD`, :py:class:`~.Adam` 在内的常见优化器。
这些优化器能够基于参数的梯度信息，按照算法定义的策略对参数执行更新。

:py:class:`~.Optimizer` 是所有优化器的基类，优化指定模型 ``model`` 中参数的基本流程如下：

.. code-block:: python

   from megengine.autodiff import GradManager
   import megengine.optimizer as optim

   model = MyModel()
   gm = GradManager.attach(model.parameters())
   optimizer = optim.SGD(model.parameters())  # or other optimizers

   for data, label in dataset:
       with gm:
           pred = model(data)
           loss = loss_fn(pred, label)
           gm.backward()
           optimizer.step().clear_grad()

* 通过执行 :py:meth:`~.Optimizer.step` 方法，参数将被进行一次优化；
* 通过执行 :py:meth:`~.Optimizer.clear_grad` 方法，将清空参数的梯度。

.. admonition:: 为何需要手动清空梯度？
   :class: warning

   梯度管理器执行 :py:meth:`~.GradManager.backward` 方法时，
   会将当前计算所得到的梯度以累加的形式积累到原有梯度上，而不是直接做替换。
   因此对于新一轮的梯度计算，通常需要将上一轮计算得到的梯度信息清空。
   何时进行梯度清空是由人为控制的，这样可允许灵活进行梯度的累积。


了解更多
--------

.. toctree::
   :maxdepth: 1

   advanced-parameter-optimization
