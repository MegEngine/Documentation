.. _optimizer-guide:

=======================
使用 Optimizer 优化参数
=======================
MegEngine 的 :py:mod:`optimizer` 模块中实现了大量的优化算法，
其中 :py:class:`~.Optimizer` 是所有优化器的抽象基类，规定了必须提供的接口。
同时为用户提供了包括 :py:class:`~.SGD`, :py:class:`~.Adam` 在内的常见优化器实现。
这些优化器能够基于参数的梯度信息，按照算法所定义的策略对参数执行更新。

以 ``SGD`` 优化器为例，优化神经网络模型参数的基本流程如下：

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

* 我们需要构造一个优化器，并且传入需要被优化的参数 ``Parameter`` 或其迭代；
* 通过执行 :py:meth:`~.Optimizer.step` 方法，参数将基于梯度信息被进行一次优化；
* 通过执行 :py:meth:`~.Optimizer.clear_grad` 方法，将清空参数的梯度。

.. admonition:: 为何需要手动清空梯度？
   :class: warning

   梯度管理器执行 :py:meth:`~.GradManager.backward` 方法时，
   会将当前计算所得到的梯度以累加的形式积累到原有梯度上，而不是直接做替换。
   因此对于新一轮的梯度计算，通常需要将上一轮计算得到的梯度信息清空。
   何时进行梯度清空是由人为控制的，这样可允许灵活进行梯度的累积。

Optimizer 状态字典
------------------

``Optimizer`` 构造函数中还可接受一个含有优化器默认参数的字典（如含有学习率、动量、权重衰减系数等等），
这些信息可以通过 :py:meth:`~.Optimizer.state_dict` 和 :py:meth:`~.Optimizer.load_state_dict` 获取和加载。

子类在实现时可自定义这些参数，同样以 ``SGD`` 为例：

>>> model = megengine.module.Linear(3, 2)
>>> optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
>>> optimizer.state_dict()
{'param_groups': [{'lr': 0.001,
   'momentum': 0.9,
   'weight_decay': 0.0001,
   'params': [0, 1]}],
 'state': {0: {'momentum_buffer': array([0., 0.], dtype=float32)},
  1: {'momentum_buffer': array([[0., 0., 0.],
          [0., 0., 0.]], dtype=float32)}}}

大多数的 Optimizer 状态字典中会存储参数梯度的统计信息（例如运行时均值、反差等），
在暂停/恢复模型训练时，这些信息需要被保存/加载，以保证前后状态的一致性。

.. seealso::

   通过 :py:meth:`~.Optimizer.load_state_dict` 我们可以加载 ``Optimizer`` 状态字典，常用于模型训练过程的保存与加载。

   * ``Module`` 中也有用于保存和加载的状态字典，参考 :ref:`module-guide` 。
   * 关于模型训练过程中保存与加载的最佳实践，请参考 :ref:`serialization-guide` 。

了解更多
--------

.. toctree::
   :maxdepth: 1

   advanced-parameter-optimization
