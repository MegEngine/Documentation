.. py:module:: megengine.optimizer
.. currentmodule:: megengine.optimizer

===================
优化器（Optimizer）
===================
.. autosummary::
   :toctree: api
   :nosignatures:

   Optimizer
   Optimizer.step
   Optimizer.clear_grad
   Optimizer.add_param_group
   Optimizer.state_dict
   Optimizer.load_state_dict

常见优化器
----------
.. autosummary::
   :toctree: api
   :nosignatures:

   SGD
   Adam
   Adagrad
   Adadelta

学习率调整
----------
.. autosummary::
   :toctree: api
   :nosignatures:

   LRScheduler
   MultiStepLR

