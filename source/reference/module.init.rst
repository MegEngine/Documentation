.. py:module:: megengine.module.init
.. currentmodule:: megengine.module.init

=====================
megengine.module.init
=====================

>>> import megengine.module as M
>>> m = M.Conv2d(16, 33, 3, stride=2)
>>> M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")

.. note::

   良好的初始化策略有助于你的模型在训练时更快地收敛。

Initialization
--------------
.. autosummary::
   :toctree: api
   :nosignatures:

   fill_
   zeros_
   ones_
   uniform_
   normal_
   calculate_gain
   calculate_fan_in_and_fan_out
   calculate_correct_fan
   xavier_uniform_
   xavier_normal_
   msra_uniform_
   msra_normal_

