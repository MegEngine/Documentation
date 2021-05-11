.. py:module:: megengine.functional
.. currentmodule:: megengine.functional

====================
函数式（Functional）
====================

基础算子
--------

创建张量
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   eye
   zeros
   zeros_like
   ones
   ones_like
   full
   full_like

数值范围 
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:
  
   arange
   linspace

处理张量
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   copy

改变张量形状
^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   reshape
   flatten

转置操作
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   transpose

改变张量维度
^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   broadcast_to
   expand_dims
   squeeze
   
拼接张量
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   concat
   stack

切割张量
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   split

存取张量
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   gather
   scatter
   cond_take

重复张量
^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   tile
   repeat

算术运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   add
   sub
   mul
   div
   floor_div
   neg
   pow
   mod

舍入运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   round
   ceil
   floor
   clip

指数与对数运算
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   exp
   expm1
   log
   log1p

三角运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sin
   cos
   tan
   asin
   acos
   atan
   atan2

双曲函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sinh
   cosh
   tanh
   acosh
   asinh
   atanh

位运算
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   left_shift
   right_shift

逻辑运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   isnan
   isinf
   logical_and
   logical_not
   logical_or
   logical_xor

比较运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   equal
   not_equal
   less
   less_equal
   greater
   greater_equal

归约计算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sum
   prod
   mean
   min
   max

线性代数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dot
   matinv
   matmul
   svd

概率统计
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   var
   std
   norm
   normalize

排序与搜索
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk
   sort
   argsort
   argmin
   argmax
   where

杂项
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sqrt
   square
   abs
   sign
   maximum
   minimum

.. py:module:: megengine.functional.nn
.. currentmodule:: megengine.functional.nn

神经网络算子
------------

卷积函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   conv1d
   conv2d
   conv3d
   local_conv2d
   conv_transpose2d
   deformable_conv2d

池化函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   avg_pool2d
   max_pool2d
   adaptive_avg_pool2d
   adaptive_max_pool2d
   deformable_psroi_pooling

非线性激活函数
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sigmoid
   hsigmoid
   hswish
   relu
   relu6
   prelu
   leaky_relu
   softplus
   softmax
   logsoftmax
   logsigmoid
   logsumexp

归一化函数
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   batch_norm
   sync_batch_norm

线性函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   linear

随机失活函数
~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dropout

稀疏函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   one_hot
   indexing_one_hot
   embedding

.. currentmodule:: megengine.functional.loss

损失函数
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   l1_loss
   square_loss
   hinge_loss
   binary_cross_entropy
   cross_entropy

.. currentmodule:: megengine.functional.metric

评估指标
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk_accuracy

.. currentmodule:: megengine.functional.vision

计算机视觉
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   cvt_color
   interpolate
   remap
   warp_affine
   warp_perspective
   roi_pooling
   roi_align
   nms
   nvof

.. py:module:: megengine.functional.distributed
.. currentmodule:: megengine.functional.distributed

分布式算子
----------
.. autosummary::
   :toctree: api
   :nosignatures:

   all_reduce_max
   all_reduce_min
   all_reduce_sum
   broadcast
   remote_send
   remote_recv
   all_gather
   all_to_all
   reduce_sum
   gather
   scatter
   reduce_scatter_sum
