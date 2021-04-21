.. py:module:: megengine.functional
.. currentmodule:: megengine.functional

====================
函数式（Functional）
====================

.. py:module:: megengine.functional.tensor
.. currentmodule:: megengine.functional

tensor 子模块
-------------

创建张量
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   zeros
   zeros_like
   ones
   ones_like
   full
   full_like
   arange
   linspace
   eye

处理张量
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   copy
   broadcast_to
   concat
   stack
   split
   gather
   scatter
   where
   cond_take
   transpose
   reshape
   flatten
   repeat
   tile
   expand_dims
   squeeze

.. py:module:: megengine.functional.elemwise
.. currentmodule:: megengine.functional

elemwise 子模块
---------------

基本运算
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
   abs
   exp
   expm1
   log
   log1p
   sqrt
   square
   round
   ceil
   floor
   maximum
   minimum
   clip

三角运算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   cos
   sin
   tan
   acos
   asin
   atan
   atan2
   cosh
   sinh
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

.. py:module:: megengine.functional.math
.. currentmodule:: megengine.functional

math 子模块
-----------

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
   argmin
   argmax

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

条件计算
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   isnan
   isinf
   sign
   sort
   argsort
   topk

.. py:module:: megengine.functional.nn
.. currentmodule:: megengine.functional

nn 子模块
---------
.. autosummary::
   :toctree: api
   :nosignatures:

   nn.nvof

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

.. py:module:: megengine.functional.loss
.. currentmodule:: megengine.functional.loss

loss 子模块
-----------
.. autosummary::
   :toctree: api
   :nosignatures:

   l1_loss
   square_loss
   hinge_loss
   binary_cross_entropy
   cross_entropy

.. py:module:: megengine.functional.metric
.. currentmodule:: megengine.functional.metric

metric 子模块
-------------
.. autosummary::
   :toctree: api
   :nosignatures:

   topk_accuracy

.. py:module:: megengine.functional.distributed
.. currentmodule:: megengine.functional.distributed

distributed 子模块
------------------
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

.. py:module:: megengine.functional.vision
.. currentmodule:: megengine.functional.vision

vision 模块
-----------
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


