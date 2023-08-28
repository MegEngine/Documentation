.. py:module:: megengine.functional
.. currentmodule:: megengine.functional

====================
megengine.functional
====================

.. code-block:: python

   import megengine.functional as F

.. note::

   顾名思义，:mod:`megengine.functional` 模块中包含着所有与 Tensor 有关的计算接口：

   * 与神经网络（Neural Network）相关的算子统一封装在 :mod:`megengine.functional.nn` 中；
   * 分布式算子统一封装在 :mod:`megengine.functional.distributed` 中，方便调用；
   * 其它的常见算子均可在 :mod:`megengine.functional` 中直接调用；

.. seealso::

   用户指南中对于 :ref:`functional-guide` 有另外一套分类逻辑，可作为参考。

.. _general-tensor-operations:

General tensor operations
--------------------------

.. note::

   该部分的 API 设计接纳了 `Python 数据 API 标准联盟 <https://data-apis.org/>`_ 中的倡导,
   尽可能地向 NumPy API 靠拢。 

.. _creation:

Creation Functions
~~~~~~~~~~~~~~~~~~
.. seealso::

   :ref:`tensor-creation`

.. autosummary::
   :toctree: api
   :nosignatures:

   arange
   linspace
   eye
   diag
   zeros
   zeros_like
   ones
   ones_like
   full
   full_like
   complex
   polar

.. _manipulation:

Manipulation Functions
~~~~~~~~~~~~~~~~~~~~~~
.. seealso::

   :ref:`tensor-manipulation`

.. autosummary::
   :toctree: api
   :nosignatures:

   copy
   reshape
   flatten
   transpose
   swapaxes
   broadcast_to
   expand_dims
   squeeze
   concat
   stack
   split
   tile
   repeat
   roll
   real
   imag
   cross

.. _arithmetic-operations:

Arithmetic operations
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   add
   sub
   mul
   div
   floor_div
   neg
   abs
   sign
   pow
   mod
   sqrt
   square
   maximum
   minimum
   round
   ceil
   floor
   clip
   exp
   expm1
   log
   log1p
   logaddexp

.. _trigonometric-functions:

Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~
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

.. _hyperbolic-functions:

Hyperbolic functions
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sinh
   cosh
   tanh
   acosh
   asinh
   atanh

.. _bit-operations:

Bit operations
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   left_shift
   right_shift

.. _logic-functions:

Logic functions
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   isnan
   isinf
   logical_and
   logical_not
   logical_or
   logical_xor
   greater
   greater_equal
   less
   less_equal
   equal
   not_equal

.. _statistical-functions:

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sum
   cumsum
   prod
   mean
   min
   max
   var
   std
   norm
   normalize

.. seealso::

   想要返回 ``min``, ``max`` 的索引而不是元素值，请参考 :ref:`searching-functions` 

.. _linear-algebra-functions:

Linear Algebra Functions
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dot
   matinv
   matmul
   svd

.. _indexing-functions:

Indexing Functions
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   gather
   scatter
   cond_take
   where
   non_zero

.. _searching-functions:

Searching Functions
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   argmin
   argmax

.. _sorting-functions:

Sorting Functions
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sort
   argsort
   topk

.. _metric-functions:

.. TODO: metric functions should could be called in F.nn.xxx format
.. It will be fixed in v1.12 and later version

Metric functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk_accuracy

.. py:module:: megengine.functional.nn
.. currentmodule:: megengine.functional.nn

.. _neural-network-operations:

Neural network operations
-------------------------

.. _convolution-functions:

Convolution functions
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   conv1d
   conv2d
   conv3d
   local_conv2d
   conv_transpose2d
   conv_transpose3d
   deformable_conv2d
   sliding_window
   sliding_window_transpose

.. _pooling-functions:

Pooling functions
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   avg_pool2d
   max_pool2d
   adaptive_avg_pool2d
   adaptive_max_pool2d
   deformable_psroi_pooling

.. _non-linear-activation-functions:

Non-linear activation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
   silu
   gelu
   softplus
   softmax
   logsoftmax
   logsigmoid
   logsumexp

.. _normalization-functions:

Normalization functions
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   batch_norm
   sync_batch_norm
   layer_norm
   local_response_norm

.. _linear-functions:

Linear functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   linear

.. _dropout-functions:

Dropout functions
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dropout

.. _sparse-functions:

Sparse functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   one_hot
   indexing_one_hot
   embedding

.. _loss-functions:

Loss functions
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   l1_loss
   square_loss
   hinge_loss
   binary_cross_entropy
   cross_entropy
   ctc_loss

.. _vision-functions:

Vision functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   cvt_color
   pad
   pixel_shuffle
   interpolate
   remap
   warp_affine
   warp_perspective
   roi_pooling
   roi_align
   nms
   correlation
   nvof

.. py:module:: megengine.functional.quantized
.. currentmodule:: megengine.functional.quantized

.. _quantized-function:

Quantized functions
-------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   conv_bias_activation
   batch_conv_bias_activation
   conv_transpose2d

.. py:module:: megengine.functional.distributed
.. currentmodule:: megengine.functional.distributed

.. _distributed-functions:

Distributed functions
---------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   all_gather
   all_reduce_max
   all_reduce_min
   all_reduce_sum
   all_to_all
   broadcast
   reduce_scatter_sum
   collective_comm
   remote_send
   remote_recv
   reduce_sum
   gather
   scatter
   
.. py:module:: megengine.functional.external
.. currentmodule:: megengine.functional.external

.. _external-functions:

External functions
---------------------
.. autosummary::
   :toctree: api
   :nosignatures:

   extern_opr_subgraph
   tensorrt_runtime_opr
   cambricon_runtime_opr
   atlas_runtime_opr
   magicmind_runtime_opr