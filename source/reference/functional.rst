.. py:module:: megengine.functional
.. currentmodule:: megengine.functional

====================
megengine.functional
====================

.. code-block:: python

   import megengine.functional as F

   tensor_c = F.add(tensor_a, tensor_b)  # tensor_c = tensor_a + tensor_b

   loss = F.nn.square_loss(pred, label)  # Equals to: F.loss.square_loss()

.. note::

   顾名思义，:mod:`megengine.functional` 模块中包含着所有与 Tensor 有关的计算接口：

   * 与神经网络（Neural Network）相关的算子统一封装在 :mod:`megengine.functional.nn` 中；
   * 分布式算子统一封装在 :mod:`megengine.functional.distributed` 中，方便调用；
   * 其它的常见算子均可在 :mod:`megengine.functional` 中直接调用；

.. _general-tensor-operations:

General tensor operations
-------------------------

.. _tensor-creation:

Tensor creation
~~~~~~~~~~~~~~~
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
   arange
   linspace

.. _tensor-manipulation:

Tensor manipulation
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   copy

.. _changing-tensor-shape:

Changing tensor shape
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   reshape
   flatten

.. _transpose-like:

Transpose-like
^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   transpose

.. _changing-number-of-dimensions:

Changing number of dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   broadcast_to
   expand_dims
   squeeze

.. _joining-splitting-tiling-and-others:
   
Joining, Splitting, Tiling and others
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   concat
   stack
   split
   tile
   repeat
   gather
   scatter
   cond_take

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
   pow
   mod
   sqrt
   square
   abs
   sign
   maximum
   minimum

.. _rounding:

Rounding
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   round
   ceil
   floor
   clip

.. _exponents-and-logarithms:

Exponents and logarithms
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   exp
   expm1
   log
   log1p

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

   logical_and
   logical_not
   logical_or
   logical_xor

.. _comparison:

Comparison
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   isnan
   isinf
   equal
   not_equal
   less
   less_equal
   greater
   greater_equal

.. _sums-products-and-others:

Sums, products and others
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   sum
   prod
   mean
   min
   max

.. _matrix-operations:

Matrix operations
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dot
   matinv
   matmul
   svd

.. _statistics:

Statistics
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   var
   std
   norm
   normalize

.. _sorting-and-searching:

Sorting and searching
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk
   sort
   argsort
   argmin
   argmax
   where

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

.. currentmodule:: megengine.functional.metric

.. _metric-functions:

Metric functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk_accuracy

.. currentmodule:: megengine.functional.loss

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

.. currentmodule:: megengine.functional.vision

.. _vision-functions:

Vision functions
~~~~~~~~~~~~~~~~
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

.. _distributed-functions:

Distributed functions
---------------------
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
