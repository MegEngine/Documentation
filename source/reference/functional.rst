.. py:module:: megengine.functional
.. currentmodule:: megengine.functional

====================
megengine.functional
====================

General Operations
------------------

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

Tensor manipulation
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   copy

Changing tensor shape
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   reshape
   flatten

Transpose-like
^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   transpose

Changing number of dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: api
   :nosignatures:

   broadcast_to
   expand_dims
   squeeze
   
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

Rounding
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   round
   ceil
   floor
   clip

Exponents and logarithms
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   exp
   expm1
   log
   log1p

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

Bit operations
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   left_shift
   right_shift

Logic functions
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   logical_and
   logical_not
   logical_or
   logical_xor

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

Matrix operations
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dot
   matinv
   matmul
   svd

Statistics
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   var
   std
   norm
   normalize

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

Neural network operations
-------------------------

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
   deformable_conv2d

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

Normalization functions
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   batch_norm
   sync_batch_norm

Linear functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   linear

Dropout functions
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   dropout

Sparse functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   one_hot
   indexing_one_hot
   embedding

.. currentmodule:: megengine.functional.metric

Metric functions
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   topk_accuracy

.. currentmodule:: megengine.functional.loss

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
