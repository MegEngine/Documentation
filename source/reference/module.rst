.. py:module:: megengine.module
.. currentmodule:: megengine.module

================
megengine.module
================

>>> import megengine.module as M

.. seealso::

   * 关于 Module 的使用案例，请参考 :ref:`module-guide` ；
   * 关于如何进行模型量化以及几类 Module 的转换原理，请参考 :ref:`quantization-guide` 。

.. _float-module:

Float Module
------------

Containers
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Module
   Sequential

General operations
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Elemwise
   Concat

Convolution Layers
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Conv1d
   Conv2d
   Conv3d
   ConvTranspose2d
   ConvTranspose3d
   LocalConv2d
   DeformableConv2d
   SlidingWindow
   SlidingWindowTranspose

Pooling layers
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   AvgPool2d
   MaxPool2d
   AdaptiveAvgPool2d
   AdaptiveMaxPool2d
   DeformablePSROIPooling

Padding layers
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Pad

Non-linear Activations
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Sigmoid
   Softmax
   ReLU
   LeakyReLU
   PReLU
   SiLU
   GELU

Normalization Layers
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   BatchNorm1d
   BatchNorm2d
   SyncBatchNorm
   GroupNorm
   InstanceNorm
   LayerNorm
   LocalResponseNorm

Recurrent Layers
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   RNN
   RNNCell
   LSTM
   LSTMCell

Linear Layers
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Identity
   Linear

Dropout Layers
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Dropout

Sparse Layers
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Embedding

Vision Layers
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   PixelShuffle
   Remap
   GaussianBlur
   
.. autosummary::
   :toctree: api
   :nosignatures:

Fused operations
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
   BatchMatMulActivation

Quantization
~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   QuantStub
   DequantStub

.. py:module:: megengine.module.qat
.. currentmodule:: megengine.module.qat

.. _qat-module:

QAT Module
----------

Containers
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   QATModule

Operations
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Linear
   Elemwise
   Concat
   Conv2d
   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
   ConvTranspose2d
   BatchMatMulActivation
   QuantStub
   DequantStub

.. py:module:: megengine.module.quantized
.. currentmodule:: megengine.module.quantized

.. _quantized-module:

Quantized Module
----------------
.. autosummary::
   :toctree: api
   :nosignatures:

   QuantizedModule

Operations
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Linear
   Elemwise
   Concat
   Conv2d
   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
   ConvTranspose2d
   BatchMatMulActivation
   QuantStub
   DequantStub


.. py:module:: megengine.module.external
.. currentmodule:: megengine.module.external

External Layers
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api
   :nosignatures:

   ExternOprSubgraph
   TensorrtRuntimeSubgraph
   CambriconRuntimeSubgraph
   AtlasRuntimeSubgraph
   MagicMindRuntimeSubgraph
