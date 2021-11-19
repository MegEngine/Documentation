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
   :template: autosummary/api-class.rst

   Module
   Sequential

General operations
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Elemwise
   Concat

Convolution Layers
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

   Pad

Non-linear Activations
~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

   BatchNorm1d
   BatchNorm2d
   SyncBatchNorm
   GroupNorm
   InstanceNorm
   LayerNorm

Linear Layers
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Identity
   Linear

Dropout Layers
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Dropout

Sparse Layers
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Embedding

Fused operations
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
   BatchMatMulActivation

Quantization
~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

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
   :template: autosummary/api-class.rst

   QATModule

Operations
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Linear
   Elemwise
   Concat
   Conv2d
   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
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
   :template: autosummary/api-class.rst

   QuantizedModule

Operations
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Linear
   Elemwise
   Concat
   Conv2d
   ConvRelu2d
   ConvBn2d
   ConvBnRelu2d
   BatchMatMulActivation
   QuantStub
   DequantStub

