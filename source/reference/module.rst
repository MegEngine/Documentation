.. py:module:: megengine.module
.. currentmodule:: megengine.module

================
megengine.module
================

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

.. py:module:: megengine.module.init
.. currentmodule:: megengine.module.init

Initialization
~~~~~~~~~~~~~~
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

.. currentmodule:: megengine.module

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

