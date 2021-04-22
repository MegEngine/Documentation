.. py:module:: megengine.module
.. currentmodule:: megengine.module

================
模块式（Module）
================

Float Module
------------

基础容器
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Module
   Sequential
   Module.forward
   Module.parameters
   Module.named_parameters
   Module.buffers
   Module.named_buffers
   Module.children
   Module.named_children
   Module.modules
   Module.named_modules
   Module.apply
   Module.train
   Module.eval
   Module.disable_quantize
   Module.state_dict
   Module.load_state_dict

.. py:module:: megengine.module.init
.. currentmodule:: megengine.module.init

初始化
~~~~~~
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

算子支持
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Elemwise
   Concat

卷积层
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Conv1d
   Conv2d
   Conv3d
   ConvRelu2d
   ConvTranspose2d
   LocalConv2d
   DeformableConv2d

池化层
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   AvgPool2d
   MaxPool2d
   AdaptiveAvgPool2d
   AdaptiveMaxPool2d
   DeformablePSROIPooling

非线性激活层
~~~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Sigmoid
   Softmax
   ReLU
   LeakyReLU
   PReLU

归一化层
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   BatchNorm1d
   BatchNorm2d
   SyncBatchNorm
   GroupNorm
   InstanceNorm
   LayerNorm

线性层
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Linear
   Identity

随机失活层
~~~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Dropout

稀疏层
~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Embedding

融合操作
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   ConvBn2d
   ConvBnRelu2d
   BatchMatMulActivation

量化支持
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   QuantStub
   DequantStub

.. py:module:: megengine.module.qat
.. currentmodule:: megengine.module.qat

QAT Module
----------

基础容器
~~~~~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   QATModule
   QATModule.set_qconfig
   QATModule.set_fake_quant
   QATModule.set_observer
   QATModule.apply_quant_weight
   QATModule.apply_quant_activation
   QATModule.apply_quant_bias
   QATModule.get_weight_dtype
   QATModule.get_activation_dtype
   QATModule.get_weight_qparams
   QATModule.get_activation_qparams
   QATModule.from_float_module

算子支持
~~~~~~~~
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

   QuantizedModule
   QuantizedModule.from_qat_module

算子支持
~~~~~~~~
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
   BatchMatMulActivation
   QuantStub
   DequantStub

