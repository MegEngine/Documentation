.. py:module:: megengine.quantization
.. currentmodule:: megengine.quantization

====================
量化（Quantization）
====================

.. py:module:: megengine.quantization.qconfig
.. currentmodule:: megengine.quantization

量化配置
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   QConfig
   min_max_fakequant_qconfig
   ema_fakequant_qconfig
   sync_ema_fakequant_qconfig
   ema_lowbit_fakequant_qconfig
   calibration_qconfig
   tqt_qconfig
   passive_qconfig
   easyquant_qconfig

.. py:module:: megengine.quantization.observer
.. currentmodule:: megengine.quantization

观察
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Observer

.. currentmodule:: megengine.quantization.observer

.. autosummary::
   :toctree: api
   :nosignatures:

   MinMaxObserver
   SyncMinMaxObserver
   ExponentialMovingAverageObserver
   SyncExponentialMovingAverageObserver
   HistogramObserver
   PassiveObserver

.. py:module:: megengine.quantization.fake_quant
.. currentmodule:: megengine.quantization

模拟
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   FakeQuantize
   
.. currentmodule:: megengine.quantization.fake_quant

.. autosummary::
   :toctree: api
   :nosignatures:

   TQT

.. py:module:: megengine.quantization.quantize
.. currentmodule:: megengine.quantization

量化操作
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   quantize_qat
   quantize
   apply_easy_quant
   enable_fake_quant
   disable_fake_quant
   enable_observer
   disable_observer
   propagate_qconfig
   reset_qconfig

.. py:module:: megengine.quantization.utils
.. currentmodule:: megengine.quantization

Utils
-----
.. autosummary::
   :toctree: api
   :nosignatures:

   QParams
   QuantMode
   create_qparams
   fake_quant_bias
   fake_quant_tensor
