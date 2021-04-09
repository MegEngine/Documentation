.. py:module:: megengine.quantization
.. currentmodule:: megengine.quantization

====================
量化（Quantization）
====================

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

观察
~~~~
.. currentmodule:: megengine.quantization.observer
.. autosummary::
   :toctree: api
   :nosignatures:

   Observer
   MinMaxObserver
   SyncMinMaxObserver
   ExponentialMovingAverageObserver
   SyncExponentialMovingAverageObserver
   HistogramObserver
   PassiveObserver

模拟
~~~~
.. currentmodule:: megengine.quantization.fake_quant
.. autosummary::
   :toctree: api
   :nosignatures:

   FakeQuantize
   TQT


量化操作
--------
.. currentmodule:: megengine.quantization
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
