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
   calibration_qconfig
   easyquant_qconfig
   ema_fakequant_qconfig
   ema_lowbit_fakequant_qconfig
   min_max_fakequant_qconfig
   passive_qconfig
   sync_ema_fakequant_qconfig
   tqt_qconfig

.. py:module:: megengine.quantization.observer
.. currentmodule:: megengine.quantization

观察
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   Observer
   observer.MinMaxObserver
   observer.SyncMinMaxObserver
   observer.ExponentialMovingAverageObserver
   observer.SyncExponentialMovingAverageObserver
   HistogramObserver
   observer.PassiveObserver

.. py:module:: megengine.quantization.fake_quant
.. currentmodule:: megengine.quantization

模拟
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:

   FakeQuantize
   fake_quant.TQT

.. py:module:: megengine.quantization.quantize
.. currentmodule:: megengine.quantization.quantize

量化操作
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   quantize_qat
   quantize

.. py:module:: megengine.quantization.utils
.. currentmodule:: megengine.quantization.utils

Utils
-----
.. autosummary::
   :toctree: api
   :nosignatures:

   QuantMode
