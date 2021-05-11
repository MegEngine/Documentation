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
   :template: autosummary/api-class.rst

   QConfig
   
可用预设配置
~~~~~~~~~~~~

``min_max_fakequant_qconfig``
  使用 :class:`~.MinMaxObserver` 和 :class:`~.FakeQuantize` 预设。

``ema_fakequant_qconfig``
  使用 :class:`~.ExponentialMovingAverageObserver` 和 :class:`~.FakeQuantize` 预设。

``sync_ema_fakequant_qconfig``
  使用 :class:`~.SyncExponentialMovingAverageObserver` 和 :class:`~.FakeQuantize` 的预设。
 
``ema_lowbit_fakequant_qconfig``
  使用 :class:`~.ExponentialMovingAverageObserver` 和 :class:`~.FakeQuantize` 且数值类型为 ``qint4`` 的预设。

``calibration_qconfig``
  对激活值使用 :class:`~.HistogramObserver` 进行后量化（无 :class:`~.FakeQuantize` ）的预设。

``tqt_qconfig``
  使用 :class:`~.TQT` 进行假量化的预设。

``passive_qconfig``
  使用 :class:`~.PassiveObserver` 和 :class:`~.FakeQuantize` 的预设。

``easyquant_qconfig``
  用于 easyquant 算法的 QConfig，等价于 ``passive_qconfig``.


观察
~~~~
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Observer
   MinMaxObserver
   SyncMinMaxObserver
   ExponentialMovingAverageObserver
   SyncExponentialMovingAverageObserver
   HistogramObserver
   PassiveObserver

模拟
~~~~
.. currentmodule:: megengine.quantization
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   FakeQuantize
   TQT


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

Utils
-----
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   QParams
   QuantMode

.. autosummary::
   :toctree: api
   :nosignatures:

   create_qparams
   fake_quant_bias
   fake_quant_tensor
