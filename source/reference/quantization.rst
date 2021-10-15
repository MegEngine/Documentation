.. py:module:: megengine.quantization
.. currentmodule:: megengine.quantization

======================
megengine.quantization
======================

.. note::

   .. code-block:: python

      import megengine.quantization as Q

      model = ... # The pre-trained float model that needs to be quantified

      Q.quantize_qat(model, qconfig=...) # 

      for _ in range(...):
          train(model)

      Q.quantize(model)

   具体用法说明请参考用户指南页面 —— :ref:`quantization-guide` 。

量化配置 QConfig
----------------
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   QConfig
   
.. _qconfig-list:

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

.. _qconfig-obsever:

Observer
~~~~~~~~
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

FakeQuantize
~~~~~~~~~~~~
.. currentmodule:: megengine.quantization
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   FakeQuantize
   TQT
   LSQ


.. _quantize-operation:

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
