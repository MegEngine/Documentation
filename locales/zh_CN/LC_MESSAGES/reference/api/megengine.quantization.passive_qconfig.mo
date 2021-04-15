��          t               �   �   �   �   m  	   Q  �  [  _   �  j   @  '   �  7   �  5        A  g  H  �   �  �   P  	   4  �  >  _   �	  j   #
  '   �
  7   �
  5   �
     $   A config class indicating how to do quantize toward :class:`~.QATModule`'s ``activation`` and ``weight``. See :meth:`~.QATModule.set_qconfig` for detail usage. Each parameter is a ``class`` rather than an instance. And we recommand using ``functools.partial`` to add initialization parameters of the ``class``, so that don't need to provide parameters in :meth:`~.QATModule.set_qconfig`. Examples: Usually we choose narrow version dtype (like ``qint8_narrow``) for weight related paramters and normal version for activation related ones. For the result of multiplication and addition as ``a * b + c * d``, if four variables are all -128 of dtype ``qint8``, then the result will be ``2^15`` and cause overflow. Weights are commonly calculated in this way, so need to narrow qmin to -127. interface to instantiate a :class:`~.FakeQuantize` indicating how to do fake_quant calculation. interface to instantiate an :class:`~.Observer` indicating how to collect scales and zero_point of wegiht. megengine.quantization.passive\_qconfig similar to ``weight_fake_quant`` but toward activation. similar to ``weight_observer`` but toward activation. 参数 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:44+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 A config class indicating how to do quantize toward :class:`~.QATModule`'s ``activation`` and ``weight``. See :meth:`~.QATModule.set_qconfig` for detail usage. Each parameter is a ``class`` rather than an instance. And we recommand using ``functools.partial`` to add initialization parameters of the ``class``, so that don't need to provide parameters in :meth:`~.QATModule.set_qconfig`. Examples: Usually we choose narrow version dtype (like ``qint8_narrow``) for weight related paramters and normal version for activation related ones. For the result of multiplication and addition as ``a * b + c * d``, if four variables are all -128 of dtype ``qint8``, then the result will be ``2^15`` and cause overflow. Weights are commonly calculated in this way, so need to narrow qmin to -127. interface to instantiate a :class:`~.FakeQuantize` indicating how to do fake_quant calculation. interface to instantiate an :class:`~.Observer` indicating how to collect scales and zero_point of wegiht. megengine.quantization.passive\_qconfig similar to ``weight_fake_quant`` but toward activation. similar to ``weight_observer`` but toward activation. 参数 