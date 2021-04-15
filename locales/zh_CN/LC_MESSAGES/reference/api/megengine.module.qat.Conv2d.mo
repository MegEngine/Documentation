��    G      T              �  g   �  8   �  ^   .  a   �  V   �  G   F  Q   �  I   �  F   *  U   q  4   �  =   �  [   :  T   �  X   �  L   D	  P   �	  b   �	  D   E
  \   �
  R   �
  P   :  b   �  M   �  Z   <  b   �  d   �  L   _  N   �  J   �  I   F  \   �  J   �  7   8  =   p  >   �     �  �   �  P   �  F   �  
     C   (  )   l  ?   �  %   �  H   �     E  *   M  +   x  l   �  Y     M   k  U   �  �     x   �  ?     2   ]  �   �  �   5  o   �  H   C  "   �  Y   �  X   	  4   b     �  =   �  
   �     �  ^     g  a  g   �  8   1  ^   j  a   �  V   +  G   �  Q   �  I     F   f  U   �  4     =   8  [   v  T   �  X   '  L   �  P   �  b      D   �   \   �   R   #!  P   v!  b   �!  M   *"  Z   x"  b   �"  d   6#  L   �#  N   �#  J   7$  I   �$  \   �$  J   )%  7   t%  =   �%  >   �%     )&  �   8&  P   �&  F   '  
   Y'  C   d'  )   �'  ?   �'  %   (  H   8(     �(  *   �(  +   �(  l   �(  Y   M)  M   �)  U   �)  �   K*  x   �*  ?   Y+  2   �+  �   �+  �   q,  o   -  H   -  "   �-  Y   �-  X   E.  4   �.     �.  =   �.  
   -/     8/  ^   >/   :obj:`__init__ <megengine.module.qat.Conv2d.__init__>`\ \(in\_channels\, out\_channels\, kernel\_size\) :obj:`apply <megengine.module.qat.Conv2d.apply>`\ \(fn\) :obj:`apply_quant_activation <megengine.module.qat.Conv2d.apply_quant_activation>`\ \(target\) :obj:`apply_quant_bias <megengine.module.qat.Conv2d.apply_quant_bias>`\ \(target\, inp\, w\_qat\) :obj:`apply_quant_weight <megengine.module.qat.Conv2d.apply_quant_weight>`\ \(target\) :obj:`buffers <megengine.module.qat.Conv2d.buffers>`\ \(\[recursive\]\) :obj:`calc_conv <megengine.module.qat.Conv2d.calc_conv>`\ \(inp\, weight\, bias\) :obj:`calc_conv_qat <megengine.module.qat.Conv2d.calc_conv_qat>`\ \(inp\) :obj:`children <megengine.module.qat.Conv2d.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.qat.Conv2d.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.qat.Conv2d.eval>`\ \(\) :obj:`forward <megengine.module.qat.Conv2d.forward>`\ \(inp\) :obj:`from_float_module <megengine.module.qat.Conv2d.from_float_module>`\ \(float\_module\) :obj:`get_activation_dtype <megengine.module.qat.Conv2d.get_activation_dtype>`\ \(\) :obj:`get_activation_qparams <megengine.module.qat.Conv2d.get_activation_qparams>`\ \(\) :obj:`get_weight_dtype <megengine.module.qat.Conv2d.get_weight_dtype>`\ \(\) :obj:`get_weight_qparams <megengine.module.qat.Conv2d.get_weight_qparams>`\ \(\) :obj:`load_state_dict <megengine.module.qat.Conv2d.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.qat.Conv2d.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.qat.Conv2d.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.qat.Conv2d.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.qat.Conv2d.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.qat.Conv2d.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.qat.Conv2d.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.qat.Conv2d.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.qat.Conv2d.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.qat.Conv2d.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.qat.Conv2d.reset_parameters>`\ \(\) :obj:`set_fake_quant <megengine.module.qat.Conv2d.set_fake_quant>`\ \(enable\) :obj:`set_observer <megengine.module.qat.Conv2d.set_observer>`\ \(enable\) :obj:`set_qconfig <megengine.module.qat.Conv2d.set_qconfig>`\ \(qconfig\) :obj:`state_dict <megengine.module.qat.Conv2d.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.qat.Conv2d.train>`\ \(\[mode\, recursive\]\) :obj:`with_act <megengine.module.qat.Conv2d.with_act>`\ :obj:`with_weight <megengine.module.qat.Conv2d.with_weight>`\ :obj:`zero_grad <megengine.module.qat.Conv2d.zero_grad>`\ \(\) :py:obj:`None` A :class:`~.QATModule` :class:`~.module.Conv2d` with QAT support. Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`. Applies function ``fn`` to all the modules within this module, including itself. Apply weight's observer and fake_quant from ``qconfig`` on ``target``. Attributes Get activation's quantization dtype as the method from ``qconfig``. Get activation's quantization parameters. Get weight's quantization dtype as the method from ``qconfig``. Get weight's quantization parameters. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QATModule` instance converted from a float :class:`~.Module` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Set quantization related configs with ``qconfig``, including observer and fake_quant for weight and activation. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. Use :func:`~.fake_quant_bias` to process ``target``. megengine.module.qat.Conv2d module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.conv.Conv2d`, :class:`megengine.module.qat.module.QATModule` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:49+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`__init__ <megengine.module.qat.Conv2d.__init__>`\ \(in\_channels\, out\_channels\, kernel\_size\) :obj:`apply <megengine.module.qat.Conv2d.apply>`\ \(fn\) :obj:`apply_quant_activation <megengine.module.qat.Conv2d.apply_quant_activation>`\ \(target\) :obj:`apply_quant_bias <megengine.module.qat.Conv2d.apply_quant_bias>`\ \(target\, inp\, w\_qat\) :obj:`apply_quant_weight <megengine.module.qat.Conv2d.apply_quant_weight>`\ \(target\) :obj:`buffers <megengine.module.qat.Conv2d.buffers>`\ \(\[recursive\]\) :obj:`calc_conv <megengine.module.qat.Conv2d.calc_conv>`\ \(inp\, weight\, bias\) :obj:`calc_conv_qat <megengine.module.qat.Conv2d.calc_conv_qat>`\ \(inp\) :obj:`children <megengine.module.qat.Conv2d.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.qat.Conv2d.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.qat.Conv2d.eval>`\ \(\) :obj:`forward <megengine.module.qat.Conv2d.forward>`\ \(inp\) :obj:`from_float_module <megengine.module.qat.Conv2d.from_float_module>`\ \(float\_module\) :obj:`get_activation_dtype <megengine.module.qat.Conv2d.get_activation_dtype>`\ \(\) :obj:`get_activation_qparams <megengine.module.qat.Conv2d.get_activation_qparams>`\ \(\) :obj:`get_weight_dtype <megengine.module.qat.Conv2d.get_weight_dtype>`\ \(\) :obj:`get_weight_qparams <megengine.module.qat.Conv2d.get_weight_qparams>`\ \(\) :obj:`load_state_dict <megengine.module.qat.Conv2d.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.qat.Conv2d.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.qat.Conv2d.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.qat.Conv2d.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.qat.Conv2d.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.qat.Conv2d.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.qat.Conv2d.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.qat.Conv2d.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.qat.Conv2d.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.qat.Conv2d.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.qat.Conv2d.reset_parameters>`\ \(\) :obj:`set_fake_quant <megengine.module.qat.Conv2d.set_fake_quant>`\ \(enable\) :obj:`set_observer <megengine.module.qat.Conv2d.set_observer>`\ \(enable\) :obj:`set_qconfig <megengine.module.qat.Conv2d.set_qconfig>`\ \(qconfig\) :obj:`state_dict <megengine.module.qat.Conv2d.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.qat.Conv2d.train>`\ \(\[mode\, recursive\]\) :obj:`with_act <megengine.module.qat.Conv2d.with_act>`\ :obj:`with_weight <megengine.module.qat.Conv2d.with_weight>`\ :obj:`zero_grad <megengine.module.qat.Conv2d.zero_grad>`\ \(\) :py:obj:`None` A :class:`~.QATModule` :class:`~.module.Conv2d` with QAT support. Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`. Applies function ``fn`` to all the modules within this module, including itself. Apply weight's observer and fake_quant from ``qconfig`` on ``target``. Attributes Get activation's quantization dtype as the method from ``qconfig``. Get activation's quantization parameters. Get weight's quantization dtype as the method from ``qconfig``. Get weight's quantization parameters. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QATModule` instance converted from a float :class:`~.Module` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Set quantization related configs with ``qconfig``, including observer and fake_quant for weight and activation. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. Use :func:`~.fake_quant_bias` to process ``target``. megengine.module.qat.Conv2d module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.conv.Conv2d`, :class:`megengine.module.qat.module.QATModule` 