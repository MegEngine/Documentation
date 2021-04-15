��    E      D              l  y   m  G   �  m   /  p   �  e     V   t  U   �  d   !  C   �  L   �  j     c   �  g   �  [   N	  _   �	  q   

  S   |
  k   �
  a   <  _   �  q   �  \   p  i   �  q   7  s   �  [     ]   y  Y   �  X   1  k   �  Y   �  F   P  L   �  M   �     2  P   A  P   �  F   �  
   *  C   5  )   y  ?   �  %   �  H   	     R  *   Z  +   �  l   �  Y     M   x  U   �  �     x   �  ?   *  2   j  �   �  �   B  o   �  H   P  "   �  Y   �  X     4   o  *   �  =   �  
          �     g  �  y     G   �  m   �  p   7  e   �  V     U   e  d   �  C      L   d  j   �  c      g   �   [   �   _   D!  q   �!  S   "  k   j"  a   �"  _   8#  q   �#  \   
$  i   g$  q   �$  s   C%  [   �%  ]   &  Y   q&  X   �&  k   $'  Y   �'  F   �'  L   1(  M   ~(     �(  P   �(  P   ,)  F   })  
   �)  C   �)  )   *  ?   =*  %   }*  H   �*     �*  *   �*  +   +  l   K+  Y   �+  M   ,  U   `,  �   �,  x   K-  ?   �-  2   .  �   7.  �   �.  o   z/  H   �/  "   30  Y   V0  X   �0  4   	1  *   >1  =   i1  
   �1     �1  �   �1   :obj:`__init__ <megengine.module.qat.BatchMatMulActivation.__init__>`\ \(batch\, in\_features\, out\_features\[\, ...\]\) :obj:`apply <megengine.module.qat.BatchMatMulActivation.apply>`\ \(fn\) :obj:`apply_quant_activation <megengine.module.qat.BatchMatMulActivation.apply_quant_activation>`\ \(target\) :obj:`apply_quant_bias <megengine.module.qat.BatchMatMulActivation.apply_quant_bias>`\ \(target\, inp\, w\_qat\) :obj:`apply_quant_weight <megengine.module.qat.BatchMatMulActivation.apply_quant_weight>`\ \(target\) :obj:`buffers <megengine.module.qat.BatchMatMulActivation.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.qat.BatchMatMulActivation.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.qat.BatchMatMulActivation.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.qat.BatchMatMulActivation.eval>`\ \(\) :obj:`forward <megengine.module.qat.BatchMatMulActivation.forward>`\ \(inp\) :obj:`from_float_module <megengine.module.qat.BatchMatMulActivation.from_float_module>`\ \(float\_module\) :obj:`get_activation_dtype <megengine.module.qat.BatchMatMulActivation.get_activation_dtype>`\ \(\) :obj:`get_activation_qparams <megengine.module.qat.BatchMatMulActivation.get_activation_qparams>`\ \(\) :obj:`get_weight_dtype <megengine.module.qat.BatchMatMulActivation.get_weight_dtype>`\ \(\) :obj:`get_weight_qparams <megengine.module.qat.BatchMatMulActivation.get_weight_qparams>`\ \(\) :obj:`load_state_dict <megengine.module.qat.BatchMatMulActivation.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.qat.BatchMatMulActivation.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.qat.BatchMatMulActivation.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.qat.BatchMatMulActivation.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.qat.BatchMatMulActivation.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.qat.BatchMatMulActivation.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.qat.BatchMatMulActivation.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.qat.BatchMatMulActivation.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.qat.BatchMatMulActivation.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.qat.BatchMatMulActivation.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.qat.BatchMatMulActivation.reset_parameters>`\ \(\) :obj:`set_fake_quant <megengine.module.qat.BatchMatMulActivation.set_fake_quant>`\ \(enable\) :obj:`set_observer <megengine.module.qat.BatchMatMulActivation.set_observer>`\ \(enable\) :obj:`set_qconfig <megengine.module.qat.BatchMatMulActivation.set_qconfig>`\ \(qconfig\) :obj:`state_dict <megengine.module.qat.BatchMatMulActivation.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.qat.BatchMatMulActivation.train>`\ \(\[mode\, recursive\]\) :obj:`with_act <megengine.module.qat.BatchMatMulActivation.with_act>`\ :obj:`with_weight <megengine.module.qat.BatchMatMulActivation.with_weight>`\ :obj:`zero_grad <megengine.module.qat.BatchMatMulActivation.zero_grad>`\ \(\) :py:obj:`None` A :class:`~.QATModule` :class:`~.module.BatchMatMulActivation` with QAT support. Applies function ``fn`` to all the modules within this module, including itself. Apply weight's observer and fake_quant from ``qconfig`` on ``target``. Attributes Get activation's quantization dtype as the method from ``qconfig``. Get activation's quantization parameters. Get weight's quantization dtype as the method from ``qconfig``. Get weight's quantization parameters. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QATModule` instance converted from a float :class:`~.Module` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Set quantization related configs with ``qconfig``, including observer and fake_quant for weight and activation. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. Use :func:`~.fake_quant_bias` to process ``target``. megengine.module.qat.BatchMatMulActivation module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.batch_matmul_activation.BatchMatMulActivation`, :class:`megengine.module.qat.module.QATModule` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:48+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`__init__ <megengine.module.qat.BatchMatMulActivation.__init__>`\ \(batch\, in\_features\, out\_features\[\, ...\]\) :obj:`apply <megengine.module.qat.BatchMatMulActivation.apply>`\ \(fn\) :obj:`apply_quant_activation <megengine.module.qat.BatchMatMulActivation.apply_quant_activation>`\ \(target\) :obj:`apply_quant_bias <megengine.module.qat.BatchMatMulActivation.apply_quant_bias>`\ \(target\, inp\, w\_qat\) :obj:`apply_quant_weight <megengine.module.qat.BatchMatMulActivation.apply_quant_weight>`\ \(target\) :obj:`buffers <megengine.module.qat.BatchMatMulActivation.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.qat.BatchMatMulActivation.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.qat.BatchMatMulActivation.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.qat.BatchMatMulActivation.eval>`\ \(\) :obj:`forward <megengine.module.qat.BatchMatMulActivation.forward>`\ \(inp\) :obj:`from_float_module <megengine.module.qat.BatchMatMulActivation.from_float_module>`\ \(float\_module\) :obj:`get_activation_dtype <megengine.module.qat.BatchMatMulActivation.get_activation_dtype>`\ \(\) :obj:`get_activation_qparams <megengine.module.qat.BatchMatMulActivation.get_activation_qparams>`\ \(\) :obj:`get_weight_dtype <megengine.module.qat.BatchMatMulActivation.get_weight_dtype>`\ \(\) :obj:`get_weight_qparams <megengine.module.qat.BatchMatMulActivation.get_weight_qparams>`\ \(\) :obj:`load_state_dict <megengine.module.qat.BatchMatMulActivation.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.qat.BatchMatMulActivation.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.qat.BatchMatMulActivation.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.qat.BatchMatMulActivation.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.qat.BatchMatMulActivation.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.qat.BatchMatMulActivation.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.qat.BatchMatMulActivation.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.qat.BatchMatMulActivation.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.qat.BatchMatMulActivation.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.qat.BatchMatMulActivation.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.qat.BatchMatMulActivation.reset_parameters>`\ \(\) :obj:`set_fake_quant <megengine.module.qat.BatchMatMulActivation.set_fake_quant>`\ \(enable\) :obj:`set_observer <megengine.module.qat.BatchMatMulActivation.set_observer>`\ \(enable\) :obj:`set_qconfig <megengine.module.qat.BatchMatMulActivation.set_qconfig>`\ \(qconfig\) :obj:`state_dict <megengine.module.qat.BatchMatMulActivation.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.qat.BatchMatMulActivation.train>`\ \(\[mode\, recursive\]\) :obj:`with_act <megengine.module.qat.BatchMatMulActivation.with_act>`\ :obj:`with_weight <megengine.module.qat.BatchMatMulActivation.with_weight>`\ :obj:`zero_grad <megengine.module.qat.BatchMatMulActivation.zero_grad>`\ \(\) :py:obj:`None` A :class:`~.QATModule` :class:`~.module.BatchMatMulActivation` with QAT support. Applies function ``fn`` to all the modules within this module, including itself. Apply weight's observer and fake_quant from ``qconfig`` on ``target``. Attributes Get activation's quantization dtype as the method from ``qconfig``. Get activation's quantization parameters. Get weight's quantization dtype as the method from ``qconfig``. Get weight's quantization parameters. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QATModule` instance converted from a float :class:`~.Module` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Set quantization related configs with ``qconfig``, including observer and fake_quant for weight and activation. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. Use :func:`~.fake_quant_bias` to process ``target``. megengine.module.qat.BatchMatMulActivation module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.batch_matmul_activation.BatchMatMulActivation`, :class:`megengine.module.qat.module.QATModule` 