��    3      �              L  s   M  D   �  S     ]   Z  w   �  R   0  a   �  @   �  I   &  a   p  n   �  P   A  h   �  ^   �  \   Z  n   �  Y   &	  f   �	  n   �	  p   V
  X   �
  h      V   �  J   �     +  P   :  H   �     �  1   �  *     +   9  l   e  \   �  M   /  U   }  �   �  x   h  ?   �  2   !  �   T  �   �  H   �  "   �  Y     X   ]  '   �  =   �  
        '  H   -  g  v  s   �  D   R  S   �  ]   �  w   I  R   �  a     @   v  I   �  a     n   c  P   �  h   #  ^   �  \   �  n   H  Y   �  f     n   x  p   �  X   X  h   �  V     J   q     �  P   �  H        e  1   m  *   �  +   �  l   �  \   c  M   �  U      �   d   x   �   ?   r!  2   �!  �   �!  �   �"  H   (#  "   q#  Y   �#  X   �#  '   G$  =   o$  
   �$     �$  H   �$   :obj:`__init__ <megengine.module.quantized.ConvBnRelu2d.__init__>`\ \(in\_channels\, out\_channels\, kernel\_size\) :obj:`apply <megengine.module.quantized.ConvBnRelu2d.apply>`\ \(fn\) :obj:`buffers <megengine.module.quantized.ConvBnRelu2d.buffers>`\ \(\[recursive\]\) :obj:`calc_conv <megengine.module.quantized.ConvBnRelu2d.calc_conv>`\ \(inp\, weight\, bias\) :obj:`calc_conv_quantized <megengine.module.quantized.ConvBnRelu2d.calc_conv_quantized>`\ \(inp\[\, nonlinear\_mode\]\) :obj:`children <megengine.module.quantized.ConvBnRelu2d.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.quantized.ConvBnRelu2d.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.quantized.ConvBnRelu2d.eval>`\ \(\) :obj:`forward <megengine.module.quantized.ConvBnRelu2d.forward>`\ \(inp\) :obj:`from_qat_module <megengine.module.quantized.ConvBnRelu2d.from_qat_module>`\ \(qat\_module\) :obj:`load_state_dict <megengine.module.quantized.ConvBnRelu2d.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.quantized.ConvBnRelu2d.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.quantized.ConvBnRelu2d.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.quantized.ConvBnRelu2d.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.quantized.ConvBnRelu2d.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.quantized.ConvBnRelu2d.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.quantized.ConvBnRelu2d.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.quantized.ConvBnRelu2d.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.quantized.ConvBnRelu2d.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.quantized.ConvBnRelu2d.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.quantized.ConvBnRelu2d.reset_parameters>`\ \(\) :obj:`state_dict <megengine.module.quantized.ConvBnRelu2d.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.quantized.ConvBnRelu2d.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.quantized.ConvBnRelu2d.zero_grad>`\ \(\) :py:obj:`None` Applies function ``fn`` to all the modules within this module, including itself. Loads a given dictionary created by :func:`state_dict` into this module. Methods Quantized version of :class:`~.qat.ConvBnRelu2d`. Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QuantizedModule` instance converted from a :class:`~.QATModule` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. megengine.module.quantized.ConvBnRelu2d module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.quantized.conv_bn._ConvBnActivation2d` Project-Id-Version:  megengine
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
 :obj:`__init__ <megengine.module.quantized.ConvBnRelu2d.__init__>`\ \(in\_channels\, out\_channels\, kernel\_size\) :obj:`apply <megengine.module.quantized.ConvBnRelu2d.apply>`\ \(fn\) :obj:`buffers <megengine.module.quantized.ConvBnRelu2d.buffers>`\ \(\[recursive\]\) :obj:`calc_conv <megengine.module.quantized.ConvBnRelu2d.calc_conv>`\ \(inp\, weight\, bias\) :obj:`calc_conv_quantized <megengine.module.quantized.ConvBnRelu2d.calc_conv_quantized>`\ \(inp\[\, nonlinear\_mode\]\) :obj:`children <megengine.module.quantized.ConvBnRelu2d.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.quantized.ConvBnRelu2d.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.quantized.ConvBnRelu2d.eval>`\ \(\) :obj:`forward <megengine.module.quantized.ConvBnRelu2d.forward>`\ \(inp\) :obj:`from_qat_module <megengine.module.quantized.ConvBnRelu2d.from_qat_module>`\ \(qat\_module\) :obj:`load_state_dict <megengine.module.quantized.ConvBnRelu2d.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.quantized.ConvBnRelu2d.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.quantized.ConvBnRelu2d.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.quantized.ConvBnRelu2d.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.quantized.ConvBnRelu2d.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.quantized.ConvBnRelu2d.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.quantized.ConvBnRelu2d.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.quantized.ConvBnRelu2d.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.quantized.ConvBnRelu2d.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.quantized.ConvBnRelu2d.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.quantized.ConvBnRelu2d.reset_parameters>`\ \(\) :obj:`state_dict <megengine.module.quantized.ConvBnRelu2d.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.quantized.ConvBnRelu2d.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.quantized.ConvBnRelu2d.zero_grad>`\ \(\) :py:obj:`None` Applies function ``fn`` to all the modules within this module, including itself. Loads a given dictionary created by :func:`state_dict` into this module. Methods Quantized version of :class:`~.qat.ConvBnRelu2d`. Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Return a :class:`~.QuantizedModule` instance converted from a :class:`~.QATModule` instance. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. megengine.module.quantized.ConvBnRelu2d module's name, can be initialized by the ``kwargs`` parameter param name rtype 基类：:class:`megengine.module.quantized.conv_bn._ConvBnActivation2d` 