��    ,      |              �  E   �  9   #  H   ]  G   �  V   �  5   E  >   {  c   �  E     ]   d  S   �  Q     c   h  N   �  [     c   w  e   �  ]   A  K   �  ?   �  �   +	  P   �	  H   %
     n
  *   v
  +   �
  l   �
  M   :  U   �  �   �  x   s  ?   �  2   ,  �   _  �     H   �  "   �  Y     X   h     �  =   �  
     0   '  g  X  E   �  9     H   @  G   �  V   �  5   (  >   ^  c   �  E     ]   G  S   �  Q   �  c   K  N   �  [   �  c   Z  e   �  ]   $  K   �  ?   �  �     P   �  H        Q  *   Y  +   �  l   �  M     U   k  �   �  x   V  ?   �  2     �   B  �   �  H   �  "   �  Y   �  X   K     �  =   �  
   �  0   
   :obj:`__init__ <megengine.module.DequantStub.__init__>`\ \(\[name\]\) :obj:`apply <megengine.module.DequantStub.apply>`\ \(fn\) :obj:`buffers <megengine.module.DequantStub.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.DequantStub.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.DequantStub.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.DequantStub.eval>`\ \(\) :obj:`forward <megengine.module.DequantStub.forward>`\ \(inp\) :obj:`load_state_dict <megengine.module.DequantStub.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.DequantStub.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.DequantStub.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.DequantStub.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.DequantStub.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.DequantStub.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.DequantStub.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.DequantStub.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.DequantStub.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.DequantStub.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`state_dict <megengine.module.DequantStub.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.DequantStub.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.DequantStub.zero_grad>`\ \(\) A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule` version :class:`~.qat.DequantStub` using :func:`~.quantize.quantize_qat`. Applies function ``fn`` to all the modules within this module, including itself. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. megengine.module.DequantStub module's name, can be initialized by the ``kwargs`` parameter param name 基类：:class:`megengine.module.module.Module` Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:46+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :obj:`__init__ <megengine.module.DequantStub.__init__>`\ \(\[name\]\) :obj:`apply <megengine.module.DequantStub.apply>`\ \(fn\) :obj:`buffers <megengine.module.DequantStub.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.DequantStub.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.DequantStub.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.DequantStub.eval>`\ \(\) :obj:`forward <megengine.module.DequantStub.forward>`\ \(inp\) :obj:`load_state_dict <megengine.module.DequantStub.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.DequantStub.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.DequantStub.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.DequantStub.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.DequantStub.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.DequantStub.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.DequantStub.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.DequantStub.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.DequantStub.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.DequantStub.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`state_dict <megengine.module.DequantStub.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.DequantStub.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.DequantStub.zero_grad>`\ \(\) A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule` version :class:`~.qat.DequantStub` using :func:`~.quantize.quantize_qat`. Applies function ``fn`` to all the modules within this module, including itself. Loads a given dictionary created by :func:`state_dict` into this module. Methods Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. megengine.module.DequantStub module's name, can be initialized by the ``kwargs`` parameter param name 基类：:class:`megengine.module.module.Module` 