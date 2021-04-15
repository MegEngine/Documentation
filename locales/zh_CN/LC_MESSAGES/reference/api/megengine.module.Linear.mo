��    7      �              �  _   �  4   �  C   "  B   f  Q   �  0   �  7   ,  ^   d  @   �  X     N   ]  L   �  ^   �  I   X  V   �  ^   �  `   X  H   �  X   	  F   [	  :   �	     �	  \   �	  P   I
  	   �
  H   �
     �
     �
  *   �
  +   )  l   U  M   �  U     �   f  x   �  ?   t  2   �  �   �  �   �  H   *  "   s  Y   �  X   �  U   I     �  =   �  
   �                !  *   =     h     u  0   |  g  �  _     4   u  C   �  B   �  Q   1  0   �  7   �  ^   �  @   K  X   �  N   �  L   4  ^   �  I   �  V   *  ^   �  `   �  H   A  X   �  F   �  :   *     e  \   t  P   �  	   "  H   ,     u     }  *   �  +   �  l   �  M   J  U   �  �   �  x   �  ?   �  2   <  �   o  �     H   �  "   �  Y     X   x  U   �     '   =   ?   
   }      �      �      �   *   �      �      �   0   !   :obj:`__init__ <megengine.module.Linear.__init__>`\ \(in\_features\, out\_features\[\, bias\]\) :obj:`apply <megengine.module.Linear.apply>`\ \(fn\) :obj:`buffers <megengine.module.Linear.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.Linear.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.Linear.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.Linear.eval>`\ \(\) :obj:`forward <megengine.module.Linear.forward>`\ \(x\) :obj:`load_state_dict <megengine.module.Linear.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.Linear.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.Linear.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.Linear.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.Linear.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.Linear.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.Linear.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.Linear.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.Linear.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.Linear.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.Linear.reset_parameters>`\ \(\) :obj:`state_dict <megengine.module.Linear.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.Linear.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.Linear.zero_grad>`\ \(\) :py:obj:`None` Applies a linear transformation to the input. For instance, if input is x, then output y is: Applies function ``fn`` to all the modules within this module, including itself. Examples: Loads a given dictionary created by :func:`state_dict` into this module. Methods Outputs: Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. if it's ``False``, the layer will not learn an additional ``bias``. Default: ``True`` megengine.module.Linear module's name, can be initialized by the ``kwargs`` parameter param name rtype size of each input sample. size of each output sample. where :math:`y_i= \sum_j W_{ij} x_j + b_i` y = xW^T + b 参数 基类：:class:`megengine.module.module.Module` Project-Id-Version:  megengine
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
 :obj:`__init__ <megengine.module.Linear.__init__>`\ \(in\_features\, out\_features\[\, bias\]\) :obj:`apply <megengine.module.Linear.apply>`\ \(fn\) :obj:`buffers <megengine.module.Linear.buffers>`\ \(\[recursive\]\) :obj:`children <megengine.module.Linear.children>`\ \(\*\*kwargs\) :obj:`disable_quantize <megengine.module.Linear.disable_quantize>`\ \(\[value\]\) :obj:`eval <megengine.module.Linear.eval>`\ \(\) :obj:`forward <megengine.module.Linear.forward>`\ \(x\) :obj:`load_state_dict <megengine.module.Linear.load_state_dict>`\ \(state\_dict\[\, strict\]\) :obj:`modules <megengine.module.Linear.modules>`\ \(\*\*kwargs\) :obj:`named_buffers <megengine.module.Linear.named_buffers>`\ \(\[prefix\, recursive\]\) :obj:`named_children <megengine.module.Linear.named_children>`\ \(\*\*kwargs\) :obj:`named_modules <megengine.module.Linear.named_modules>`\ \(\[prefix\]\) :obj:`named_parameters <megengine.module.Linear.named_parameters>`\ \(\[prefix\, recursive\]\) :obj:`parameters <megengine.module.Linear.parameters>`\ \(\[recursive\]\) :obj:`register_forward_hook <megengine.module.Linear.register_forward_hook>`\ \(hook\) :obj:`register_forward_pre_hook <megengine.module.Linear.register_forward_pre_hook>`\ \(hook\) :obj:`replace_param <megengine.module.Linear.replace_param>`\ \(params\, start\_pos\[\, seen\]\) :obj:`reset_parameters <megengine.module.Linear.reset_parameters>`\ \(\) :obj:`state_dict <megengine.module.Linear.state_dict>`\ \(\[rst\, prefix\, keep\_var\]\) :obj:`train <megengine.module.Linear.train>`\ \(\[mode\, recursive\]\) :obj:`zero_grad <megengine.module.Linear.zero_grad>`\ \(\) :py:obj:`None` Applies a linear transformation to the input. For instance, if input is x, then output y is: Applies function ``fn`` to all the modules within this module, including itself. Examples: Loads a given dictionary created by :func:`state_dict` into this module. Methods Outputs: Registers a hook to handle forward inputs. Registers a hook to handle forward results. Replaces module's parameters with ``params``, used by :class:`~.ParamPack` to speedup multimachine training. Returns an iterable for all the modules within this module, including itself. Returns an iterable for all the submodules that are direct attributes of this module. Returns an iterable for key :class:`~.Parameter` pairs of the module, where ``key`` is the dotted path from this module to the :class:`~.Parameter`. Returns an iterable for key buffer pairs of the module, where ``key`` is the dotted path from this module to the buffer. Returns an iterable for the :class:`~.Parameter` of the module. Returns an iterable for the buffers of the module. Returns an iterable of key-module pairs for all the modules within this module, including itself, where 'key' is the dotted path from this module to the submodules. Returns an iterable of key-submodule pairs for all the submodules that are direct attributes of this module, where 'key' is the attribute name of submodules. Sets ``module``'s ``quantize_disabled`` attribute and return ``module``. Sets all parameters' grads to zero Sets training mode of all the modules within this module (including itself) to ``False``. Sets training mode of all the modules within this module (including itself) to ``mode``. if it's ``False``, the layer will not learn an additional ``bias``. Default: ``True`` megengine.module.Linear module's name, can be initialized by the ``kwargs`` parameter param name rtype size of each input sample. size of each output sample. where :math:`y_i= \sum_j W_{ij} x_j + b_i` y = xW^T + b 参数 基类：:class:`megengine.module.module.Module` 