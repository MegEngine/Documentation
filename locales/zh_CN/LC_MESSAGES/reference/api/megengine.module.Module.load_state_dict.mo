��          \               �   7   �   �   �   U   �  �   �  9   �  )   �     (  g  /  7   �  �   �  U   �  �   �  9   �  )   �     "   Here returning ``None`` means skipping parameter ``k``. Loads a given dictionary created by :func:`state_dict` into this module. If ``strict`` is ``True``, the keys of :func:`state_dict` must exactly match the keys returned by :func:`state_dict`. To prevent shape mismatch (e.g. load PyTorch weights), we can reshape before loading: Users can also pass a closure: ``Function[key: str, var: Tensor] -> Optional[np.ndarray]`` as a `state_dict`, in order to handle complex situations. For example, load everything except for the final linear classifier: We can also perform inplace re-initialization or pruning: megengine.module.Module.load\_state\_dict 参数 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:47+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 Here returning ``None`` means skipping parameter ``k``. Loads a given dictionary created by :func:`state_dict` into this module. If ``strict`` is ``True``, the keys of :func:`state_dict` must exactly match the keys returned by :func:`state_dict`. To prevent shape mismatch (e.g. load PyTorch weights), we can reshape before loading: Users can also pass a closure: ``Function[key: str, var: Tensor] -> Optional[np.ndarray]`` as a `state_dict`, in order to handle complex situations. For example, load everything except for the final linear classifier: We can also perform inplace re-initialization or pruning: megengine.module.Module.load\_state\_dict 参数 