��          �                 $     �   2  	   �     �    �     �  W   �  '   N     v  2   �  (   �     �     �     �  g  �  $   d  �   �  	            &     ?	  W   M	  '   �	     �	  2   �	  (   
     8
     ?
     F
   :py:class:`~megengine.tensor.Tensor` Down/up samples the input tensor to either the given size or with the given scale_factor. ``size`` can not coexist with ``scale_factor``. Examples: Outputs: This only has an effect when `mode` is "BILINEAR" or "LINEAR". Geometrically, we consider the pixels of the input and output as squares rather than points. If set to ``True``, the input and output tensors are aligned by the center points of their corner pixels, preserving the values at the corner pixels. If set to ``False``, the input and output tensors are aligned by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary values, making this operation *independent* of input size input tensor. interpolation methods, acceptable values are: "BILINEAR", "LINEAR". Default: "BILINEAR" megengine.functional.vision.interpolate output tensor. scaling factor of the output tensor. Default: None size of the output tensor. Default: None 参数 返回 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:40+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 :py:class:`~megengine.tensor.Tensor` Down/up samples the input tensor to either the given size or with the given scale_factor. ``size`` can not coexist with ``scale_factor``. Examples: Outputs: This only has an effect when `mode` is "BILINEAR" or "LINEAR". Geometrically, we consider the pixels of the input and output as squares rather than points. If set to ``True``, the input and output tensors are aligned by the center points of their corner pixels, preserving the values at the corner pixels. If set to ``False``, the input and output tensors are aligned by the corner points of their corner pixels, and the interpolation uses edge value padding for out-of-boundary values, making this operation *independent* of input size input tensor. interpolation methods, acceptable values are: "BILINEAR", "LINEAR". Default: "BILINEAR" megengine.functional.vision.interpolate output tensor. scaling factor of the output tensor. Default: None size of the output tensor. Default: None 参数 返回 返回类型 