��          �                 (     $   F  2   k  	   �     �  ~   �     0  O   <  !   �     �  v   �  3   4     h     o     v  g  �  (   �  $     2   9  	   l     v  ~        �  O   
  !   Z     |  v   �  3        6     =     D   (batch, oh, ow, 2) transformation matrix :py:class:`~megengine.tensor.Tensor` Applies remap transformation to batched 2D images. Examples: Outputs: The input images are transformed to the output images by the tensor map_xy. The output's H and W are same as map_xy's H and W. input image interpolation methods. Default: "LINEAR". Currently only support "LINEAR" mode. megengine.functional.vision.remap output tensor. pixel extrapolation method. Default: "REPLICATE". Currently also support "CONSTANT", "REFLECT", "REFLECT_101", "WRAP". value used in case of a constant border. Default: 0 参数 返回 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:41+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 (batch, oh, ow, 2) transformation matrix :py:class:`~megengine.tensor.Tensor` Applies remap transformation to batched 2D images. Examples: Outputs: The input images are transformed to the output images by the tensor map_xy. The output's H and W are same as map_xy's H and W. input image interpolation methods. Default: "LINEAR". Currently only support "LINEAR" mode. megengine.functional.vision.remap output tensor. pixel extrapolation method. Default: "REPLICATE". Currently also support "CONSTANT", "REFLECT", "REFLECT_101", "WRAP". value used in case of a constant border. Default: 0 参数 返回 返回类型 