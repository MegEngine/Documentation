��          �                      $   7  4   \  0   �  {   �  2   >  ;   q  6   �     �  ,   �  �   (     %  1  A     s     z  g  �     �  $   	  4   .  0   c  {   �  2     ;   C  6        �  ,   �  �   �     �	  1  
     E     L   1D convolution operation. :py:class:`~megengine.tensor.Tensor` Dilation of the 1D convolution operation. Default: 1 Refer to :class:`~.Conv1d` for more information. Size of the paddings added to the input on both sides of its spatial dimensions. Only zero-padding is supported. Default: 0 Stride of the 1D convolution operation. Default: 1 Supports 'CROSS_CORRELATION'. Default: 'CROSS_CORRELATION'. The bias added to the result of convolution (if given) The convolution kernel The feature map of the convolution operation When set to 'DEFAULT', no special requirements will be placed on the precision of intermediate results. When set to 'FLOAT32', Float32 would be used for accumulator and intermediate result, but only effective when input and output are of Float16 dtype. megengine.functional.conv1d number of groups to divide input and output channels into, so as to perform a "grouped convolution". When ``groups`` is not 1, ``in_channels`` and ``out_channels`` must be divisible by ``groups``, and the shape of weight should be ``(groups, out_channel // groups, in_channels // groups, height, width)``. 参数 返回类型 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:43+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 1D convolution operation. :py:class:`~megengine.tensor.Tensor` Dilation of the 1D convolution operation. Default: 1 Refer to :class:`~.Conv1d` for more information. Size of the paddings added to the input on both sides of its spatial dimensions. Only zero-padding is supported. Default: 0 Stride of the 1D convolution operation. Default: 1 Supports 'CROSS_CORRELATION'. Default: 'CROSS_CORRELATION'. The bias added to the result of convolution (if given) The convolution kernel The feature map of the convolution operation When set to 'DEFAULT', no special requirements will be placed on the precision of intermediate results. When set to 'FLOAT32', Float32 would be used for accumulator and intermediate result, but only effective when input and output are of Float16 dtype. megengine.functional.conv1d number of groups to divide input and output channels into, so as to perform a "grouped convolution". When ``groups`` is not 1, ``in_channels`` and ``out_channels`` must be divisible by ``groups``, and the shape of weight should be ``(groups, out_channel // groups, in_channels // groups, height, width)``. 参数 返回类型 