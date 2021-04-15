��          �                 )     P   G  \   �  G   �  \   =     �      �     �  `   �     9  !   W  Q   y  b   �     .     5  g  <  )   �  P   �  \     G   |  \   �     !      /     P  `   _     �  !   �  Q      b   R     �     �   Applies batch normalization to the input. Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information. a boolean value to indicate whether batch norm is performed in training mode. Default: False a value added to the denominator for numerical stability. Default: 1e-5 bias tensor in the learnable affine parameters. See :math:`\beta` in :class:`~.BatchNorm2d`. input tensor. megengine.functional.batch\_norm output tensor. scaling tensor in the learnable affine parameters. See :math:`\gamma` in :class:`~.BatchNorm2d`. tensor to store running mean. tensor to store running variance. value used for the ``running_mean`` and ``running_var`` computation. Default: 0.9 whether to update ``running_mean`` and ``running_var`` inplace or return new tensors Default: True 参数 返回 Project-Id-Version:  megengine
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-15 18:59+0800
PO-Revision-Date: 2021-04-15 09:42+0000
Last-Translator: 
Language: zh_Hans_CN
Language-Team: Chinese Simplified
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 Applies batch normalization to the input. Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information. a boolean value to indicate whether batch norm is performed in training mode. Default: False a value added to the denominator for numerical stability. Default: 1e-5 bias tensor in the learnable affine parameters. See :math:`\beta` in :class:`~.BatchNorm2d`. input tensor. megengine.functional.batch\_norm output tensor. scaling tensor in the learnable affine parameters. See :math:`\gamma` in :class:`~.BatchNorm2d`. tensor to store running mean. tensor to store running variance. value used for the ``running_mean`` and ``running_var`` computation. Default: 0.9 whether to update ``running_mean`` and ``running_var`` inplace or return new tensors Default: True 参数 返回 