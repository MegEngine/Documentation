.. _comparison-nn-conv:

===================
Conv 差异对比
===================

.. admonition:: 背景：卷积运算

   对于形状为 :math:`(N, C_\text {in}, ...)` 的输入和形状为 :math:`(N, C_\text {out}, ...)` 的输出，有：

   .. math::

      \operatorname{out}\left(N_i, C_{\text {out }_j}\right)
      =\operatorname{bias}\left(C_{\text {out }_j}\right)+
      \sum_{k=0}^{C_{\text {in }}-1} \text { weight }\left(C_{\text {out }_j}, k\right) \star \operatorname{input}\left(N_i, k\right)

   其中 :math:`\star` 表示互相关计算，具体的计算方式可参考以下接口文档：

   * MegEngine - :class:`~megengine.module.Conv1d` / :class:`~megengine.module.Conv2d` / :class:`~megengine.module.Conv3d` ...
   * Pytorch - :class:`~torch.nn.Conv1d` / :class:`~torch.nn.Conv2d` / :class:`~torch.nn.Conv3d` ... 

   但 MegEngine 的 Conv 和 Pytorch 的 Conv 存在如下差异——


.. admonition:: Weight 形状不同
   :class: warning

   Pytorch 的 weight 形状为 ``(out_channels, in_channels // groups, kernel_size...)``,
   而 MegEngine 的 weight 在 ``groups=1`` 时为 ``(out_channels, in_channels, kernel_size)``,
   其它情况下为 ``(groups, out_channels // groups, in_channels // groups, kernel_size...)``.

   其中 ``kernel_size`` 满足：

   * Conv1d - ``kernel_size`` 为 ``kernel_length``
   * Conv2d - ``kernel_size`` 为 ``kernel_height, kernel_width``
   * Conv3d - ``kernel_size`` 为 ``kernel_depth, kernel_height, kernel_width``

   .. code-block:: python3

      import megengine
      import torch
   
      m_conv = megengine.module.Conv2d(10, 20, kernel_size=3, padding=1, groups=2)
      t_conv = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1, groups=2)
      print(m_conv.weight.shape) # (2, 10, 5, 3, 3)
      print(t_conv.weight.shape) # torch.Size([20, 5, 3, 3])

.. admonition:: Bias 形状不同
   :class: warning

   Pytorch 的 bias 形状为 ``(out_channels,)``,
   而 MegEngine 的 bias 形状为 ``(1, out_channels, dims...)``, 省略的维度为多个 1.

   其中 ``dims`` 满足：

   * Conv1d - ``dims`` 为 ``1``
   * Conv2d - ``dims`` 为 ``1, 1``
   * Conv3d - ``dims`` 为 ``1, 1, 1``

   .. code-block:: python3

      import megengine
      import torch

      m_conv = megengine.module.Conv2d(10, 20, kernel_size=3)
      t_conv = torch.nn.Conv2d(10, 20, kernel_size=3)
      print(m_conv.bias.shape) # (1, 20, 1, 1)
      print(t_conv.bias.shape) # (torch.Size([20])
