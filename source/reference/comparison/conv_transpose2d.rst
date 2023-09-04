.. _comparison-conv-transpose2d:

=========================
ConvTranspose2d 差异对比
=========================

.. panels::

  torch.nn.ConvTranspose2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
			stride=1,
			padding=0,
        output_padding=0,
			groups=1,
			bias=True,
			dilation=1,
			padding_mode=‘zeros’,
			device=None,
			dtype=None
     )

  更多请查看 :py:class:`torch.nn.ConvTranspose2d`.

  ---

  megengine.module.ConvTranspose2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.ConvTranspose2d(
			in_channels,
        out_channels,
        kernel_size,
			stride=1,
			padding=0,
        output_padding=0,
			dilation=1,
			groups=1,
			bias=True,
        conv_mode=’cross_correlation’
			compute_mode=‘default’
 			** kwargs
     )

  更多请查看 :py:class:`megengine.module.ConvTranspose2d`.

使用差异
--------

padding
~~~~~~~~~~~~
PyTorch padding 可以是单个数字或元组，MegEngine padding 仅支持数值填充 0.

 compute_mode 参数
~~~~~~~~~~~~~~~~~
MegEngine 中包含  ``compute_mode`` 参数，PyTorch 中无此参数，该参数用于指定计算模式，当设置 “default” 时, 不会对中间结果的精度有特殊要求。当设置 “float32” 时, “float32” 将被用作中间结果的累加器, 但是只有当输入和输出的 dtype 是 float16 时有效。 

conv_mode 参数
~~~~~~~~~~~~~~~
MegEngine 中包含  ``conv_mode`` 参数，PyTorch 中无此参数，该参数用于指定卷积模式，默认值为 “cross_correlation” 时, 表示使用交叉相关计算模式。在这种模式下，输入张量被视为滤波器的目标图像，而滤波器在输入上进行滑动以生成输出张量。

需要注意的是，交叉相关计算模式和转置卷积计算模式是不同的。在交叉相关计算模式下，输入张量的大小不需要与滤波器的大小匹配，输出张量的大小会比输入张量大。而在转置卷积计算模式下，输入张量的大小需要与滤波器的大小匹配，输出张量的大小会比输入张量小。

.. code-block::: python

    import megengine
    import torch

    # 定义输入张量
    input_tensor = torch.randn(1, 3, 64, 64)

    # 使用MegEngine的ConvTranspose2d
    me_conv_transpose = megengine.nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
    me_output = me_conv_transpose(input_tensor.astype(me.float32))

    # 使用PyTorch的ConvTranspose2d
    torch_conv_transpose = torch.nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
    torch_output = torch_conv_transpose(input_tensor)

    # 打印输出结果
    print("MegEngine output:", me_output.numpy())
    print("PyTorch output:", torch_output.detach().numpy())
