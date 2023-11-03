.. _comparison-avg-pool2d:

===================
Avg_Pool2d 差异对比
===================

.. panels::

  torch.nn.AvgPool2d
  ^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.AvgPool2d(
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None
     )

  更多请查看 :py:class:`torch.nn.Avg_Pool2d`.

  ---

  megengine.module.Avg_Pool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Avg_Pool2d(
         kernel_size, 
         stride=None, 
         padding=0, 
         mode='average_count_exclude_padding',
         ** kwargs
     )

  更多请查看 :py:class:`megengine.module.Avg_Pool2d`.

参数差异
--------

mode 参数
~~~~~~~~~~~~~
   MegEngine 中 ``mode`` 参数对应 PyTorch 的 ``count_include_pad`` 参数，两者默认值设置有差别，MegEngine 中默认不计算 padding 的值，PyTorch 反之；
   
ceil_mode 参数
~~~~~~~~~~~~~~~
PyTorch 中有 ``ceil_mode`` 参数，MegEngine 无此参数，此参数通过计算公式达到便于控制 output 的尺寸，当输出后的图像大小不满足 output 输出为整数时，此参数可以不用 padding，用此参数向上或者向下取整。MegEngine 则通过手动增加 padding 控制 output，如果输出不为整数，则自动向下取整。


divisor_override 参数
~~~~~~~~~~~~~~~~~~~~~~
PyTorch 中有 divisor_override 参数，MegEngine 无此参数，该参数的目的是可以控制求期望时的分母，将结果缩放/扩大，MenEgngine 中 无此参数。

.. code-block:: python

    import megengine
    import torch
    import numpy as np

    # 定义输入张量
    input_tensor1 = torch.randn(1, 3, 64, 64)
    input_tensor2 = megengine.random.normal(size=(1,3,64,64))

    # 使用MegEngine的avg_pool2d
    me_pool = megengine.module.AvgPool2d(kernel_size=2, stride=2)
    me_output = me_pool(input_tensor2.astype(np.float32))

    # 使用PyTorch的avg_pool2d
    torch_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    torch_output = torch_pool(input_tensor1)

    # 打印输出结果
    print("MegEngine output:", me_output.numpy())
    print("PyTorch output:", torch_output.numpy())
