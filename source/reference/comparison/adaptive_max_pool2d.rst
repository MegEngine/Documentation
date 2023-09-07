.. _comparison-adaptive_max_pool2d:

=========================
AdaptiveMaxPool2d 差异对比
=========================

.. panels::

  torch.nn.AdaptiveMaxPool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.AdaptiveMaxPool2d(
        output_size
        return_indices
     )

  更多请查看 :py:class:`torch.nn.AdaptiveMaxPool2d`.

  ---

  megengine.module.AdaptiveMaxPool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.AdaptiveMaxPool2d(
         oshp
         ** kwargs

     )

  更多请查看 :py:class:`megengine.module.AdaptiveMaxPool2d`.

使用差异
--------

输入张量的形状
~~~~~~~~~~~~
   PyTorch 支持 NCHW 或者 CHW 的输入，MegEngine 支持 NCHW 的输入。

参数差异
--------

return_indices 参数
~~~~~~~~~~~~~~~~~
   PyTorch 中包含  ``return_indices`` 参数，MegEngine 无此参数，该参数是一个布尔值，用于指定是否返回输出张量的最大值位置的索引。该参数设置为 True 时，函数会返回一个元组的两部分：输出张量和最大值位置的索引。其中，输出张量是经过自适应最大池化操作后的结果，而最大值位置的索引是一个二维数组，用于表示在每个输出区域内最大值的位置。
  

.. code-block::: python

    import megengine 
    import torch 

    # 定义输入张量 
    input_tensor = torch.randn(1, 3, 64, 64) 

    # 使用MegEngine的AdaptiveMaxPool2d 
    me_pool = megengine.nn.AdaptiveMaxPool2d((32, 32)) 
    me_output = me_pool(input_tensor.astype(me.float32)) 

    # 使用PyTorch的AdaptiveMaxPool2d 
    torch_pool = torch.nn.AdaptiveMaxPool2d((32, 32)) 
    torch_output = torch_pool(input_tensor) 

    # 打印输出结果 
    print("MegEngine output:", me_output.numpy()) 
    print("PyTorch output:", torch_output.detach().numpy())
