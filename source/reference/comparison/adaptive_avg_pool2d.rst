.. _comparison-adaptive_avg_pool2d:

==========================
AdaptiveAvgPool2d 差异对比
==========================

.. panels::

  torch.nn.AdaptiveAvgPool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.AdaptiveAvgPool2d(
        output_size
     )

  更多请查看 :py:class:`torch.nn.AdaptiveAvgPool2d`.

  ---

  megengine.module.AdaptiveAvgPool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.AdaptiveAvgPool2d(
         oshp
         ** kwargs

     )

  更多请查看 :py:class:`megengine.module.AdaptiveAvgPool2d`.

使用差异
--------

输入张量的形状
~~~~~~~~~~~~
   PyTorch 支持 NCHW 或者 CHW 的输入，MegEngine 支持 NCHW 的输入。
   


.. code-block::: python

    import megengine 
    import torch 
    import numpy

    # 定义输入张量 
    input_tensor1 = torch.randn(1, 3, 64, 64) 
    input_tensor2 = megengine.random.normal(size=(1,3,64,64))
    
    # 使用MegEngine的AdaptiveAvgPool2d 
    me_pool = megengine.module.AdaptiveAvgPool2d((32, 32)) 
    me_output = me_pool(input_tensor2.astype(numpy.float32)) 

    # 使用PyTorch的AdaptiveAvgPool2d 
    torch_pool = torch.nn.AdaptiveAvgPool2d((32, 32)) 
    torch_output = torch_pool(input_tensor1) 

    # 打印输出结果 
    print("MegEngine output:", me_output.numpy()) 
    print("PyTorch output:", torch_output.numpy())
