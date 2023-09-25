.. _comparison-max-pool2d:

===================
Max_Pool2d 差异对比
===================

.. panels::

  torch.nn.MaxPool2d
  ^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.MaxPool2d(
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False
     )

  更多请查看 :py:class:`torch.nn.Max_Pool2d`.

  ---

  megengine.module.Max_Pool2d
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Avg_Pool2d(
         kernel_size, 
         stride=None, 
         padding=0, 
         ** kwargs
     )

  更多请查看 :py:class:`megengine.module.Max_Pool2d`.

参数差异
--------

dilation 参数
~~~~~~~~~~~~
   Pytorch 中有 ``dilation`` ，MegEngine 中无此参数，该参数用于窗口的元素间隔控制;
   
return_indices 参数
~~~~~~~~~~~~~~~~~~~
PyTorch 中有 ``return_indices`` 参数，MegEngine 无此参数，该参数等于 True 时，会返回输出最大值的序号。


ceil_mode 参数
~~~~~~~~~~~~~~~~~~~~~
PyTorch 中有 ceil_mode 参数，MegEngine 无此参数，该参数为 True 时表示在计算输出形状的过程中采用向上取整的操作，为 False 时，采用向下取整。

.. code-block:: python
    import megengine 
    import torch 

    # 定义输入张量 
    input_tensor1 = torch.randn(1, 3, 64, 64) 
    input_tensor2 = megengine.random.normal(size=(1,3,64,64))

    # 使用MegEngine的max_pool2d 
    me_pool = megengine.module.MaxPool2d(kernel_size=2, stride=2) 
    me_output = me_pool(input_tensor2.astype(me.float32)) 

    # 使用PyTorch的max_pool2d 
    torch_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) 
    torch_output = torch_pool(input_tensor1) 

    # 打印输出结果 
    print("MegEngine output:", me_output.numpy()) 
    print("PyTorch output:", torch_output.numpy())
