.. _comparison-softmax:

=========================
Softmax 差异对比
=========================

.. panels::

  torch.nn.Softmax
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.Softmax(
        dim=None
     )

  更多请查看 :py:class:`torch.nn.Softmax`.

  ---

  megengine.module.Softmax
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Softmax(
        axis=None
     )

  更多请查看 :py:class:`megengine.module.Softmax`.

参数差异
--------


两者功能一致且参数用法一致，仅参数名不一致，dim 与 axis 都是对输入 Tensor 进行运算的轴，仅参数名不一致。



.. code-block::: python

    import torch 
    import megengine 
  
    # 创建一个张量  
    x1 = torch.tensor([-1.0, 0.0, 1.0])  
    x2 = megengine.tensor([-1.0, 0.0, 1.0])  

    # 创建一个softmax 激活函数对象
    softmax_torch = torch.nn.Softmax(dim=0)
    softmax_meg = megengine.module.Softmax(axis=0)
  
    # 在张量上应用softmax函数  
    y1 = softmax_torch(x1) 
    y2 = softmax_meg(x2)   
  
      





 
  
  
   
 
