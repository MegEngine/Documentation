.. _comparison-leaky-relu:

=========================
LeakyRelu 差异对比
=========================

.. panels::

  torch.nn.LeakyRelu
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.LeakyRelu(
        negative_slope=0.01,
        inplace=False
     )

  更多请查看 :py:class:`torch.nn.LeakyRelu`.

  ---

  megengine.module.LeakyRelu
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.LeakyRelu(
        negative_slope=0.01,
        ** kwargs
     )

  更多请查看 :py:class:`megengine.module.LeakyRelu`.


参数差异
--------

inplace
~~~~~~~~~~~~

Pytorch 中的 inplace 参数表示是否要原地操作，默认不开启，MegEngine 无此参数。



.. code-block::: python

    import torch 
    import megengine 
  
    # 创建一个张量  
    x1 = torch.tensor([-1.0, 0.0, 1.0, 2.0])  
    x2 = megengine.tensor([-1.0, 0.0, 1.0, 2.0]) 

    # 创建一个 LeakyReLU 激活函数对象  
    leakyrelu_torch = torch.nn.LeakyReLU(negative_slope=0.1)
    leakyrelu_meg = megengine.module.LeakyReLU(negative_slope=0.1) 
  
    # 在张量上应用 LeakyReLU 函数  
    y1 = leakyrelu_torch(x1) 
    y2 = leakyrelu_meg(x2)   
  
      





 
  
  
   
