.. _comparison-relu:

=========================
ReLU 差异对比
=========================

.. panels::

  torch.nn.ReLU
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.ReLU(
        inplace=False
     )

  更多请查看 :py:class:`torch.nn.ReLU`.

  ---

  megengine.module.ReLU
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.ReLU(
        name=None
     )

  更多请查看 :py:class:`megengine.module.ReLU`.


参数差异
--------


inplace
~~~~~~~~~~~~

Pytorch 中的 inplace 参数表示在不更改变量的内存地址的情况下，直接修改变量的值，MegEngine 无此参数。



.. code-block::: python

    import torch 
    import megengine 
  
    # 创建一个张量  
    x1 = torch.tensor([-1.0, 0.0, 1.0, 2.0])  
    x2 = megengine.tensor([-1.0, 0.0, 1.0, 2.0])  
 
    # 创建一个 ReLU 激活函数对象
    relu_torch = torch.nn.ReLU()
    relu_meg = megengine.module.ReLU()

    #
    # 在张量上应用 ReLU 函数  
    y1 = relu_torch(x1) 
    y2 = relu_meg(x2)   
  
      





 
  
  
   
