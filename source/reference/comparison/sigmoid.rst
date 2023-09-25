.. _comparison-sigmoid:

=========================
Pad 差异对比
=========================

.. panels::

  torch.nn.Sigmoid
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.functional.pad(
        input,
        out=None
     )

  更多请查看 :py:class:`torch.nn.Sigmoid`.

  ---

  megengine.module.Sigmoid
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Sigmoid(
        name = None
     )

  更多请查看 :py:class:`megengine.module.Sigmoid`.

参数差异
--------

out
~~~~~~~~~~~~
   PyTorch 中的 out 参数表示输出的 Tensor，MegEngine 中无此参数。



.. code-block::: python

    import megengine 
    import torch  
  
    # 创建一个张量  
    x1 = megengine.tensor([-1.0, 0.0, 1.0]) 
    x2 = torch.tensor([-1.0, 0.0, 1.0])  
 
    # 创建一个 LeakyReLU 激活函数对象
    sigmoid_torch = torch.nn.Sigmoid()
    sigmoid_meg = megengine.module.Sigmoid()

    
    # 在张量上应用 sigmoid 函数  
    y1 = sigmoid_meg(x1) 
    y2 = sigmoid_torch(x2) 
  





 
  
  
   
 
