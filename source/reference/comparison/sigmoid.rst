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

  更多请查看 :py:class:`megengine.module.Pad`.

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
  
    # 在张量上应用 sigmoid 函数  
    y1 = megengine.sigmoid(x) 
    y2 = torch.sigmoid(x) 





 
  
  
   
 
