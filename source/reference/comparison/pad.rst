.. _comparison-pad:

=========================
Pad 差异对比
=========================

.. panels::

  torch.nn.functional.pad
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.functional.pad(
        input,
        pad,
        mode='constant',
        value=None
     )

  更多请查看 :py:class:`torch.nn.functional.pad`.

  ---

  megengine.module.Pad
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Pad(
        pad_width,
        mode='constant',
        constant_val=0.0
			
     )

  更多请查看 :py:class:`megengine.module.Pad`.

参数差异
--------

pad
~~~~~~~~~~~~
   PyTorch pad 参数表示一个 one-hot 向量的长度，MegEngine pad_width 与之对应，表示一个元组。元组中的每个元素都是两个元素的元组.



.. code-block::: python

    import megengine  
  
    # 创建一个张量  
    x1 = megengine.tensor([[1, 2, 3], [4, 5, 6]])  
  
    # 对张量进行边界填充  
    y1 = megengine.module.Pad(x1, pad_width=((0, 1), (0, 1)), mode='constant', constant_val=0)  

    print(y1.numpy())



    import torch  
 
    # 创建一个张量  
    x2 = torch.tensor([[1, 2, 3], [4, 5, 6]])  
  
    # 对张量进行边界填充  
    y2 = torch.nn.functional.pad(x, pad=((0, 1), (0, 1)), mode='constant', value=0)  
  
 
