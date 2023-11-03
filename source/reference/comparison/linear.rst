.. _comparison-linear:

=========================
Linear 差异对比
=========================

.. panels::

  torch.nn.functional.Linear
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.functional.Linear(
        in_features,
        out_features,
        bias=True
     )

  更多请查看 :py:class:`torch.nn.functional.Linear`.

  ---

  megengine.module.Linear
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Linear(
        in_features, 
        out_features, 
        bias=True, 
        compute_mode='default'
     )

  更多请查看 :py:class:`megengine.module.Linear`.

参数差异
--------

compute_mode
~~~~~~~~~~~~~
MegEngine 中 ``compute_mode`` 参数，该参数用于指定计算模式，默认值为 “default”，表示不会对中间结果的精度有特殊要求。MegEngine 中无此参数。

用法相似：
.. code-block:: python
   
    import torch  
    import torch.nn.functional as F  
  
    # 定义权重矩阵和偏置项  
    weight = torch.randn(10, 5)  # 假设输入有10个特征，输出有5个特征  
    bias = torch.randn(5)  # 偏置项的大小与输出特征的数量相同  
  
    # 定义输入数据  
    input = torch.randn(3, 10)  # 假设有3个样本，每个样本有10个特征  
  
    # 执行线性变换  
    output = F.linear(input, weight, bias)  
  
    print(output)


.. code-block:: python
   
    import megengine  
    import megengine.module as M  
  
    # 定义权重矩阵和偏置项  
    weight = megengine.random.normal(size=(10,5))
    bias = megengine.random.normal(size=(5))
  
    # 定义输入数据  
    input = megengine.random.normal(size=(3, 10))  # 假设有3个样本，每个样本有10个特征  
  
    # 创建线性层实例  
    linear_layer = M.Linear(10, 5, bias=True)  
  
    # 执行线性变换  
    output = linear_layer(input)  
  
    print(output)


