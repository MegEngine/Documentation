.. _comparison-instance-norm:

===============================
InstanceNorm 差异对比
===============================

.. panels::

  torch.nn.InstanceNorm
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.Dropout(
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None
     )

  更多请查看 :py:class:`torch.nn.InstanceNorm`.

  ---

  megengine.module.InstanceNorm
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.InstanceNorm(
        num_channels,
        eps=1e-05,
        affine=True
     )

  更多请查看 :py:class:`megengine.module.InstanceNorm`.


使用差异
--------
Pytorch 支持 Instance Normalization1d（对输入的最后一维算均值和方差），Instance Normalization2d，Instance Normalization3d（将输入的后三维（即除了batchsize维和channel维）合并在一起算均值和方差）；MegEngine 仅提供对 2d 即 4 维输入数据（N,C,H,W）的支持。两者均要求输入输出的维度保持一致。

参数差异
--------

track_running_stats
~~~~~~~~~~~~~~~~~~~
Pytorch 中存在 ``track_running_stats`` 参数，当此参数为 True 时，在训练时会始终记录并更新（通过动量方法更新）全局的均值和方差，在测试时可以用这个均值和方差来归一化（可以理解为这个均值和方差是所有训练样本的均值和方差，是全局的，对整个样本集的统计信息的描述更加准确一些）；当此参数为 False 时，不记录更新全局的均值和方差，如此的话，测试时用 batch 的测试数据本身的样本和方差来归一化。MegEngine 中无此参数。

momentum
~~~~~~~~~~
Pytorch 中存在 ``momentum`` 参数，在 ``track_running_stats`` 为 True 时， ``momentum`` 是训练过程中对均值和方差进行动量更新的动量参数，在 ``track_running_stats`` 为 False 时， ``momentum`` 不起作用。MegEngine 中无此参数。



Pytorch 中的 ``num_features`` 对应 MegEngine 中的 ``num_channels``,表示通道数

用法相似：
.. code-block:: python

 
    import megengine as mge  
    import torch  
    import torch.nn as nn  
  
    # MegEngine 实例化  
    mge.set_default_config(dtype=mge.float32)  
    mge_instance_norm = mge.nn.InstanceNorm2d(num_features=64)  
  
    # PyTorch 实例化  
    torch_instance_norm = nn.InstanceNorm2d(64)  
  
    # 创建随机输入张量  
    input_tensor = mge.randn((32, 64, 32, 32))  
    torch_input_tensor = torch.randn(32, 64, 32, 32)  
  
    # 在 MegEngine 中使用 Instance Norm 2D  
    mge_output = mge_instance_norm(input_tensor)  
  
    # 在 PyTorch 中使用 Instance Norm 2D  
    torch_output = torch_instance_norm(torch_input_tensor)  
  
    # 打印输出张量的形状  
    print("MegEngine output shape:", mge_output.shape)  
    print("PyTorch output shape:", torch_output.shape)






 
