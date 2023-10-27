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


参数差异
--------

track_running_stats
~~~~~~~~~~~~~~~~~~~
Pytorch 中存在 ``track_running_stats`` 参数，当此参数为 True 时，在训练时会始终记录并更新（通过动量方法更新）全局的均值和方差，在测试时可以用这个均值和方差来归一化当前batch的输入（可以理解为这个均值和方差是所有训练样本的均值和方差，是全局的，对整个样本集的统计信息的描述更加准确一些）；当此参数为 False 时，不记录更新全局的均值和方差，如此的话，测试时用 batch 的测试数据本身的样本和方差来归一化。MegEngine 中无此参数。

momentum
~~~~~~~~~~
Pytorch 中存在 ``momentum`` 参数，在 ``track_running_stats`` 为 True 时， ``momentum`` 是训练过程中对均值和方差进行动量更新的超参参；在 ``track_running_stats`` 为 False 时， ``momentum`` 不起作用。MegEngine 中无此参数。



Pytorch 中的 ``num_features`` 对应 MegEngine 中的 ``num_channels``,表示通道数。

用法相似：
.. code-block:: python

    import torch

    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)

.. code-block:: python

    import megengine

    m = megengine.module.Linear(20, 30)
    input = megengine.random.normal(size=(128,20))
    output = m(input)
 
