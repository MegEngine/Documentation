.. _comparison-layernorm:

=========================
LayerNorm 差异对比
=========================

.. panels::

  torch.nn.LayerNorm
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.LayerNorm(
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None
     )

  更多请查看 :py:class:`torch.nn.LayerNorm`.

  ---

  megengine.module.LayerNorm
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.LayerNorm(
        normalized_shape,
        eps=1e-05,
        affine=True,
        ** kwargs
     )

  更多请查看 :py:class:`megengine.module.LayerNorm`.


参数差异
--------

device
~~~~~~~~~~~~

Pytorch 中的 device 表示设备类型，MegEngine 无此参数，一般对网络训练结果影响不大，可直接删除。


dtype
~~~~~~~~~~~~

Pytorch 中的 dtype 表示参数类型，MegEngine 无此参数，一般对网络训练结果影响不大，可直接删除。




      





 
  
  
   
