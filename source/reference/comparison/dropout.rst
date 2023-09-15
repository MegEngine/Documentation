.. _comparison-dropout:

=========================
Linear 差异对比
=========================

.. panels::

  torch.nn.Dropout
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.Dropout(
        p=0.5,
        inplace=False
     )

  更多请查看 :py:class:`torch.nn.Dropout`.

  ---

  megengine.module.Dropout
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Dropout(
        drop_prob=0.0
     )

  更多请查看 :py:class:`megengine.module.Dropout`.

参数差异
--------

drop_prob
~~~~~~~~~~~~~
MegEngine 中 ``drop_prob`` 参数对应 PyTorch 中的 ``p`` 参数，表示每个元素被丢弃（置为 0）的概率。


inplace
~~~~~~~~~~~~~
PyTorch 中 ``inplace`` 参数，表示在不更改变量的内存地址的情况下，直接修改变量的值，MegEngine 中无此参数，一般对网络训练结果影响不大。

