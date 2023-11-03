.. _comparison-rnn-cell:

===============================
RNN 差异对比
===============================

.. panels::

  torch.nn.RNNCell
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.RNNCell(
        input_size,
        hidden_size,
        bias=True,
        nonlinearity='tanh',
        device=None,
        dtype=None
     )

  更多请查看 :py:class:`torch.nn.RNNCell`.

  ---

  megengine.module.RNNCell
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.RNNCell(
        input_size,
        hidden_size,
        bias=True,
        nonlinearity='tanh'
   )

  更多请查看 :py:class:`megengine.module.RNNCell`.


参数与使用方式基本一致。




 
