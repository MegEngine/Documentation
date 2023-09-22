.. _comparison-lstm-cell:

===============================
RNN 差异对比
===============================

.. panels::

  torch.nn.LSTMCell
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.LSTM(
        input_size,
        hidden_size,
        bias=True
     )
:
  更多请查看 :py:class:`torch.nn.LSTMCell`.

  ---

  megengine.module.LSTMCell
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.LSTMCell(
        input_size,
        hidden_size,
        bias=True
   )

  更多请查看 :py:class:`megengine.module.LSTMCell`.


参数与使用方式基本一致。




 
