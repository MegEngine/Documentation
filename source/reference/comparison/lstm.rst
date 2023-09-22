.. _comparison-lstm:

===============================
RNN 差异对比
===============================

.. panels::

  torch.nn.LSTM
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.LSTM(
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False
        dropout=0,
        bidirectional=False,
        proj_size=0   
     )

  更多请查看 :py:class:`torch.nn.LSTM`.

  ---

  megengine.module.LSTM
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.LSTM(
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False
        dropout=0,
        bidirectional=False,
        proj_size=0  
   )

  更多请查看 :py:class:`megengine.module.LSTM`.


参数与使用方式基本一致。




 
