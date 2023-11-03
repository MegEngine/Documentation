.. _comparison-rnn:

===============================
RNN 差异对比
===============================

.. panels::

  torch.nn.RNN
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity='tanh',
        bias=True,
        batch_first=False,
        drop=0,
        Bidirectional=False
     )

  更多请查看 :py:class:`torch.nn.RNN`.

  ---

  megengine.module.RNN
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.RNN(
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity='tanh',
        bias=True,
        batch_first=False,
        drop=0,
        Bidirectional=Fals
     )

  更多请查看 :py:class:`megengine.module.RNN`.


参数与使用方式基本一致。




 
