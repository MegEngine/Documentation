.. _comparison-embedding:

=========================
Pad 差异对比
=========================

.. panels::

  torch.nn.Embedding
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     torch.nn.Embedding(
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=None,
        device=None,
        dtype=None
     )

  更多请查看 :py:class:`torch.nn.Embedding`.

  ---

  megengine.module.Embedding
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  .. code-block:: python

     megengine.module.Embedding(
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=None,
        inital_weight=None,
        freeze=None
     )

  更多请查看 :py:class:`megengine.module.Embedding`.

参数差异
--------

padding_idx
~~~~~~~~~~~~
PyTorch ``padding_idx`` 参数表示在此区间内的参数及对应的梯度将会以 0 进行填充，MegEngine 中暂不支持，需要设置为 None。

max_norm
~~~~~~~~~~~~
PyTorch ``max_norm`` 如果给定，Embeddding 向量的范数（范数的计算方式由 norm_type 决定）超过了 max_norm 这个界限，就要再进行归一化，MegEngine 中暂不支持，需要设置为 None。

norm_type
~~~~~~~~~~~~
PyTorch ``norm_type`` 参数为 maxnorm 选项计算 p-范数的 p，默认值 2。MegEngine 中暂不支持，需要设置为 None。

scale_grad_by_freq
~~~~~~~~~~~~~~~~~~
PyTorch ``scale_grad_by_freq`` 参数是否根据单词在 mini-batch 中出现的频率，对梯度进行放缩，MegEngine 无此参数。

sparse
~~~~~~~
PyTorch ``sparse`` 参数表示是否使用稀疏更新，MegEngine 无此参数。

initial_weight
~~~~~~~~~~~~
MegEngine ``initial_weight`` 参数该模块的可学习权重，形状为(num_embeddings, embedding_dim) ，PyTorch 中无此参数。

.. code-block::: python
    

    import megengine
    import numpy as np
    embedding = MegEngine.module.Embedding(10, 3)
    input = megengine.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]，dtype=np.int32))
    embedding(input)


.. code-block::: python
    

    import torch

    input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    # an embedding matrix containing 10 tensors of size 3
    embedding_matrix = torch.rand(10, 3)
    torch.nn.embedding(input, embedding_matrix)
