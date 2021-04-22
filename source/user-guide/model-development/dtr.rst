.. _dtr:

=========================
动态图 Sublinear 显存优化
=========================

MegEngine 通过引入 `Dynamic Tensor Rematerialization <https://arxiv.org/pdf/2006.09616.pdf>`_ 
（简称 DTR）技术，进一步工程化地解决了动态图显存优化的问题，从而享受到大 Batchsize 训练带来的收益。

使用方式十分简单，在训练代码开头添加两行代码：

.. code-block:: python

   from megengine.utils.dtr import DTR
   dtr = DTR(memory_budget=5*1024**3)

即可启用动态图的 Sublinear 显存优化。

详细示例如下：

.. code-block:: python

   class DTR:
       r"""
       DTR implements `Dynamic Tensor Rematerialization <https://arxiv.org/abs/2006.09616>`_ in MegEngine.

       It is basically an online algorithm for checkpointing driven by certain eviction policies.

       .. code-block::
    
          from megengine.utils.dtr import DTR
          dtr = DTR(memory_budget=5*1024**3)

        # your training code

        """

       def __init__(self, memory_budget=0, tensor_lowerbound=1048576):
           r"""
           :param memory_budget: int. The threshold of memory usage. When memory
           usage exceeds this value, auto evict will be triggered.
           :param tensor_lowerbound: int. The minimum memory limit of the tensor
           that can be evicted. Default: 1MB.
           """


