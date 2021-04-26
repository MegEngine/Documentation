.. _dtr:

=========================
动态图 Sublinear 显存优化
=========================

MegEngine 通过引入 `Dynamic Tensor Rematerialization <https://arxiv.org/pdf/2006.09616.pdf>`_
（简称 DTR）技术，进一步工程化地解决了动态图显存优化的问题，从而享受到大 Batchsize 训练带来的收益。


单卡训练
--------

使用方式十分简单，在训练代码之前添加两行代码：

.. code-block:: python

    from megengine.utils.dtr import DTR
    dtr = DTR(memory_budget=5*1024**3) # 设置显存阈值为 5 GB

    # ... 你的训练代码

即可启用动态图的 Sublinear 显存优化。



分布式训练
----------

关于分布式训练的开启，请参考 :ref:`分布式训练 <distribution>`

:class:`~.distributed.launcher` 将一个 function 包装成一个多进程运行的 function，你需要在这个 function 中定义 DTR 的参数：

.. code-block:: python

    import megengine.distributed as dist

    @dist.launcher
    def main():

        from megengine.utils.dtr import DTR
        dtr = DTR(memory_budget=5*1024**3) # 设置显存阈值为 5 GB

        # ... 你的训练代码
