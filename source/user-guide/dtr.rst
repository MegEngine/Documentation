.. _dtr:

================================
动态图 Sublinear 显存优化（DTR）
================================

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

关于参数设置
------------

``memory_budget`` 表示显存阈值，它是一个软限制。当活跃的显存大小超过该阈值时，动态图显存优化会生效，
根据 DTR 的策略找出最优的 tensor 并释放其显存，直到活跃的显存大小不超过该阈值。因此实际运行时的活跃显存峰值比该阈值高一些属于正常现象。

一般情况下，显存阈值设得越小，显存峰值就越低，训练耗时也会越大；显存阈值设得越大，显存峰值就越高，训练耗时也会越小。

值得注意的是，当显存阈值接近显卡容量时，容易引发碎片问题。因为 DTR 是根据活跃的显存大小来执行释放操作的，释放掉的 tensor 在显卡上的物理地址很可能不连续。
例如：释放了两个物理位置不相邻的 100MB 的 tensor，仍然无法满足一次 200MB 显存的申请。此时就会自动触发碎片整理操作，对性能造成巨大影响。

下图是 ResNet50（batch size=200）在2080Ti（显存：11GB）上设定不同显存阈值后的性能表现。

.. image:: ../_static/images/resnet50_wrt_mb.png
   :align: center

可以看到，当显存阈值从 2 增长到 7 的时候，训练耗时是越来越低的，因为随着显存阈值升高，释放掉的 tensor 数量变少，重计算的开销降低；
当显存阈值增长到 8 和 9 的时候，可供申请的空闲显存总和已经不多，并且大概率地址不连续，导致需要不断地进行碎片整理，造成训练耗时显著增长。

因此在实际训练过程中，显存阈值需要用户根据模型和显卡的具体情况设定。
