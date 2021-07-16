.. _dtr-guide:

================================
动态图 Sublinear 显存优化（DTR）
================================

MegEngine 通过引入 `Dynamic Tensor Rematerialization <https://arxiv.org/pdf/2006.09616.pdf>`_
（简称 DTR）技术，进一步工程化地解决了动态图显存优化的问题，从而享受到大 Batchsize 训练带来的收益。


单卡训练
--------

使用方式十分简单，在训练代码之前添加两行代码：

.. code-block:: python

   import megengine as mge

   mge.dtr.eviction_threshold = "5GB" # set the eviction threshold to 5 GB
   mge.dtr.enable()                   # enable the DTR optimization

   # ... your training code

即可启用动态图的 Sublinear 显存优化。

.. note::

   从 MegEngine V1.5 开始，在开启 DTR 优化时不设置 ``eviction_threshold`` 也是被允许的。此时动态图显存优化将会在空闲的显存无法满足一次申请时生效，根据 DTR 的策略找出最优的 tensor 并释放其显存，直到该次显存申请成功。

分布式训练
----------

关于分布式训练的开启，请参考 :ref:`分布式训练 <distributed-guide>` 。

:class:`~.distributed.launcher` 将一个 function 包装成一个多进程运行的 function，你需要在这个 function 中定义 DTR 的参数：

.. code-block:: python

   import megengine as mge
   import megengine.distributed as dist

   @dist.launcher
   def main():

       mge.dtr.eviction_threshold = "5GB" # set the eviction threshold to 5 GB
       mge.dtr.enable()                   # enable the DTR optimization

       # ... your training code

更多设置
--------

参数介绍
~~~~~~~~~

.. list-table::
    :widths: 20 40
    :header-rows: 1

    * - 参数名
      - 实际含义
    * - ``eviction_threshold``
      - 显存阈值（单位：字节）。当被使用的显存总和超过该阈值后，显存优化会生效，根据 DTR 的策略找出最优的 tensor 并释放其显存，直到被使用的显存总和低于该阈值。默认值：0
    * - ``evictee_minimum_size``
      - 被释放显存的 tensor 的大小下限（单位：字节）。只有当 tensor 的大小不小于该下限时，才有可能被 DTR 策略选中释放其显存。默认值：1048576
    * - ``enable_sqrt_sampling``
      - 是否开启根号采样。设当前候选 tensor 集合的大小为 :math:`n`，开启该设置后，每次需要释放 tensor 时只会遍历 :math:`\sqrt{n}` 个候选 tensor。默认值：False

设置方法请参考 :py:mod:`~.dtr`

显存阈值的设置技巧
~~~~~~~~

``eviction_threshold`` 表示开始释放 tensor 的显存阈值。当被使用的显存大小超过该阈值时，动态图显存优化会生效，
根据 DTR 的策略找出最优的 tensor 并释放其显存，直到活跃的显存大小不超过该阈值。因此实际运行时的活跃显存峰值比该阈值高一些属于正常现象。

一般情况下，显存阈值设得越小，显存峰值就越低，训练耗时也会越大；显存阈值设得越大，显存峰值就越高，训练耗时也会越小。

值得注意的是，当显存阈值接近显卡容量时，容易引发碎片问题。因为 DTR 是根据活跃的显存大小来执行释放操作的，释放掉的 tensor 在显卡上的物理地址很可能不连续。
例如：释放了两个物理位置不相邻的 100MB 的 tensor，仍然无法满足一次 200MB 显存的申请。此时就会自动触发碎片整理操作，对性能造成巨大影响。

下图是 ResNet50（batch size=200）在2080Ti（显存：11GB）上设定不同显存阈值后的性能表现。

.. image:: ../../../_static/images/resnet50_wrt_mb.png
   :align: center

性能表现
''''''''

如上图（左）所示，

* 当显存阈值从 2 增长到 7 的时候，训练耗时是越来越低的，因为随着显存阈值升高，释放掉的 tensor 数量变少，重计算的开销降低；
* 当显存阈值增长到 8 和 9 的时候，可供申请的空闲显存总和已经不多，并且地址大概率不连续，导致需要不断地进行碎片整理，造成训练耗时显著增长，
* 当显存阈值增长到 10 之后，空闲的显存甚至无法支持一次 kernel 的计算，导致 OOM.

显存峰值
''''''''

如上图（右）所示，可以看出显存阈值和显存峰值之间有很大的差距。

* 当显存阈值在 2 到 5 之间时，显存峰值都在 8 左右；
* 当显存阈值在 6 到 9 之间时，显存峰值更是逼近显存总容量。

前者的原因是，DTR 只能保证在任意时刻，被使用的显存总和在显存阈值附近，但是这些被使用的显存的地址不一定连续。
被释放掉的空闲块会被 MegEngine 收集起来，当最大的空闲块大小也满足不了一次申请时, MegEngine 会从 CUDA 申请一段新的显存，
虽然被使用的显存总量在显存阈值附近，但是显存峰值上升了；
后者的原因是显存容量总共只有 11G，如果最大的空闲块大小也无法满足申请时只能靠碎片整理来满足申请，峰值不会变得更大。

所以从 ``nvidia-smi`` 上看到的显存峰值会显著高于显存阈值。

综上所述，在实际训练过程中，显存阈值需要用户根据模型和显卡的具体情况设定。

FAQ
---

Q：为什么 ``eviction_threshold=2GB`` 的时候训练耗时远高于 ``eviction_threshold=3GB`` 的训练耗时？

A：因为在该模型中，不可被释放的 tensor（例如：参数、执行当前算子需要用到的输入 tensor 和产生的输出 tensor 等等）的大小之和一直保持在 2GB 以上，所以几乎所有的 tensor 都会在不被用到的时刻立即被释放，所以会产生非常可观的重计算时间开销。

Q：为什么 ``eviction_threshold=2GB`` 的时候显存峰值高于 ``eviction_threshold=3GB`` 的显存峰值？

A：原因同上，由于 ``eviction_threshold=2GB`` 时重计算次数远多于 ``eviction_threshold=3GB`` ，需要频繁地申请和释放显存，
一旦某次空闲块大小不能满足申请，显存峰值就会增加，所以 ``eviction_threshold=2GB`` 时显存峰值大概率更高。

Q：用不同的 ``eviction_threshold`` 训练模型时的显存峰值可以估算吗？

A：很难。这取决于 DTR 策略释放和重计算了哪些 tensor，以及具体到某次显存申请时空闲块大小能否满足要求，这些都会影响最终的显存峰值。

