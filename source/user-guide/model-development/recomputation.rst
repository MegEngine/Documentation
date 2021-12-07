.. _recomputation-guide:

=========================================
通过重计算节省显存（Recomputation）
=========================================

通常而言，使用更大的模型和更大的 Batch size 可以取得更好的训练效果，但随之而来的是更大的显存占用。

重计算（Recomputation）本质上是一种用时间换空间的策略，可以将它类比成一种 Tensor 缓存（Cache）策略，
当显存空间不足时，可以选择把一些前向计算的结果清除；
当需要再次用到这些计算结果时，再根据之前缓存的检查点（Checkpoint）去重新计算它们。
参考下面这个示意图，蓝色为占用的显存（ `图片来源 <https://www.zhihu.com/question/274635237/answer/755102181>`_ ）： 

.. panels::
   :container: +full-width text-center
   :card:

   Vanilla backprop
   ^^^^^^^^^^^^^^^^
   .. figure:: ../../_static/images/vanilla-backprop.gif
      :align: center

   ---
   Checkpointed backprop
   ^^^^^^^^^^^^^^^^^^^^^
   .. figure:: ../../_static/images/checkpointed-backprop.gif
      :align: center

MegEngine 将经典的重计算策略应用到了工程实现中，具体使用方式请参考以下页面：

.. toctree::
   :maxdepth: 1

   dtr/index
   sublinear/index
