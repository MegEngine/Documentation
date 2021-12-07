.. _dtr-guide:

================================
使用 DTR 进行显存优化
================================

MegEngine 通过引入 `DTR <https://arxiv.org/pdf/2006.09616.pdf>`_ :footcite:p:`kirisame2021dynamic` 技术来进行动态图下的显存优化，同时也支持在静态图下开启。 

.. footbibliography::

DTR 使用与配置方式
---------------------

在训练代码之前添加一行代码，即可启用动态图的 DTR 显存优化：

>>> megengine.dtr.enable()

.. versionadded:: 1.5
   用户现在可以直接开启 DTR 优化，不再需要设置一个显存阈值  :py:data:`~.dtr.eviction_threshold` 作为触发条件。
   MegEngine 默认会在当前空闲的显存无法满足一次申请时尝试进行优化，根据 DTR 策略找出最优的 Tensor 并释放其显存，直到该次显存申请成功。

   而在 1.4 版本中，必须提前设置好显存阈值，才能开启 DTR 显存优化：

   >>> megengine.dtr.eviction_threshold = "5GB"
   >>> megengine.dtr.enable()
   
.. admonition:: 显存阈值的设置技巧
   :class: note

   一般情况下，显存阈值设得越小，显存峰值就越低，训练耗时也会越大；显存阈值设得越大，显存峰值就越高，训练耗时也会越小。
   但值得注意的是，当显存阈值接近显卡容量时，容易引发碎片问题。因为 DTR 是根据活跃的显存大小来执行释放操作的，释放掉的 Tensor 在显卡上的物理地址很可能不连续。
   例如：释放了两个物理位置不相邻的 100MB 的 Tensor, 仍然无法满足一次 200MB 显存的申请。此时就会自动触发碎片整理操作，对性能造成巨大影响。

.. admonition:: 结合分布式训练
   :class: note

   在分布式情景下，我们通常会使用 :class:`~.distributed.launcher` 将一个函数包装成一个多进程运行的函数，
   此时如果想要开启 DTR 显存优化，需要在被包装的函数中定义 DTR 的参数：

   .. code-block:: python

      @dist.launcher
      def main():

          megengine.dtr.enable()

   如果你还不清楚相关概念，可参考 :ref:`distributed-guide` 页面了解细节。

.. seealso::

   还有一些其它的接口如 :py:data:`~.dtr.evictee_minimum_size`, :py:data:`~.dtr.enable_sqrt_sampling` ...
   可以对 DTR 策略进行自定义，更多配置说明请参考 :py:mod:`~.dtr` 模块的 API 文档页面。

在静态图下开启 DTR
-------------------

用户在编译静态图时使用 :class:`~.jit.DTRConfig` 设置 :class:`~.jit.trace` 的参数 ``dtr_config``, 就可以打开 DTR 优化：

.. code-block:: python

   from megengine.jit import trace, DTRConfig

   config = DTRConfig(eviction_threshold=8*1024**3)

   @trace(symbolic=True, dtr_config=config)
   def train_func(data, label, * , net, optimizer, gm):
       ...

