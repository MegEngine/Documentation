.. _sublinear-guide:

==================================
使用 Sublinear 进行显存优化
==================================

.. warning::

   Sublinear 仅支持在静态图模式下开启，参考 :ref:`convert-dynamic-graph-to-static` 。

MegEngine 通过引入 `Sublinear <https://arxiv.org/pdf/2006.09616.pdf>`_ :footcite:p:`chen2016training` 技术来进行静态图下的显存优化。 

.. footbibliography::

用户在编译静态图时使用 :class:`~.jit.SublinearMemoryConfig` 设置 :class:`~.jit.trace` 
的参数 ``sublinear_memory_config``, 就可以打开 Sublinear 优化：

.. code-block:: python

   from megengine.jit import trace, SublinearMemoryConfig

   config = SublinearMemoryConfig()

   @trace(symbolic=True, sublinear_memory_config=config)
   def train_func(data, label, * , net, optimizer, gm):
        ...

