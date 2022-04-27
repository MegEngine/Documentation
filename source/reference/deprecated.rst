.. _deprecated:

===============
Deprecated APIs
===============

我们在开发 MegEngine 的过程中始终注意着 API 的兼容性，但某些时候由于一些原因需要更改接口。
在这种情况下，我们会将它们标记为已弃用（Deprecated），并在后续几个版本中进行保留。

（更多详细信息，请参考 :ref:`deprecation-policy` ）

.. note::

   如果你想要阅读 <=0.6 版本的 MegEngine API 文档，请访问：

   https://megengine.org.cn/api/0.6/zh/api.html

不推荐使用/弃用的接口
---------------------

.. tabularcolumns:: |>{\raggedright}\Y{.4}|>{\centering}\Y{.1}|>{\centering}\Y{.12}|>{\raggedright\arraybackslash}\Y{.38}|

.. list-table::
   :header-rows: 1
   :class: deprecated
   :widths: 40, 10, 10, 40

   * - 曾使用
     - 已弃用
     - 移除
     - 替代品

   * - Tensor.reset_zero
     - 1.0
     - 待定
     - ``tensor_name[:] = 0``
   * - Tensor.set_value
     - 1.0
     - 待定
     - ``tensor_name[:] = value``
   * - functional.debug_param.get_execution_strategy  [1]_
     - 1.9
     - 待定
     - * :attr:`~.config.benchmark_kernel`
       * :attr:`~.config.deterministic_kernel`
   * - functional.debug_param.set_execution_strategy [1]_
     - 1.9
     - 待定
     - * :attr:`~.config.benchmark_kernel`
       * :attr:`~.config.deterministic_kernel`

.. [1] 这两个接口尚未移除，但已经在文档 API 页面中隐藏（表明不再推荐），
       替换为 :mod:`~.config` 配置：

       * ``benchmark_kernel = False`` 对应 ``HEURISTIC`` 策略，否则对应 ``PROFILE``;
       * ``deterministic_kernel = True`` 对应 ``REPRODUCIBLE`` 配置。

.. _deprecation-policy:

弃用政策
--------

我们正在制定相关政策中。
