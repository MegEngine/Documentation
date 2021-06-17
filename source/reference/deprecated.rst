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

.. list-table:: deprecated APIs
   :header-rows: 1
   :class: deprecated
   :widths: 40, 10, 10, 40

   * - 目标
     - 已弃用
     - 将被删除
     - 备选方案

   * - megengine.funtional.nn.interpolate
     - 1.3
     - 待定
     - megengine.functional.vision.interpolate

   * - megengine.funtional.nn.roi_pooling
     - 1.3
     - 待定
     - megengine.functional.vision.roi_pooling

   * - megengine.funtional.nn.roi_align
     - 1.3
     - 待定
     - megengine.functional.vision.roi_align

   * - megengine.funtional.nn.nms
     - 1.3
     - 待定
     - megengine.functional.vision.nms

   * - megengine.funtional.nn.resize
     - 1.3
     - 待定
     - megengine.functional.vision.resize

   * - megengine.funtional.nn.remap
     - 1.3
     - 待定
     - megengine.functional.vision.remap

   * - megengine.funtional.nn.nvof
     - 1.3
     - 待定
     - megengine.functional.vision.nvof

   * - megengine.funtional.nn.warp_affine
     - 1.3
     - 待定
     - megengine.functional.vision.warp_affine

   * - megengine.funtional.nn.warp_perspective
     - 1.3
     - 待定
     - megengine.functional.vision.warp_perspective

.. _deprecation-policy:

弃用政策
--------

我们正在制定相关政策中。
