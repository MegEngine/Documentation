.. _lar-model-support:

模型以及相关支持
====================

Load and run 支持 MegEngine 以及 Megengine Lite 提供的相关模型格式的推理，各种模型的获取可以参考 :ref:`get-model`

.. note::

   Load and run 默认使用 Megengine 的相关接口进行模型推理。

.. versionadded:: 1.7

   提供了对 MegEngine Lite 接口的支持，使用时加上 ``--Lite`` 选项即可切换为 MegEngine Lite 相关的接口进行推理。
