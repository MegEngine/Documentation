.. _megengine-reference:

========
API 参考
========
.. meta::
   :description lang=zh-Hans: 天元 MegEngine API 定义与使用介绍，提供权威参考信息。
   :description lang=en: MegEngine API manual and official reference.
   :keywords lang=zh-Hans: 天元, 深度学习, 接口, 规格, 定义, 查询, 引用, 参考, 用例
   :keywords lang=en: MegEngine, deep learning, interface, specification, define, reference, example

:对应版本: |version|
:更新时间: |today|

当前板块详细列举了 MegEngine 中包含的模块、对象和方法，描述了它们的功能和作用，以便于检索和浏览。
想要了解如何使用 MegEngine, 请参阅 :ref:`getting-started` 和 :ref:`user-guide` 页面。

Python API
----------
.. note::

   MegEngine 在 GitHub 上的 Python 包源码位置在：:src:`imperative/python/megengine`

   WEB 文档中仅仅列举出了面向用户提供的公开接口，并提供兼容性保证。
   私有模块（如 Core）以及不做兼容性保证的模块（如 Utils）中的接口将不会在此出现，
   但你始终可以在源码中找到它们。

.. note::

   如果你发现部分 API 内容未被及时更新成中文，欢迎通过 Crowdin_ 平台协助翻译。

.. _Crowdin: https://crowdin.com/project/megengine/zh-CN#/Documentation/main/locales/zh_CN/LC_MESSAGES/reference/api

.. warning::

   * 对于具有 NumPy / Pytorch 等框架使用经验的用户，推荐参考 :ref:`comparison` 页面；
   * 如果你正在从旧版本的 MegEngine 迁移到最新版本，请务必阅读 :ref:`deprecated` 页面。

.. toctree::
   :hidden:
   :maxdepth: 1

   megengine
   data
   functional
   module
   module.init
   autodiff
   optimizer
   jit
   amp
   dtr
   distributed
   quantization
   traced_module
   hub
   random
   config

.. toctree::
   :caption: API 相关说明
   :maxdepth: 1
   :hidden:

   comparison
   deprecated
   environment-variables

Index
-----

* :ref:`genindex`
* :ref:`modindex`
