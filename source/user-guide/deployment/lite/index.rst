.. _megengine-lite:

==============================
使用 MegEngine Lite 做模型部署
==============================

.. note::

   当前文档处于建设状态，子页面内容和组织可能产生变化。

MegEngine Lite 是对 MegEngine 的推理功能的封装，具有配置统一、接口完备、第三方硬件支持等优点。

将 Lite 结合到整个生产流程，步骤如下：

#. 模型准备：模型开发完成后，你需要通过 :ref:`dump <dump>` 得到 Lite 支持的模型格式；
#. 环境准备：按照使用环境，:ref:`从源码编译 MegEngine <build-from-source>` .
   （ ``MGE_WITH_LITE`` 变量默认为 ``ON`` ）；
#. 工程接入：调用 MegEngine Lite 提供的接口，实现接入；
#. 集成编译：将之前编译得到的 Lite 库文件和对应调用代码一同编译成 SDK.

.. toctree::
   :maxdepth: 1

   cpp
   example
   python
   third-party


