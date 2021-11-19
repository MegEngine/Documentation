.. _megengine-lite:

==============================
使用 MegEngine Lite 做模型部署
==============================

简介
------------

MegEngine Lite 是对 MegEngine 的推理功能的封装，具有配置统一、接口完备、化繁为简、支持第三方硬件等优点。

.. figure:: ../../../_static/images/lite-structure.png
   :align: center

* **配置统一** ：配置选项可以存为json文件，在模型制作阶段和模型调用阶段均可加载，获得同一份配置选项。
* **接口完备** ：从业务需要出发，涵盖了推理和部署所涉及的网络和张量的接口。
* **化繁为简** ：lite所封装的接口，是MegEngine接口中只和推理部署相关的部分。对于只对推理部署、而非训练测试感兴趣的开发者，学习和上手的成本降到最低。
* **支持第三方硬件** ：lite里提供了新硬件接入的方案：新硬件接口被封装后，以“插件”的方式注册在lite上，而弱化对lite或MegEngine版本号的依赖。


编译
------------

将 Lite 结合到整个生产流程，步骤如下：

#. 模型准备：模型开发完成后，你需要通过 :ref:`dump <dump>` 得到 Lite 支持的模型格式；
#. 环境准备：按照使用环境，:ref:`从源码编译 MegEngine <build-from-source>` .
   （ ``MGE_WITH_LITE`` 变量默认为 ``ON`` ）；
#. 工程接入：调用 MegEngine Lite 提供的接口，实现接入；
#. 集成编译：将之前编译得到的 Lite 库文件和对应调用代码一同编译成 SDK.


文档目录
-----------

.. toctree::
   :maxdepth: 1

   cpp-basic
   cpp-advanced
   pylite-basic
   pylite-advanced
   third-party


