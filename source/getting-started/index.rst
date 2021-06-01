.. _getting-started:

========
新手入门
========
.. toctree::
   :hidden:

   quick-start
   beginner/intro

安装
----
MegEngine 可以使用 Python 包管理器 ``pip`` 直接进行安装：

.. code-block:: shell

   pip3 install megengine -f https://megengine.org.cn/whl/mge.html

如果想要安装特定的版本，或需要从源码进行编译？ :ref:`了解更多安装方式 <install>` 。

.. note::

   MegEngine 安装包中集成了使用 GPU 运行代码所需的 CUDA 环境，不用区分 CPU 和 GPU 版。
   如果想要运行 GPU 程序，请确保机器本身配有 GPU 硬件设备并安装好驱动。

   如果你想体验在云端 GPU 算力平台进行深度学习开发的感觉，欢迎访问 `MegStudio <https://studio.brainpp.com/>`_ 平台。


接下来做什么
------------

* 如果你有其它深度学习框架使用经验，请参考 `MegEngine 快速上手 <./quick-start.html>`_ 教程，以便快速熟悉 :ref:`API <reference>` .
* 如果你是机器学习/深度学习领域的初学者，想要通过学习 MegEngine 的使用加深对基础知识的理解，
  我们为你准备了一系列 :ref:`deep-learning` ，希望能有所帮助！

