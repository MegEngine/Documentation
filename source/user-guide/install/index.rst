.. _install:

==================
如何安装 MegEngine
==================

.. note::

   MegEngine 目前支持在以下环境安装 Python 包：

   * 操作系统： Linux-64bit/Windows-64bit/MacOS-10.14 及其以上
   * Python 版本：3.5 到 3.8

   其中 MacOS 只支持 Intel x86 CPU；
   此外，MegEngine 也支持在很多其它平台上进行推理运算。

.. _install-with-pip:

通过包管理器安装
----------------

通过 ``pip`` 包管理器安装 MegEngine 的命令如下：

.. code-block:: shell

   python3 -m pip install --upgrade pip  # 将 pip 更新到最新版本
   python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html

.. note::

   * 对于 ``conda`` 用户, 可以选择通过在环境中先安装 ``pip``,
     再按照上述方式进行 MegEngine 的安装。

.. _build-from-source:

通过源码编译安装
----------------

如果包管理器安装的方式无法满足你的需求，则可以尝试自行通过源码编译安装。

相关细节请参考 :src:`scripts/cmake-build/BUILD_README.md` . 
