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

   对于 ``conda`` 用户, 可以选择通过在环境中先安装 ``pip``, 再按照上述方式进行 MegEngine 的安装；

.. warning::
 
   * MegEngine 包中集成了 CUDA 环境，但用户需确保环境中已经正确地安装好 GPU 设备相关驱动；
   * 由于 EAR 限制，目前官方发布的预编译包是基于 CUDA 10.1 的，参考 :ref:`cuda-compiling` 。

.. _build-from-source:

通过源码编译安装
----------------

如果包管理器安装的方式无法满足你的需求，例如：

* 我使用的 GPU 设备非 Nvidia 厂商的，比如用的是 AMD 等厂商的 GPU;
* 我使用的 Nvidia GPU 设备比较新或者比较旧，不在当前的设备支持列表中；
* 我希望更改一些其它的编译配置选项，启用一些默认关闭的特性。

则可以尝试自行通过源码编译安装。相关细节请参考 :src:`scripts/cmake-build/BUILD_README.md` . 

.. _cuda-compiling:

CUDA 编译支持现状
~~~~~~~~~~~~~~~~~

MegEngine CMake CUDA 编译的现状如下：

* CUDA 11.1 及以上编译能适配市面上所有的 Ampere 卡，适配 sm80+sm86
* CUDA 11.0 编译能适配 A100, 但不能适配 30 系卡，仅适配 sm80
* CUDA 10 不适配 Ampere 架构（官方发布的预编译包是基于 CUDA 10.1 的）

.. note::

   用户可以使用 ``cmake -DMGE_CUDA_GENCODE="-gencode arch=compute80, code=sm80"`` 自由指定。

.. warning::

   用户在编译前需要确定有 GPU 设备，以及确定环境中所使用的 CUDA 版本。

.. seealso::

   用户可在 `Compute Capability <https://developer.nvidia.com/cuda-gpus#compute>`_
   页面找到自己的 GPU 设备对应的计算兼容性版本。
