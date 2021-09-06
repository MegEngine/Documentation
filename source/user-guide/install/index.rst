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

环境依赖
~~~~~~~~

大多数编译 MegEngine 的依赖位于 :src:`third_party` 目录，可以通过以下命令自动安装：

.. code-block:: shell

   ./third_party/prepare.sh
   ./third_party/install-mkl.sh

上述过程中需要从国外获取软件源，建议使用比较通畅的网络环境。

一些依赖需要手动安装：

* `CUDA <https://developer.nvidia.com/cuda-toolkit-archive>`_ (>=10.1), 
  `cuDNN <https://developer.nvidia.com/cudnn>`_ (>=7.6), 
  如果需要编译支持 CUDA 的版本。
* `TensorRT <https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html>`_ (>=5.1.5) ，
  如果需要编译支持 TensorRT 的版本。
* LLVM/Clang(>=6.0) ，如果需要编译支持 Halide JIT 的版本（默认开启）。
* Python(>=3.5), Numpy, SWIG(>=3.0), 如果需要编译生成 Python 模块。
 
其中 CUDA 三件套的环境变量设置如下：

.. code-block:: shell

   export CUDA_ROOT_DIR=/path/to/cuda/lib
   export CUDNN_ROOT_DIR=/path/to/cudnn/lib
   export TRT_ROOT_DIR=/path/to/tensorrt/lib

开始编译
~~~~~~~~

* :src:`scripts/cmake-build/host_build.sh` 用于本地编译。

  参数 ``-h`` 可用于查询脚本支持的参数:

  .. code-block:: shell

     scripts/cmake-build/host_build.sh -h

* :src:`scripts/cmake-build/cross_build_android_arm_inference.sh` 用于 ARM-安卓 交叉编译。

  参数 ``-h`` 可用于查询脚本支持的参数:

  .. code-block:: shell

     scripts/cmake-build/cross_build_android_arm_inference.sh -h

* :src:`scripts/cmake-build/cross_build_linux_arm_inference.sh` 用于 ARM-Linux 交叉编译。

  参数 ``-h`` 可用于查询脚本支持的参数:

  .. code-block:: shell

     scripts/cmake-build/cross_build_linux_arm_inference.sh -h

* :src:`scripts/cmake-build/cross_build_ios_arm_inference.sh` 用于 iOS 交叉编译。

  参数 ``-h`` 可用于查询脚本支持的参数:

  .. code-block:: shell

     scripts/cmake-build/cross_build_ios_arm_inference.sh -h

更多细节请参考 :src:`scripts/cmake-build/BUILD_README.md` . 
