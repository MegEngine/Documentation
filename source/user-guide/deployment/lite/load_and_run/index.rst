.. _load-and-run:

使用 Load and run 测试与验证模型
================================

.. admonition:: Load and run 简介

   Load and run （简称 LAR ） 是 MegEngine 提供与之相关的配套的推理工具，主要提供了以下功能：

   * 测试各类模型的推理性能，获取模型推理结果以及推理相关信息；
   * 测试验证不同模型优化方法的效果。

   本工具提供了大量的配置选项（可以执行 ``./load_and_run --help`` 查看详细的帮助文档），
   这些选项可以灵活的适配各种推理所需的场景，能够在不同设备上运行 MegEngine 模型，
   给出推理结果以及相应的性能分析结果，是基于 MegEngine 的模型研究和单模型的推理部署的很重要的生产工具。

.. admonition:: 模型以及相关支持

   Load and run 支持 MegEngine 以及 Megengine Lite 提供的相关模型格式的推理，各种模型的获取可以参考 :ref:`get-model` 。
   **默认使用 MegEngine 的原生接口进行模型推理。**

.. versionadded:: 1.7

   提供了对 Lite 接口的支持，使用时加上 ``--lite`` 选项即可切换为 Lite 接口进行推理。

.. note::

   * 目前发布的版本我们开放了对 CPU（x86, x64, ARM, ARMv8.2）和 GPU（CUDA）平台的支持。
   * 为了方便使用，MegEngine 中还提供了 :ref:`load-and-run-py` 。


有关 Load and run 工具的各种使用细节主要包括以下几个部分：

.. toctree::
   :maxdepth: 1

   basic-usage
   options-list
   inference-optimize
   profile-model
   accuracy-analysis
   debug
   python-interface

在使用 Load and run 之前，我们还需要了解如何获取和编译它。

Load and run 获取与编译
-----------------------

.. seealso:: Load and run 源码位于 :src:`lite/load-and-run` 目录内。

.. versionchanged:: 1.7 从 ``sdk/load-and-run`` 目录迁移到 ``lite/load_and_run`` 目录内。

Load and run 可以使用 MegEngine 提供的 `脚本 <https://github.com/MegEngine/MegEngine/tree/master/scripts/cmake-build>`__ 以及通用的 CMake 方法进行编译，具体使用方法如下：

**脚本编译**

.. code:: bash

   cd <megengine_dir>
   scripts/cmake-build/host_build.sh -e load_and_run 
   scripts/cmake-build/cross_build_android_arm_inference.sh -e load_and_run  -a <arm_arch> #arm交叉编译

.. note::

   这一脚本会自动创建一个 build_dir 文件夹，该文件夹下根据不同的编译选项有递归创建的不同的文件夹，这些不同的文件夹下都会有 build 和 install 文件夹，
   Load and run 在 ``build/Lite/load_and_run`` 以及 ``install/bin`` 下面各有一份。
   install 下的相比于 build 目录下的文件，会 strip 一些调试信息，调试时建议使用 build 目录下的文件使用。

**Cmake 编译**

.. code:: bash

   cd <megengine_dir>
   mkdir <build_dir> && cd <build_dir>
   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMGE_WITH_CUDA=OFF -DMGE_INFERENCE_ONLY=ON
   ninja load_and_run

.. note::

   如果需要设置不同的参数可以参考 `脚本 <https://github.com/MegEngine/MegEngine/tree/master/scripts/cmake-build>`__ 中实现，推荐使用 MegEngine 提供的脚本进行编译


常用编译参数说明
~~~~~~~~~~~~~~~~

不同设备和不同应用场景下上述编译方法所需要的参数差异比较大，CMake 的方法可以更加灵活的控制各种编译参数，但代价是学习使用成本很高，需要对 megenine 很熟悉。所以在此不做过多介绍，感兴趣的读者可以参考脚本编译方法中的脚本内容学习研究。下面就与 Load and run 相关的脚本参数进行说明。 

**主机非交叉编译脚本scripts/cmake-build/host_build.sh**

该脚本适用于编译在本地主机上运行的 load_and_run，常见用法如下：

.. note::

   * 在 linux 和 macos 中该脚本构建的程序位数和系统位数相当
   * 在 Windows 下， 默认构建 64bit 程序， 可以通过设置 ``-m`` 选项编译 32bit 程序，该选项仅在 Windows 下适用
   * 其他默认参数可以通过设置 ``-h`` 选项获取
   * CMake 用到的各个 options 可以通过设置 ``-l`` 来查看

.. code:: bash

   #设置编译目标为 load_and_run
   scripts/cmake-build/host_build.sh -e load_and_run

   #开启 Debug 选项
   scripts/cmake-build/host_build.sh -d -e load_and_run

   #开启 CUDA
   scripts/cmake-build/host_build.sh -c -e load_and_run

   #编译前移除之前的 build 文件
   scripts/cmake-build/host_build.sh -r -e load_and_run

   #编译时导入其他 CMake 设置参数
   EXTRA_CMAKE_ARGS=“-DMGB_ENABLE_JSON=1” scripts/cmake-build/host_build.sh -e load_and_run

.. warning::

   * 默认情况下脚本会构建所有的目标，同时会把所有目标和头文件安装到前述的 install 目录下。
   * 当使用 ``-e`` 编译单个目标时，只会构建指定的目标，不会进行 ``install`` 操作
   

**android 交叉编译脚本scripts /cmake-build/cross_build_android_arm_inference.sh**

该脚本适用于在本地机器上交叉编译 arm android 设备运行的程序，常见的用法如下：
         
.. code:: bash

   #默认参数编译
   scripts/cmake-build/cross_build_android_arm_inference.sh -e load_and_run

   #开启 Debug 选项
   scripts/cmake-build/cross_build_android_arm_inference.sh -d -e load_and_run

   #指定编译目标架构（默认为 arm64-v8a ）
   scripts/cmake-build/cross_build_android_arm_inference.sh -a armeabi-v7a -e load_and_run

   #开启 fp16 编译选项（即编译选项 -march=armv8.2a+fp16 ）
   scripts/cmake-build/cross_build_android_arm_inference.sh -f -e load_and_run

   #禁用 fp16
   scripts/cmake-build/cross_build_android_arm_inference.sh -k -e load_and_run

   #编译前移除之前的 build 文件
   scripts/cmake-build/cross_build_android_arm_inference.sh -r -e load_and_run

   #编译时导入其他 CMake 设置参数
   EXTRA_CMAKE_ARGS=“-DMGE_BUILD_WITH_ASAN=ON” scripts/cmake-build/cross_build_android_arm_inference.sh -e load_and_run

.. note::

   * 当加入 ``-r`` 参数后，会完整编译，这种情况耗时会比较多，当仅仅修改了一些 ``cpp/h`` 或者新增加一些 ``cpp/h`` 时， 可以不加此参数来进行增量构建。
   * 当删除一些 cpp/h 或者修改了 CMakeLists.txt， 因为 CMake 本身存在一些缺陷，这个时候可以先删除 CMake 的缓存，然后也可以不加 ``-r`` 来完成增量构建。

   .. code:: bash

      find build_dir/ -name CMakeCache.txt | xargs rm -rf 
      
   * 仅仅当需要类似发版本的动作时，才建议加上 ``-r``
   * 脚本各个参数可以组合使用，从而达到不同的编译目标, 其他平台编译以及更多的使用方法可以参考 `BUILD_README <https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md>`__
 
