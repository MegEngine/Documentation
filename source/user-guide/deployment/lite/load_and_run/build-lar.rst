.. _build-lar:

Load and run 获取和编译
===============================

Load and run 获取
-------------------

.. note::

   Load and run 是 MegEngine 配套的 C++ 推理工具，支持 MegEngine 以及 MegEngine Lite 相关的推理接口。

   如果需要获取 ``Load and run`` 的源码，可以拉取 `Megengine <https://github.com/MegEngine/MegEngine.git>`__  仓库，在不同版本的相应目录下即可找到相关文件。 

.. versionadded:: 1.7

      ``Load and run`` 源码从 MegEngine 工程 ``sdk/load-and-run`` 目录迁移到 ``Lite/load_and_run`` 目录 。

Load and run 编译
-------------------

Load and run 可以使用 MegEngine 提供的 `脚本 <https://github.com/MegEngine/MegEngine/tree/master/scripts/cmake-build>`__ 以及通用的 CMake 方法进行编译，具体使用方法如下：

**脚本编译**

.. code:: bash

   cd <megengine_dir>
   scripts/cmake-build/host_build.sh -e load_and_run #主机编译
   scripts/cmake-build/cross_build_android_arm_inference.sh -e load_and_run  -a <arm_arch> #arm交叉编译

.. note::

   这一脚本会自动创建一个 build_dir 文件夹，该文件夹下根据不同的编译选项有递归创建的不同的文件夹，这些不同的文件夹下都会有 build 和 install 文件夹，Load and run 在 ``build/Lite/load_and_run`` 以及 ``install/bin`` 下面各有一份。
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
-------------------

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
 
