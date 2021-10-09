.. _midout:

==================
减少二进制文件体积
==================

midout 是 MegEngine 中用来减小生成的二进制文件体积的工具，有助于在空间受限的设备上部署应用。
midout 通过记录模型推理时用到的 opr 和执行流，使用 ``if(0)`` 关闭未被记录的代码段后重新编译，
利用 ``-ffunction-sections -fdata-sections -Wl,--gc-sections -flto`` 链接参数，可以大幅度减少静态链接的可执行文件的大小。
现在基于 MegEngine 提供模型验证工具 :ref:`Load and Run <load-and-run>` ,
展示怎样在某 AArch64 架构的 Android 端上裁剪 MegEngine 库。

编译静态链接的 load_and_run
---------------------------

端上裁剪 MegEngine 库需要一个静态连接 MegEngine 的可执行程序，编译方法详见 load-and-run 的编译部分。
稍有不同的是编译时需要先设置 load_and_run 静态链接 MegEngine.

.. code-block:: bash

    EXTRA_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF" ./cross_build_android_arm_inference.sh

查看一下 load_and_run 的大小：

.. code-block:: bash

    du ./build_dir/android/arm64-v8a/Release/install/bin/load_and_run
    23200

此时 load_and_run 大小超过 20MB. load_and_run 的执行，请参考下文“代码执行”部分。

裁剪 load_and_run
-----------------

MegEngine 的裁剪可以从两方面进行：

1. 通过opr 裁剪。在 dump 模型时，可以同时将模型用到的 opr 信息以 json 文件的形式输出，
   midout 在编译期裁掉没有被模型使用到的所有 opr.
2. 通过 trace 流裁剪。运行一次模型推理，根据代码的执行流生成 trace 文件，
   通过trace文件，在二次编译时将没有执行的代码段裁剪掉。

整个裁剪过程分为两个步骤：

1. 第一步，dump 模型，获得模型 opr 信息；通过一次推理，获得 trace 文件。
2. 第二步，使用MegEngine的头文件生成工具 ``tools/gen_header_for_bin_reduce.py`` 将 opr 信息和 trace 文件作为输入，
   生成 ``src/bin_reduce_cmake.h`` CMake 会自动维护这个文件，用户无需关心。
   当然也可以单独使用模型 opr 信息或是 trace 文件来生成 ``src/bin_reduce_cmake.h`` ，
   单独使用 opr 信息时，默认保留所有 kernel，单独使用 trace 文件时，默认保留所有 opr.

dump 模型获得 opr 类型名称
~~~~~~~~~~~~~~~~~~~~~~~~~~

一个模型通常不会用到所有的opr，根据模型使用的opr，可以裁掉那些模型没有使用的 opr. 
在转换模型时，我们可以通过如下方式获得模型的 opr 信息。
使用 ``load_and_run/dump_with_testcase_mge.py`` 准备模型时，加上 ``--output-strip-info`` 参数。

.. code-block:: bash

    python3 sdk/load-and-run/dump_with_testcase_mge.py --optimize-for-inference resnet50.pkl -o resnet50.mge --enable-fuse-conv-bias-nonlinearity --data "#rand(0,1)" --no-assert --output-strip-info

执行完毕后，会生成 ``resnet50.mge`` 和 ``resnet50.mge.json`` . 查看这个 JSON 文件，它记录了模型用到的 opr 名称。

.. code-block:: bash

    cat resnet50.mge.json
    {"hash": 238912597679531219, "dtypes": ["Byte", "Float32", "Int32"], "opr_types": ["Concat", "ConvBiasForward", "ConvolutionForward", "Elemwise", "GetVarShape", "Host2DeviceCopy", "ImmutableTensor", "MatrixMul", "MultipleDeviceTensorHolder", "PoolingForward", "Reshape", "Subtensor"], "elemwise_modes": ["ADD", "FUSE_ADD_RELU"]}

执行模型获得 trace 文件
~~~~~~~~~~~~~~~~~~~~~~~

基于 trace 的裁剪需要通过一次推理获得模型的执行 trace 文件。具体步骤如下：

1. CMake 构建时，打开 ``MGE_WITH_MIDOUT_PROFILE`` 开关，编译 load_and_run：

   .. code-block:: bash

      EXTRA_CMAKE_ARGS="-DMGE_WITH_MIDOUT_PROFILE=ON -DBUILD_SHARED_LIBS=OFF" ./cross_build_android_arm_inference.sh -r

   编译完成后，将 ``build_dir/android/arm64-v8a/Release/install/bin`` 下的 ``load_and_run`` 推至设备并执行：

   .. code-block:: bash

      ./load_and_run ./resnet50.mge

   得到如下输出：

   .. code-block:: bash

      mgb load-and-run: using MegBrain MegBrain 8.4.1(0) and MegDNN 9.3.0
      load model: 70.888ms
      === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
      === prepare: 4.873ms; going to warmup
      warmup 0: 877.578ms
      === going to run test #0 for 10 times
      iter 0/10: 481.445ms (exec=481.436,device=480.794)
      iter 1/10: 481.192ms (exec=481.183,device=481.152)
      iter 2/10: 480.430ms (exec=480.420,device=480.389)
      iter 3/10: 479.593ms (exec=479.585,device=479.553)
      iter 4/10: 479.851ms (exec=479.843,device=479.811)
      iter 5/10: 479.581ms (exec=479.572,device=479.541)
      iter 6/10: 480.174ms (exec=480.165,device=480.134)
      iter 7/10: 479.443ms (exec=479.435,device=479.404)
      iter 8/10: 479.987ms (exec=479.978,device=479.948)
      iter 9/10: 480.637ms (exec=480.628,device=480.598)
      === finished test #0: time=4802.333ms avg_time=480.233ms sd=0.688ms minmax=479.443,481.445

      === total time: 4802.333ms
      midout: 110 items written to midout_trace.20717

   注意到执行模型后，生成了 ``midout_trace.20717`` 文件，该文件记录了模型在底层执行了哪些 kernel.

2. 生成 ``src/bin_reduce_cmake.h`` 并再次编译 load_and_run：

   将生成的 ``midout_trace.20717`` 拷贝至本地，
   使用上文提到的头文件生成工具 ``gen_header_for_bin_reduce.py`` 生成 ``src/bin_reduce_cmake.h`` . 

   .. code-block:: bash

      python3 ./tools/gen_header_for_bin_reduce.py resnet50.mge.json midout_trace.20717

      EXTRA_CMAKE_ARGS="-DMGE_WITH_MINIMUM_SIZE=ON -DBUILD_SHARED_LIBS=OFF" ./scripts/cmake-build/cross_build_android_arm_inference.sh -r

   编译完成后，检查 load_and_run 的大小, 注意 MGE_WITH_MINIMUM_SIZE 不是非必须的，加上它 size 会更小，但同时会关闭一些编译选项：

   .. code-block:: bash

      du build_dir/android/arm64-v8a/release/install/bin/load_and_run
      2264

   此时 load_and_run 的大小减小到 2MB 多。推到设备上运行，得到如下输出：

   .. code-block:: bash

      mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
      [02 15:03:11 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
      load model: 74.208ms
      === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
      === prepare: 1.251ms; going to warmup
      warmup 0: 377.813ms
      === going to run test #0 for 10 times
      iter 0/10: 266.996ms (exec=266.993,device=266.854)
      iter 1/10: 266.717ms (exec=266.715,device=266.702)
      iter 2/10: 266.867ms (exec=266.865,device=266.855)
      iter 3/10: 267.172ms (exec=267.171,device=267.159)
      iter 4/10: 266.820ms (exec=266.819,device=266.807)
      iter 5/10: 266.852ms (exec=266.850,device=266.838)
      iter 6/10: 267.376ms (exec=267.374,device=267.363)
      iter 7/10: 267.005ms (exec=267.003,device=266.991)
      iter 8/10: 266.685ms (exec=266.684,device=266.671)
      iter 9/10: 266.767ms (exec=266.766,device=266.755)
      === finished test #0: time=2669.257ms avg_time=266.926ms sd=0.216ms minmax=266.685,267.376

      === total time: 2669.257ms

可以看到模型依然正常运行，并且运行速度正常。

使用裁剪后的 load_and_run
-------------------------

想要裁剪前后的应用能够正常运行，需要保证裁剪前后两次推理使用同样的命令行参数。
如果使用上文裁剪的 load_and_fun 的 fast-run功能（详见 :ref:`load-and-run` ）。

.. code-block:: bash

   ./load_and_run resnet50.mge --fast-run --fast-run-algo-policy resnet50.cache

可能得到如下输出：

.. code-block:: bash

   mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
   [02 15:05:50 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
   load model: 71.927ms
   === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
   === prepare: 1.251ms; going to warmup
    Trap

这是因为程序运行到了已经被裁剪掉的函数中，未被记录在 trace 文件中的函数的实现已经被替换成 ``trap()`` .
如果想要裁剪与 fast-run 配合使用，需要按如下流程获得 trace 文件：

1. 开启 fast-run 模式，执行未裁剪的 load_and_run 获得 ``.cache`` 文件，注意本次执行生成的 trace 应该被丢弃：

   .. code-block:: bash

      ./load_and_run resnet50.mge --fast-run --fast-run-algo-policy resnet50.cache

2. 使用 ``.cache`` 文件，执行 load_and_run 获得 trace 文件：

   .. code-block:: bash

       ./load_and_run resnet50.mge --fast-run-algo-policy resnet50.cache --winograd-transform

3. 如上节，将 trace 文件拷贝回本机，生成 ``src/bin_reduce_cmake.h`` ，再次编译 load_and_run 并推至设备。

4. 使用裁剪后的 load_and_run 的 fast-run 功能，执行同 2 的命令，得到如下输出：

   .. code-block:: bash

      mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
      [04 15:34:18 from_argv@mgblar.cpp:1392][WARN] enable winograd transform
      [04 15:34:18 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
      load model: 64.228ms
      === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
      === prepare: 260.058ms; going to warmup
      warmup 0: 279.550ms
      === going to run test #0 for 10 times
      iter 0/10: 209.177ms (exec=209.164,device=209.031)
      iter 1/10: 209.010ms (exec=209.008,device=208.997)
      iter 2/10: 209.024ms (exec=209.022,device=209.011)
      iter 3/10: 208.584ms (exec=208.583,device=208.573)
      iter 4/10: 208.669ms (exec=208.667,device=208.658)
      iter 5/10: 208.849ms (exec=208.847,device=208.838)
      iter 6/10: 208.787ms (exec=208.785,device=208.774)
      iter 7/10: 208.703ms (exec=208.701,device=208.692)
      iter 8/10: 208.918ms (exec=208.916,device=208.905)
      iter 9/10: 208.669ms (exec=208.667,device=208.656)
      === finished test #0: time=2088.390ms avg_time=208.839ms sd=0.191ms minmax=208.584,209.177

      === total time: 2088.390ms

使用其他 load_and_run 提供的功能也是如此，想要裁剪前后的应用能够正常运行，
需要保证裁剪前后两次推理使用同样的命令行参数。

多个模型合并裁剪
----------------
多个模型的合并裁剪与单个模型流程相同。 ``gen_header_for_bin_reduce.py`` 接受多个输入。
假设有模型 A 与模型 B, 已经获得 ``A.mge.json`` , ``B.mge.json`` 以及 ``A.trace`` , ``B.trace`` . 执行：

.. code-block:: bash

   python3 ./tools/gen_header_for_bin_reduce.py A.mge.json A.trace B.mge.json B.trace

裁剪基于 MegEngine 的应用
-------------------------

可以通过如下几种方式集成 MegEngine，对应的裁剪方法相差无几：

1. 参照 ``CMakeLists.txt`` ，将应用集成到整个 MegEngine 的工程。
   假设已经将 ``app.cpp`` 集成到 MegEngine ，那么会编译出静态链接 MegEngine 的可执行程序 ``app`` . 
   只需要按照上文中裁剪 load_and_run 的流程裁剪 ``app`` 即可。
2. 可能一个应用想要通过静态库集成 MegEngine。此时需要获得一个裁剪过的 ``libmegengine.a`` . 
   可以依然使用 load_and_run 运行模型获得 trace 文件，
   生成 ``src/bin_reduce_cmake.h`` ，并二次编译获得裁剪过的 ``libmegengine.a`` .
   此时，用户使用自己编写的构建脚本构建应用程序，并静态链接 ``libmegengine.a`` ，
   加上链接参数 ``-flto=full -ffunction-sections -fdata-sections -Wl,--gc-sections`` . 即可得到裁剪过的基于 MegEngine 的应用。
3. 上述流程亦可以用于 ``libmegengine.so`` 的裁剪，但是动态库的裁剪效果远不及静态库。
   原因在于 libmegengine.so 没有做符号隐藏，因此链接器不会进行激进的优化。
4. 经过上述流程，同样会在 build_dir 目录生成 liblite_shared.so, 此库裁剪力度和app裁剪效果相当，推荐这种方式。
5. 经过上述流程，同样会在 build_dir 目录生成 liblite_static_all_in_one.a, 此库裁剪力度和app裁剪效果相当，也推荐这种方式,
   同样需要在自己集成的构建系统加上链接参数 ``-flto=full -ffunction-sections -fdata-sections -Wl,--gc-sections``
6. 所有基于静态库集成的地方， 如果输出是一个动态库， 则需要自己维护最终目标的符号隐藏，才能达到最佳裁剪效果， 为了方便，
   强烈建议直接集成 liblite_shared.so.
