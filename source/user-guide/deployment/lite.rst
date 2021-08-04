.. _lite:

=====================
使用Lite做模型部署
=====================

Lite是对 MegEngine 的推理功能的封装。

模型准备
---------------------
同“将模型部署到 C++ 环境”页面的“将模型序列化并导出”一节。


C++集成——编译
------------------

lite支持两种编译方式: 基于内部megvii3 workspace的 **bazel** 编译和 **cmake** 编译


bazel编译
~~~~~~~~~~~~~

下载代码
''''''''''

.. code-block:: bash

    git clone git@git-core.megvii-inc.com:brain-sdk/megvii3.git  //下载workspace
    cd megvii3
    ./utils/bazel/get_bazel.sh                                   //下载bazel编译工具到megvii3目录下，建议添加到环境变量里面去
 
    git submodule update --init brain/{midout,megbrain}          //下载megbrain代码
 
编译出现工具链下载不了的问题，需要添加一些权限，可以找engine的同事咨询

编译方法
'''''''''

各个平台的编译方法如下：

bazel 编译支持平台和compier查询方法：

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu=xxxxx  #xxxxx是没有注册平台,bazel命令执行错误,则会将所有支持的cpu和compiler组合显示出来

编译 x86 CUDA 版本

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu="k8" --compiler="gcc7_cuda10" -c opt

编译 x86 CPU 版本

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu="k8" --compiler="gcc9" -c opt

编译android arm64 OpenCL 版本

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu=android_aarch64  -c opt --define enable_opencl=1 --define enable_opencl_search=1

编译android armv7 CPU 版本

.. code-block:: bash
    
    ./bazel build //brain/megbrain/lite:lite_static --cpu=android_armv7 -c opt

编译android arm64 CPU 版本

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu=android_aarch64 -c opt

编译android arm64 CPU v8.2 版本

.. code-block:: bash

    ./bazel build //brain/megbrain/lite:lite_static --cpu=android_aarch64 --copt -march=armv8.2-a+fp16 -c opt

上面编译的目标是lite中lite_static这个目标，该目标只会编译lite_static代码为一个静态库。如果需要使用lite的动态库：需要把以上命令中的 **lite_static** 改为 **lite_shared** 。如果想把Lite作为bazel编译其他目标的依赖项，可在bazel的BUILD中添加：internal_deps = [":lite_static",]或[":lite_shared",]。

cmake编译
~~~~~~~~~~~~

下载代码
'''''''''

.. code-block:: bash

    git clone git@git-core.megvii-inc.com:brain-sdk/MegBrain.git  //下载megbrain代码
    cd MegBrain                                                   //进入Megbrain代码
    git checkout master                                           //切换到master分支
    cd third_party                                                //进入third_party
    bash ./prepare.sh                                             //准备依赖库

编译方法
'''''''''

建议用我们准备的脚本（在megbrain/scripts/cmake-build目录下）进行编译：

.. code-block:: bash

    host_build.sh #编译host端代码，host一般是x86的linux机器的脚本
 
    cross_build_ios_arm_inference.sh  #交叉编译ios的脚本
 
    cross_build_android_arm_inference.sh #交叉编译android arm 首先需要先配置NDK_ROOT环境变量，配置NDK对应路径
 
    cross_build_linux_arm_inference.sh #交叉编译linux arm

具体参数可以通过脚本的**-h**参数查看。


C++集成——工程样例参考
---------------------

Lite的代码中提供了含有常用功能的推理工程样例在/lite/example/mge。这里所有的样例都是使用 shufflenet 来进行演示。详细信息参考：https://git-core.megvii-inc.com/brain-sdk/MegBrain/-/blob/dev/lite/example/mge/README.md。


Python集成——安装
-----------------
Lite的python接口为python用户使用Lite进行模型推理提供了方便，whl包会随着megbrain的发版发布，版本号和megbrain保持一致，目前发布的Lite的whl包覆盖系统Linux、windows和macos，可以直接通过pip3安装。

.. note::

    目前支持支持的平台有：X86-CUDA，X86-CPU，X86-ATLAS。不支持X86-OpenCL，Arm-CPU，Arm-CUDA平台。

pip3的安装命令为：

.. code-block:: bash

    python3 -m pip install megenginelite -i  https://pypi.megvii-inc.com/simple


Python集成——推理
-----------------

Lite的python封装里主要有两个类：**LiteTensor** 和 **LiteNetwork** 。

LiteTensor
~~~~~~~~~~~~

LiteTensor提供了用户对数据的操作接口，提供了接口包括:

* **fill_zero**: 将tensor的内存设置为全0
* **share_memory_with**: 可以和其他LiteTensor的共享内存
* **copy_from**: 从其他LiteTensor中copy数据到自身内存中
* **reshape**: 改变该LiteTensor的shape，内存数据保持不变
* **slice**: 对该LiteTensor中的数据进行切片，需要分别指定每一维切片的start，end，和step。
* **set_data_by_share**: 调用之后使得该LiteTensor中的内存共享自输入的array的内存，输入的array必须是numpy的ndarray，并且tensor在CPU上
* **set_data_by_copy**: 该LiteTensor将会从输入的data中copy数据，data可以是list和numpy的ndarray，需要保证data的数据量不超过tensor的容量，tensor在CPU上
* **to_numpy**: 将该LiteTensor中数据copy到numpy的array中，返回给用户，如果是非连续的LiteTensor，如slice出来的，将copy到连续的numpy array中，该接口主要数为了debug，有性能问题。

对 **LiteTensor** 赋值，请参考：

.. code-block:: python

   import megenginelite as lite
   import numpy as np
   import os
    
   def test_tensor_set_data():
       layout = lite.LiteLayout([2, 16], "int8")
       tensor = lite.LiteTensor(layout)
       assert tensor.nbytes == 2 * 16
    
       data = [i for i in range(32)]
       tensor.set_data_by_copy(data)
       real_data = tensor.to_numpy()
       for i in range(32):
           assert real_data[i // 16][i % 16] == i
    
       arr = np.ones([2, 16], "int8")
       tensor.set_data_by_copy(arr)
       real_data = tensor.to_numpy()
       for i in range(32):
           assert real_data[i // 16][i % 16] == 1
    
       for i in range(32):
           arr[i // 16][i % 16] = i
       tensor.set_data_by_share(arr)
       real_data = tensor.to_numpy()
       for i in range(32):
           assert real_data[i // 16][i % 16] == i
    
       arr[0][8] = 100
       arr[1][3] = 20
       real_data = tensor.to_numpy()
       assert real_data[0][8] == 100
       assert real_data[1][3] == 20
    
   test_tensor_set_data()

让多个 **LiteTensor** 共享同一块内存数据，请参考：

.. code-block:: python

    import megenginelite as lite
    import numpy as np
    import os
     
    def test_tensor_share_memory_with():
        layout = lite.LiteLayout([4, 32], "int16")
        tensor = lite.LiteTensor(layout)
        assert tensor.nbytes == 4 * 32 * 2
     
        arr = np.ones([4, 32], "int16")
        for i in range(128):
            arr[i // 32][i % 32] = i
        tensor.set_data_by_share(arr)
        real_data = tensor.to_numpy()
        for i in range(128):
            assert real_data[i // 32][i % 32] == i
     
        tensor2 = lite.LiteTensor(layout)
        tensor2.share_memory_with(tensor)
        real_data = tensor.to_numpy()
        real_data2 = tensor2.to_numpy()
        for i in range(128):
            assert real_data[i // 32][i % 32] == i
            assert real_data2[i // 32][i % 32] == i
     
        arr[1][18] = 5
        arr[3][7] = 345
        real_data = tensor2.to_numpy()
        assert real_data[1][18] == 5
        assert real_data[3][7] == 345
     
    test_tensor_share_memory_with()

LiteNetwork
~~~~~~~~~~~~~

**LiteNetwork** 主要为用户提供模型载入，运行等功能。

以CPU为后端的模型载入、运行，请参考：

.. code-block:: python

    from megenginelite import *
    import numpy as np
    import os
     
    def test_network_basic():
        source_dir = os.getenv("LITE_TEST_RESOUCE")
        input_data_path = os.path.join(source_dir, "input_data.npy")
        # read input to input_data
        input_data = np.load(input_data_path)
        model_path = os.path.join(source_dir, "shufflenet.mge")
     
        network = LiteNetwork()
        network.load(model_path)
     
        input_tensor = network.get_io_tensor("data")
     
     
        # copy input data to input_tensor of the network
        input_tensor.set_data_by_copy(input_data)
     
        # forward the model
        for i in range(3):
            network.forward()
            network.wait()
     
        output_names = network.get_all_output_name()
        output_tensor = network.get_io_tensor(output_names[0])
     
        output_data = output_tensor.to_numpy()
        print('shufflenet output max={}, sum={}'.format(output_data.max(), output_data.sum()))
     
    test_network_basic()

以CUDA为后端，使用device内存作为模型输入，需要在构造network候配置config和IO信息。请参考：

.. code-block:: python

    from megenginelite import *
    import numpy as np
    import os
     
    def test_network_device_IO():
        source_dir = os.getenv("LITE_TEST_RESOUCE")
        input_data_path = os.path.join(source_dir, "input_data.npy")
        model_path = os.path.join(source_dir, "shufflenet.mge")
         
        # read input to input_data
        dev_input_data = LiteTensor(layout=input_layout, device_type=LiteDeviceType.LITE_CUDA)
        # fill dev_input_data with device memory
        #......
     
        # construct LiteOption
        net_config = LiteConfig(device_type=LiteDeviceType.LITE_CUDA, option=options)
     
        # constuct LiteIO, is_host=False means the input tensor will use device memory
        ios = LiteNetworkIO()
        # set the input tensor "data" memory is not in host, but in device
        ios.add_input(LiteIO("data", is_host=False))
     
        network = LiteNetwork(config=net_config, io=ios)
        network.load(model_path)
     
        dev_input_tensor = network.get_io_tensor("data")
     
        # set device input data to input_tensor of the network without copy
        dev_input_tensor.share_memory_with(dev_input_data)
        for i in range(3):
            network.forward()
            network.wait()
     
        output_names = network.get_all_output_name()
        output_tensor = network.get_io_tensor(output_names[0])
        output_data = output_tensor.to_numpy()
        print('shufflenet output max={}, sum={}'.format(output_data.max(), output_data.sum()))
     
    test_network_basic()
