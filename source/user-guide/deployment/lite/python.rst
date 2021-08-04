.. _megengine-lite-python:

=====================
MegEngine Lite Python
=====================

Lite 的 Python 封装将随着 Lite 一起开源。

Python 推理接口
---------------

Lite 的 python 封装里主要有两个类：``LiteTensor`` 和 ``LiteNetwork``.

LiteTensor
~~~~~~~~~~

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
~~~~~~~~~~~

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
