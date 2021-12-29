.. _memory_copy_optimize:

================================
输入输出内存拷贝优化
================================

MegEngine Lite 中的内存拷贝优化主要指输入输出 Tensor 内存拷贝的优化，模型内部固有的内存拷贝优化不能够被优化，主要有下面几种情况：

* device IO 优化：输入数据本来就不是 CPU 端的内存，如：是一段 CUDA 的内存或者一段 OpenCL 的内存，希望模型推理直接使用这段内存作为输入，避免将其拷贝到 CPU 端，然后再在模型内部从 CPU 拷贝到设备上，节省两次内存拷贝。

* 输入输出零拷贝：希望模型的推理结果保存在用户提供的内存中，避免将数据保存在 MegEngine 自己申请的内存中，然后再将内存拷贝到用户指定的内存中。

.. _device-io-optimize:

Device IO 优化
-----------------------

MegEngine Lite 支持模型的输入输出配置，用户可以根据实际情况灵活配置。主要方式是在创建 Network 时候配置其 IO 属性，
下面的 example 是指定模型名字为 "data" 的 Tensor 的内存为 CUDA 设备上内存，输出名字为 "TRUE_DIV" 的 Tensor 数据
保存在 CUDA 设备上。

.. code-block:: cpp

    std::string network_path = args.model_path;
    std::string input_path = args.input_path;
    //! config the network running in CUDA device
    lite::Config config{LiteDeviceType::LITE_CUDA};
    //! set NetworkIO include input and output
    NetworkIO network_io;
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV";
    bool is_host = false;
    IO device_input{input_name, is_host};
    IO device_output{output_name, is_host};
    network_io.inputs.push_back(device_input);
    network_io.outputs.push_back(device_output);

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config, network_io);
    network->load_model(network_path);

    std::shared_ptr<Tensor> input_tensor_device = network->get_input_tensor(0);
    Layout input_layout = input_tensor_device->get_layout();

    //! malloc the device memory
    auto tensor_device = Tensor(LiteDeviceType::LITE_CUDA, input_layout);

    //! copy to the device memory
    input_tensor_device->copy_from(tensor_device);

    //! forward
    network->forward();
    network->wait();

    //! output_tensor_device is in device
    std::shared_ptr<Tensor> output_tensor_device = network->get_io_tensor(output_name);

.. code-block:: python

    from megenginelite import *
    import numpy as np
    import os


    model_path = ... 
    # construct LiteOption
    net_config = LiteConfig(device_type=LiteDeviceType.LITE_CUDA)
 
    # set the input tensor "data" memory is not in host, but in device
    io_input = LiteIO("data", is_host=False)
    # set the output tensor "TRUE_DIV" memory is in device
    io_output = LiteIO("TRUE_DIV", is_host=False)
    # constuct LiteIO with LiteIO
    ios = LiteNetworkIO(inputs=[io_input], outputs=[io_output])
 
    network = LiteNetwork(config=net_config, io=ios)
    network.load(model_path)

    dev_input_tensor = network.get_io_tensor("data") 
    # read input to input_data
    dev_input_data = LiteTensor(layout=dev_input_tensor.layout, device_type=LiteDeviceType.LITE_CUDA)
    # fill dev_input_data with device memory
    #......

    # set device input data to input_tensor of the network without copy
    dev_input_tensor.share_memory_with(dev_input_data)

    # inference
    network.forward()
    network.wait()
 
    output_tensor = network.get_io_tensor("TRUE_DIV")
    output_data = output_tensor.to_numpy()
    print('output max={}, sum={}'.format(output_data.max(), output_data.sum()))

上面分别是 C++ 和 Python 使用 MegEngine Lite 配置 IO 为 device 上输入输出的示例，C++ 主要的配置为：

.. code-block:: cpp

    NetworkIO network_io;
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV";
    bool is_host = false;
    IO device_input{input_name, is_host};
    IO device_output{output_name, is_host};
    network_io.inputs.push_back(device_input);
    network_io.outputs.push_back(device_output);
    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config, network_io);

.. code-block:: python

    # constuct LiteIO, is_host=False means the input tensor will use device memory
    ios = LiteNetworkIO()
    # set the input tensor "data" memory is not in host, but in device
    ios.add_input(LiteIO("data", is_host=False))
    # set the output tensor "TRUE_DIV" memory is in device
    ios.add_output(LiteIO("TRUE_DIV", is_host=False))
    network = LiteNetwork(config=net_config, io=ios)

Network 的 IO 中 input 名字为 "data" 和 output 名字为 "TRUE_DIV" 的 IO 的 is_host 属性为 false，host 默认指 CPU 端，
为 flase 则表述输入或者输出的内存为设备端。

输入输出拷贝优化
-----------------------

.. _device-io-memcopy-optimize:

Device 上输入输出优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device 上进行模型推理除了 :ref:`device-io-optimize` 的情况外，都需要将输入从 CPU 拷贝到 Device 上，然后执行模型推理，执行完成之后，将
输出数据拷贝到 CPU 上，这是在 Device 上执行推理不可缺少的情况，(除了 :ref:`device-io-optimize` )。但是我们可以优化输入从真实数据拷贝
到模型的 CPU 输入数据和输出从 CPU 再拷贝到用户指定的内存中这些内存拷贝操作。

.. code-block:: cpp

    Config config;
    std::string model_path = ...;
    std::string input_name = "data";
    std::string output_name = "TRUE_DIV";

    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    network->load_model(model_path);
    std::shared_ptr<Tensor> input_tensor = network->get_io_tensor(input_name);

    auto src_ptr = malloc(input_tensor->get_tensor_total_size_in_byte());
    auto src_layout = input_tensor->get_layout();
    input_tensor->reset(src_ptr, src_layout);

    std::shared_ptr<Tensor> output_tensor = network->get_io_tensor(output_name);

    void* out_data = malloc(output_tensor->get_tensor_total_size_in_byte());
    output_tensor->reset(out_data, output_tensor->get_layout());

    network->forward();
    network->wait();

    delete src_ptr;
    delete out_data;

.. code-block:: python

    from megenginelite import *
    import numpy as np
    import os

    model_path = "./shufflenet.mge"
    # construct LiteOption
    net_config = LiteConfig()

    network = LiteNetwork(config=net_config)
    network.load(model_path)

    input_tensor = network.get_io_tensor("data")
    # read input to input_data
    input_data = LiteTensor(layout=input_tensor.layout)
    # fill input_data with device data

    # set device input data to input_tensor of the network without copy
    input_tensor.share_memory_with(input_data)

    output_tensor = network.get_io_tensor(network.get_output_name(0))
    out_array = np.zeros(output_tensor.layout.shapes, output_tensor.layout.dtype)

    output_tensor.set_data_by_share(out_array)

    # inference
    network.forward()
    network.wait()

    print('output max={}, sum={}'.format(out_array.max(), out_array.sum()))

该优化主要是使用 LiteTensor 的 reset 或者 memory share 的接口，将用户的内存共享到 Network 中的输入输出 LiteTensor 中。

CPU 上输入输出零拷贝
^^^^^^^^^^^^^^^^^^^^^^^^^^^

输入输出零拷贝，指用户的输入数据可以不用拷贝到 MegEngine Lite 中，模型推理完成的输出数据可以直接写到用户指定的内存中，
减少将输出数据拷贝到用户的内存中的过程，用户的内存 MegEngine Lite 不会进行管理，用户需要确保 **内存的生命周期大于模型推理的生命周期**。
  
实现这个功能主要将上面 :ref:`device-io-memcopy-optimize` 优化中配置 network 时，使能 force_output_use_user_specified_memory 选项：

* 设置 force_output_use_user_specified_memory 为 True。
* 模型运行之前通过 LiteTensor 的 reset 接口设置设置自己管理的内存到输入输出 Tensor 中，在 python 中可以调用 set_data_by_share 达到相同的功能。

.. warning::

    * 使用 force_output_use_user_specified_memory 这个参数时，只能获取模型计算的输出 Tensor 的结果，获取中间 Tensor 的计算结果是不被允许的。
    * 模型必须是静态模型，输出 LiteTensor 的 layout 需要在模型载入之后就能够被推导出来。
    * force_output_use_user_specified_memory 参数目前只在 CPU 使用，其他 Device 上不能使用。