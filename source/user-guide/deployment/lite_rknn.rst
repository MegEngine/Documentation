.. _lite_rknn:

=====================
Lite对第三方硬件（RKNN）的支持
=====================

Lite的设计为方便地支持第三方硬件做了考量。目前，我们以rknn设备（rk3568）为范例做了支持。

编译
------------------

目前lite有两个后端：**mge** 和 **rknn**。可以同时编译两个backend，也可以只编译一个backend，因为需要依赖rknn的第三方动态库，所以目前编译出来的都是动态库。目前只支持编译android版本，包括android arm64和android armv7。

lite支持的编译方式仍然是两种: **bazel** 和 **cmake**，具体方式请参考:ref:`lite`。



使用
---------------------

下面主要介绍工程集成rknn时候使用lite的接口，基本推理调用流程为：

1. 构造network并load数据
2. 获取模型的input tensor
3. copy数据到输入tensor中
4. Inference
5. 读取输出数据

.. code-block:: cpp

    bool lite::example::input_pass_through(void* input_data, size_t length) {
        //1. create and load the network
        Config config;
        config.backend = LiteBackend::LITE_RK_NPU;
        config.device_type = LiteDeviceType::LITE_NPU;
        std::shared_ptr<Network> network = std::make_shared<Network>(config);
        network->load_model(network_path);
     
        //2. set input data to input tensor, the network default is pass through
        auto input_name = network->get_all_input_name();
        std::shared_ptr<Tensor> input_tensor =
                network->get_io_tensor(input_name[0].c_str());
     
        //3. copy or forward data to network input or reset data to it
        size_t length = input_tensor->get_tensor_total_size_in_byte();
        void* dst_ptr = input_tensor->get_memory_ptr();
         
        // copy input data to input tensor
        memcpy(dst_ptr, input_data, length);
     
        //4. forward
        network->forward();
        network->wait();
     
        //5. get the output data or read tensor set in network_in
        auto output_name = network->get_all_output_name();
        std::shared_ptr<Tensor> output_tensor =
                network->get_io_tensor(output_name[0].c_str());
     
        void* out_data = output_tensor->get_memory_ptr();
    }

rknn中如果输入tensor的format和data type和模型input的数据不一样的时候，可以对输入数据进行配置，rknn内部自己转换为需要的数据格式。主要在第4步(forward)之前，将目前输入数据的format和data type设置tensor中，通过接口是 **TensorUtils::set_tensor_information**

.. code-block:: cpp

    //! set input data to input tensor, the network default is pass through
    auto input_name = network->get_all_input_name();
    std::shared_ptr<Tensor> input_tensor =
         network->get_io_tensor(input_name[0].c_str());
     
    //! config the input tensor format and dtype
    std::unordered_map<std::string, LiteAny> input_info;
    //! according to the rknn api header, UINT8 is 3
    input_info["type"] = LiteAny(static_cast<int>(3));
    //! according to the rknn api header, NHWC is 1
    input_info["fmt"] = LiteAny(static_cast<int>(1));
    TensorUtils::set_tensor_information(input_tensor, input_info);

其中input info中可以设置：

* key:"pass_through",  类型: uint8_t,  输入数据是透传到network中
* key:"fmt",  类型 (rknn_tensor_format)int,  非透传模式中，指明输入tensor的format
* key:"type",  类型 (rknn_tensor_type)int,  非透传模式下，指明输出tensor的format

输出内存预先申请主要是将输出内存提前申请，可以为用户外部申请的内存，这样可以避免输出数据的copy，lite中使用，依然是在第4步之前进行配置。

.. code-block:: cpp

    //! get the output data or read tensor set in network_in
    auto output_name = network->get_all_output_name();
    std::shared_ptr<Tensor> output_tensor =
            network->get_io_tensor(output_name[0].c_str());

    //! reset the output tensor memory with preallocated memory
    auto pre_alloc_ptr = std::shared_ptr<void>(malloc(2 * 7 * 7),
                                               [](void* ptr) { free(ptr); });
    output_tensor->reset(pre_alloc_ptr.get(), 2 * 7 * 7);

    //! or just set the is_prealloc info to the output tensor
    /*std::unordered_map<std::string, LiteAny> output_info;
    output_info["is_prealloc"] = LiteAny(static_cast<uint8_t>(true));
    TensorUtils::set_tensor_information(output_tensor, output_info);*/

设置内存预先申请有两途径：

* 通过对output tensor执行reset操作，调用output tensor的reset接口
* 通过设置output的information，"is_prealloc"设置为true。

rknn模型默认是量化的模型，正常情况下输出是int8，如果用户需要最终数据是float，可以通过在第4步(forward)之前，设置output tensor的数据类型为float，方法如下：

.. code-block:: cpp

    //! get the output data or read tensor set in network_in
    auto output_name = network->get_all_output_name();
    std::shared_ptr<Tensor> output_tensor =
            network->get_io_tensor(output_name[0].c_str());

    //! set layout dtype is float, then it will convert the network output
    //! to float
    auto output_layout = output_tensor->get_layout();
    output_layout.data_type = LiteDataType::LITE_FLOAT;
    output_tensor->set_layout(output_layout);
    //! or just set the want_float info to the output tensor
    /*std::unordered_map<std::string, LiteAny> output_info;
    output_info["want_float"] = LiteAny(static_cast<uint8_t>(true));
    TensorUtils::set_tensor_information(output_tensor, output_info);*/


设置输出为float有两途径：

* 通过对output tensor执行set_layout接口，不改变shape的情况下，将data type设置为LiteDataType::LITE_FLOAT
* 通过设置output的information，"want_float" 这只为 true。

output tensor可以设置的info有：

* key:"want_float", 类型：uint8_t, 是否输出tensor转换为float
* key:"is_prealloc", 类型：uint8_t, 输出是否提前申请好

