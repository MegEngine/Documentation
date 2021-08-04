.. _cpp-basic:

=================================
MegEngine Lite C++ 基础功能介绍
=================================


在CPU上做推理的一个例子
-----------------------
接下来，我们将逐步讲解一个使用MegEngine Lite在CPU上做推理的简单例子，即：:src:`basic.cpp <lite/example/mge/basic.cpp>` 中的 ``basic_load_from_path()`` 函数。 


1. 一些配置
~~~~~~~~~~~~~

首先需要配置Log的级别，**LiteLogLevel** 一共有 **DEBUG**、**INFO**、**WARN** 和 **ERROR** 四个级别，其中 **DEBUG** 级别会把最细致的信息作为log显示在屏幕。此外，模型文件和输入数据的路径也要设定。对应代码如下：

.. code-block:: cpp

	set_log_level(LiteLogLevel::DEBUG);
	std::string network_path = args.model_path;
	std::string input_path = args.input_path;


2. 加载模型
~~~~~~~~~~~~~

模型文件将被作为参数，传给 **Network** 类的 ``load_model`` 方法。**Network** 类包含了加载、初始化、推理模型和显示模型信息的功能。详情请参考 :src:`network.h <lite/include/lite/network.h>`

.. code-block:: cpp

	std::shared_ptr<Network> network = std::make_shared<Network>();
	network->load_model(network_path);


3. 加载输入数据
~~~~~~~~~~~~~~~~

加载前，先要构建 **Tensor** 实体，需要注意与之相关的 **LiteBackend** 、 **LiteDeviceType** Enum和 **Layout** 类。 **Layout** 类的实体是 **Tensor** 类的成员之一，用于描述 **Tensor** 的维度和数据类型，而 **LiteBackend** 、 **LiteDeviceType** 的实现如下：

.. code-block:: cpp

	typedef enum LiteBackend {
	    LITE_DEFAULT = 0, //! default backend is mge
	    LITE_RK_NPU = 1, //! for rk npu backend
	} LiteBackend;

.. code-block:: cpp

	typedef enum LiteDeviceType {
	    LITE_CPU = 0,
	    LITE_CUDA = 1,
	    LITE_OPENCL = 2,
	    LITE_ATLAS = 3,
	    LITE_NPU = 4,
	    //! when the device information is set in model, so set LITE_DEVICE_DEFAULT
	    //! in lite
	    LITE_DEVICE_DEFAULT = 5,
	} LiteDeviceType;

本例中，输入数据是从某npy文件加载进来的，具体的加载方式详见 :src:`main.cpp <lite/example/main.cpp>` 中关于 ``parse_npy()`` 函数的实现。本例通过 ``parse_npy()`` 加载数据并构建src_tensor。然后把src_tensor的数据拷贝到network的输入tensor中。

.. code-block:: cpp

	std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);

    auto layout = input_tensor->get_layout();
    for (size_t i = 0; i < layout.ndim; i++) {
        printf("model input shape[%zu]=%zu \n", i, layout.shapes[i]);
    }

    //! copy or forward data to network
    size_t length = input_tensor->get_tensor_total_size_in_byte();
    void* dst_ptr = input_tensor->get_memory_ptr();
    auto src_tensor = parse_npy(input_path);
    auto layout0 = src_tensor->get_layout();
    for (size_t i = 0; i < layout0.ndim; i++) {
        printf("src shape[%zu]=%zu \n", i, layout0.shapes[i]);
    }
    void* src = src_tensor->get_memory_ptr();
    memcpy(dst_ptr, src, length);



4. 推理
~~~~~~~~~~~~

网络的推理是通过调用 **Network** 的 ``forward()`` 方法和 ``wait()`` 方法完成的。如果想记录运行时间，可以使用 **lite\:\:Timer** 。本例中相关代码如下：

.. code-block:: cpp

	lite::Timer ltimer("warmup");
	network->forward();
	network->wait();
	ltimer.print_used_time(0);

	lite::Timer ltimer("forward_iter");
	for (int i = 0; i < 10; i++) {
		ltimer.reset_start();
		network->forward();
		network->wait();
		ltimer.print_used_time(i);
	}



5. 获取输出数据
~~~~~~~~~~~~~~~~~

推理完成后，网络的输出数据可通过 **Network** 的 ``get_output_tensor()`` 函数获取。具体用法可参看 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中的 ``output_info()`` 函数代码。



对于在N卡设备上的推理
----------------------

如果用N卡设备做推理，需要在上面例子的基础上稍作修改：把输入Tensor需要构造为 **LiteDeviceType\:\:LITE_CUDA** 类型。即 ``load_from_path_run_cuda()`` 函数中的如下部分（完整代码在 :src:`basic.cpp <lite/example/mge/basic.cpp>` 里）：

.. code-block:: cpp

    auto tensor_device = Tensor(LiteDeviceType::LITE_CUDA, input_layout);

    tensor_device.copy_from(*src_tensor);

    input_tensor->reset(tensor_device.get_memory_ptr(), input_layout);



对于支持OpenCL的后端设备上的推理
---------------------------------

在以OpenCL为后端的设备上，有两种加载并推理模型的方式：

- 首次推理的同时搜索最优算法并将搜索结果存为文件（ ``load_from_path_use_opencl_tuning()`` 函数）

- 以算法搜索结果文件中的算法推理模型（ ``load_from_path_run_opencl_cache_and_policy()`` 函数）。

前者首次推理的速度较慢，可以看做是为后者做的准备。后者的运行效率才是更贴近工程应用水平的。两者的详细实现都在文件 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中。



用异步执行模式进行推理
------------------------

实现在 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中的 ``async_forward()`` 函数 。用户通过接口注册异步回调函数将设置 Network 的 Forward 模式为异步执行模式，目前异步执行模式只有在 CPU 和 CUDA 10.0 以上才支持，在inference时异步模式，主线程可以在工作线程正在执行计算的同时做一些其他的运算，避免长时间等待，但是在一些单核处理器上没有收益。





