.. _execution_optimize:

================================
执行性能优化
================================

MegEngine Lite 中有多个关于模型执行效率优化的选项，在追求极致性能时候可以使用他们的组合，将会获得一定的性能提升，性能提升
程度在不同的模型，不同的平台上表现不一样，可以根据实测来决定开启与否。另外有一些优化选项可能有一定的限制，需要在满足条件的情况下开启。
下面的优化在 MegEngine Lite 中非常简单，主要是在 Network 创建时候的 Config 中进行配置，或者在 Network 创建之后，模型 load 之前
通过 runtime 的接口进行配置。

.. _basic_code:

基本的配置代码
-------------------

.. code-block:: cpp

    //! config the network running 
    lite::Config config;

    //！1. add options here
    config.options.xxxxxx = ...;

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>(config);

    //!2. add additional runtime config here
    lite::Runtime::xxxxx(...);

    //! load the network
    network->load_model(network_path);

    std::shared_ptr<Tensor> input_tensor = network->get_input_tensor(0);
    Layout input_layout = input_tensor->get_layout();

    //! Now the device memory if filled with user input data, set it to the
    //! input tensor
    input_tensor->reset(user_ptr, input_layout);

    //! forward
    network->forward();
    network->wait();

    //! get the output data or read tensor set in network_in
    std::shared_ptr<Tensor> output_tensor = network->get_output_tensor(0);
    void* out_data = output_tensor->get_memory_ptr();
    size_t out_length = output_tensor->get_tensor_total_size_in_byte() /
                        output_tensor->get_layout().get_elem_size();

.. code-block:: python

    config = LiteConfig()
    # 1. config the network here
    config.options.weight_preprocess = 1
    config.options.xxxxx = ...

    network = LiteNetwork(config=config)
    network.load(self.model_path)

    # 2. add some runtime config here
    network.xxxxx(...)

    # set input and forward
    input_name = network.get_input_name(0)
    input_tensor = network.get_io_tensor(input_name)
    input_tensor.set_data_by_copy(user_data)

    network.forward()
    network.wait()

    # get the output data
    output_name = network.get_output_name(0)
    output_tensor = network.get_io_tensor(output_name)
    output_data = output_tensor.to_numpy()

上面是后面各种配置的代码上下文，模型配置的主要地方为上面注释 1 和注释 2 的地方，注释 1 的地方为通过 config 配置，注释 2 的地方为通过 Runtime API 进行配置

weight preprocess 优化
--------------------------------

大多数模型比较耗时的计算优化时都需要对权重（weights）进行处理，如使用 Im2col + Matmul 优化的 Convolution 需要对 weights 先进行 pack 重排数据，使用
winograd 的 Convolution 需要将 weights 的预处理放到模型执行前，这样可以优化模型执行的时间，具体能够优化多少跟模型相关。

配置方式：

* 将上面 :ref:`basic_code` 注释 1 的地方的 Option 中的 weight_preprocess 设置为 True 即可。

下面是在 Arm CPU 运行 resnet18，开启 weight_preprocess 前后的性能变化：

* 开启之前：多次执行平均时间 112 ms。
* 开启之后：多次执行平均时间 105 ms。

Fast-Run 优化
--------------------------------

目前在 MegEngine 中，针对某些 Operator，尤其是 Convolution，我们内部存在很多种不同的算法，如 direct, winograd, 或者 im2col 等，
这些算法在不同的 shape 或者不同的硬件平台上，其性能表现不同，导致很难写出一个比较有效的启发式搜索的算法，使其在执行的时候跑到最快的算法上。
为此，我们 MegEngine 集成了 Fast-Run 的功能，其是在执行模型的时候会将每个 Operator 的可选所有算法都执行一遍，
然后选择一个最优的算法记录下来，然后保留在 Fast-Run 的 algo cache 中。

编译支持： MegEngine Lite 在编译时默认关闭了 Fast-Run 功能，需要在编译时候设置 MGB_ENABLE_FASTRUN=1，如使用 MegEngine 的脚本编译时：

.. code-block:: bash

    EXTRA_CMAKE_ARGS="-DMGB_ENABLE_FASTRUN=1" ./scripts/cmake-build/host_build.sh

上面是编译 X86 本地时候的示例，用户可以根据需要选择编译脚本。

配置方法：将上面 :ref:`basic_code` 注释 2 的地方，模型 load 完成之后，运行之前，调用 Runtime 的接口进行配置，参考 C++ :ref:`set_network_algo_policy_cpp` 和 
python :ref:`set_network_algo_policy_python` 。

.. code-block:: cpp

    set_network_algo_policy(
            std::shared_ptr<Network> dst_network, LiteAlgoSelectStrategy strategy,
            uint32_t shared_batch_size = 0, bool binary_equal_between_batch = false);

.. code-block:: python

         def set_network_algo_policy(
        self, policy, shared_batch_size=0, binary_equal_between_batch=False
    ):

其中：LiteAlgoSelectStrategy 支持一下几种类型的组合：

* LITE_ALGO_HEURISTIC：不进行 Fast-Run 选择算法，使用经验性的 heuristic 方法进行算法选择。
* LITE_ALGO_PROFILE：使用 Fast-Run 来选择最优的算法，选择方法是对每一个算法进行 Profile，选择最优的。
* LITE_ALGO_REPRODUCIBLE：选择算法时只在可以每次计算结果可以稳定复现的算法中选择，有的算法相同的输入，每次计算结果不相同。
* LITE_ALGO_OPTIMIZED：在 Fast-Run 选择算法中，只从优化过的算法中进行选择，节省选择算法的时间。

.. note::

    上面 4 个 LiteAlgoSelectStrategy 在不冲突的情况下可以通过 | 操作组合在一起。

Format 优化
--------------------------------

由于不同的平台架构差异巨大，为了充分发挥目前设备的计算性能，需要将访存的效率做到最优，尽量不阻塞设备的计算单元，
让设备的计算单元尽量发挥出最大的效率。因为通常情况下 NCHW 的 Tensor 排布不能达到最优的访存性能，因此 MegEngine 
提供了多种 Tensor 的内存排布，就是上面提到的 Format，不同设备上最优的 Format 不一样，如：

* 在 Arm Float32 的情况下最优的 Format 为 MegEngine 内部定义的：NCHW44 Format。
* 在 x86 avx256 Float32 的情况下最优的 Format 为 MegEngine 内部定义的：NCHW88 Format。
* 在 NVDIA 的 GPU 设备上，最优的 Format 可能为 NCHW4，NCHW32，CHWN4 等。

配置方式：

* 将上面 :ref:`basic_code` 注释 1 的地方，将 Option 中的 enable_xxxx 设置为 True 即可，参考：:ref:`option_config`。

下面是在 Arm CPU 运行 resnet18，开启 enable-nchw44 前后的性能变化：

* 开启之前：多次执行平均时间 112 ms。
* 开启之后：多次执行平均时间 105 ms。

如此多的 Format，如何选择正确的 Format 非常困难，目前 MegEngine 正在支持自动选择最合适的 Format，即：全局图优化。

.. _record_optimize:

Record 优化：
--------------------------------

MegEngine 底层主要是静态图，而对于静态图来讲，它的执行序列是确定的，如下:

    data → conv1 → conv2 → elemwise → pooling → resize → ...

对于每个 Operator 来讲，都主要分为两个步骤：准备 kernel + 执行, 比如 conv 来讲：

* **准备 kernel**  ：根据 filter size、stride，shape 等信息决定需要跑的算法，如 imcol + matmul 或者 direct 等等，也就是选择一系列的 kerns (函数)，
    为了泛化，这些 kern 里面将 Operator 的输入输出的指针都包含进一个无参数的函数对象，后续只需要调用这个函数对象指针就可以执行这些 kerns。
* **执行** ：挨个执行这些无参数的函数对象就完成模型执行。

上面可以看到对于准备kernel这一步，如果输入 shape 不变，内存不变的情况下，这个完全可以省略掉。也就是我们可以在第一次执行的时候记录整个计算过程中会调用的 kerns，
然后之后的执行过程中只需要输入数据换掉。

目前 MegEngine 中支持两种 record，分别是 record1，和record2，record1 主要优化执行时间，record2 在 record1 的基础上进一步优化内存，其会在执行
阶段将整个 Graph 析构掉，释放多余的内存空间。

.. warning::

    record 主要有如下3个 **限制条件** :

    * 执行的模型必须所有 shape 是静态可推导的，不能有动态图情况。
    * 输入 Tensor 的 shape 不能改变，改变 shape 之后，所有的内存计划都会改变，导致记录的 kerns 不可用。    
    * 模型中只能在一个设备上进行，目前 Record 功能只支持 CPU 上模型执行。

配置方法：将上面 :ref:`basic_code` 注释 1 的地方，将 Option 中的 comp_node_seq_record_level 设置为 1 即 record1，设置为 2 即 record2，参考：:ref:`option_config`。
下面是在 Arm CPU 运行 resnet18，开启 record1 前后的性能变化：

* 开启之前：多次执行平均时间 112 ms。
* 开启之后：多次执行平均时间 109 ms。

多线程优化：
--------------------------------

目前 CPU 都有多个核心，多核心并行计算将大大提高模型推理速度，MegEngine 在 CPU 上支持多线程对模型进行推理加速。
配置方法：将上面代码注释 2 的地方，即：模型 load 之后，推理之前，调用 MegEngine Lite 的 Runtime API 进行配置：

.. code-block:: cpp

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_threads_number(network, 2);
    network->load_model(model_path);

.. code-block:: python

    network = LiteNetwork()
    network.threads_number = 2
    network.load(self.model_path)

下面是在 Arm CPU 运行 resnet18，开启 2 个线程前后的性能变化：

* 开启之前：多次执行平均时间 112 ms
* 开启之后：多次执行平均时间 65 ms

CPU Inplace 优化
--------------------------------

MegEngine 中 Operator 的执行模型是：一个 CPU 上的线程进行 Kern 的发送，Device（CUDA，NPU等）负责执行 Kern，
当在 CPU 上这时候就相当于：一个线程进行 kern 发送，一个线程进行 Kern 执行，模拟 Device 上执行的场景，这是默认的
执行方式，CPU Inplace 模式指：在 CPU 上推理时，为了避免两个线程之间的同步带来性能影响，将只用一个线程同时完成 Kern
发送和 Kern 执行。该模式在一些 **低端 CPU，或者核心少的 CPU 上有性能优势**。

配置方法：在上面 :ref:`basic_code` 注释 2 的地方，即：模型 load 之后，推理之前，调用 MegEngine Lite 的 Runtime API 进行配置：

.. code-block:: cpp

    std::shared_ptr<Network> network = std::make_shared<Network>(config);
    Runtime::set_cpu_inplace_mode(network);
    network->load_model(model_path);

.. code-block:: python

    network = LiteNetwork()
    network.enable_cpu_inplace_mode()
    network.load(self.model_path)

下面是在 Arm CPU 运行 resnet18，开启 CPU Inplace 模式前后的性能变化：

* 开启之前：多次执行平均时间 112 ms。
* 开启之后：多次执行平均时间 110 ms。

.. note::

    上面的测试结果不在 **低端或者单核CPU上执行**，不具有代表性，用户需要根据实际情况测试。

JIT 优化
--------------------------------

MegEngine 提供了细粒度的 Tensor 计算原语(Tensor Add/Reduce/Concat/TypeCvt等)，赋予了研究员极大的灵活性，能够让研究员们更好的搭建各种各样的神经网络。
然而细粒度的算子会引入更多的存储和计算芯片间的数据传输，降低了深度学习网络的训练和推理性能。为了解决这一问题，我们把访存密集型的
细粒度算子(Elemwise/TypeCvt/Reduce等)构成的连通区域融合成一个子图，而子图对应了一个粒度更大的算子。在计算时，大粒度的算子只需要将输入节点
的数据搬运到计算芯片上，消除了子图中间节点引入的额外数据传输量，这样减少了计算过程中对存储带宽的压力，从而提升了网络在训练和推理时的性能。

不同于提前(Ahead Of Time)编译，JIT 编译是一种动态编译/运行时编译的技术，它允许自适应优化，可以针对特定的微架构进行加速，
在理论上，可以获得比静态编译更好的执行速度。JIT编译相比静态编译的优点如下：

* JIT 编译可以针对目标 CPU/GPU 进行特定优化。
* JIT 编译可以减少静态编译出来的二进制文件的大小。
* JIT 编译可以利用运行时的信息(如：TensorShape )进行特定优化。

MegEngine 采用JIT编译技术将融合后的大粒度算子翻译成特定平台上的计算 kernel。MegEngine 支持 JIT 编译的平台主要是 Nvidia 的 CUDA 平台，
JIT 的级别，为 0 时：将关闭 JIT，为 1 时：仅仅只开启基本的 elemwise 的 JIT，为 2 时：将开启 elemwise 和 reduce Operator 的 JIT，

配置方法：在上面 :ref:`basic_code` 注释 1 的地方，将 Option 中的 jit_level 设置为对应等级即可。

总结
--------------------------------

上面分别通过不同的方法对模型进行加速，但是这些方法是可以组合的，组合之后将会获得最优的执行时间，下面是添加
weight preprocess，fast-run，recored1，enable-nchw44 在 Arm 上运行 Resnet18 float32 模型优化前后的性能对比：

* 优化之前：多次执行平均时间 112 ms。
* 优化之后：多次执行平均时间 51 ms。

因为上面的模型在 Fast-Run 模式下选择到了 winograd 算法，因此性能提升较大。不同的组合可能性能不一样，用户可以根据自己
模型特点进行测试，后续 MegEngine Lite 将在 :ref:`load-and-run` 中开发一个 fitting 模型，自动的选择最优的参数组合。