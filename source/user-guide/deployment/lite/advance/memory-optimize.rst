.. _memory_optimize:

================================
内存优化
================================

模型 Inference 时候需要占用处理器上面的内存用于代码执行，数据缓存等，目前处理器上的内存（RAM）通常都比较大，但是在一些情况下内存依然是瓶颈，如：

* 在 NVIDIA 设备上进行多路视频解码和检测时候，会被 CUDA device 上的显存大小限制同时开启的路数。
* 在 `TEE <https://en.wikipedia.org/wiki/Trusted_execution_environment>`_ 环境中，内存资源非常受限。
* 以及在一些特定的嵌入式平台上，为了成本考虑一般内存有限。

MegEngine Lite 对 Inference 阶段的内存进行分类，主要包含下面3类：

* **存放模型权重的内存** ：用于加载模型中的权重（weights）数据，并在 Inference 阶段参与计算。
* **计算图结构内存** ：模型 Load 之后保存其结构信息所需的内存。
* **Runtime 内存** ：模型计算过程中用于保存 Operator 输入输出的 Tensor 内存，以及具体 Operator 计算时候需要的临时内存。

MegEngine Lite 中默认已经对静态图的内存做了极致的内存规划，如果在这个极致的内存规划上还需要继续进行内存优化，MegEngine lite 还有下面的方法进行内存优化：

* Record2 内存优化
* 共享运行时内存
* 多 Network 共享权重
* 用户注册自定义内存池

Record2 内存优化
------------------------

Record2 是在 :ref:`record_optimize` 基础上进一步优化内存，原理为：在 record1 的基础上，
会析构掉原来模型 load 时候创建的计算图，即上面提到的 **计算图结构内存** 。它和 record1 有相同的
限制。

.. warning::

    record 主要有如下3个 **限制条件** :

    * 执行的模型必须所有 shape 是静态可推导的，不能有动态图情况。
    * 输入 Tensor 的 shape 不能改变，改变 shape 之后，所有的内存计划都会改变，导致记录的 kerns 不可用。    
    * 模型中只能在一个设备上进行，目前 Record 功能只支持 CPU 上模型执行。

配置方法：将 :ref:`basic_code` 注释 1 的地方，将 Option 中的 comp_node_seq_record_level，设置为 2 即 record2，参考：:ref:`option_config`，
**另外开启 record2 还需要设置 Option 中 var_sanity_check_first_run 为 false， fake_next_exec 为 true**。

.. note::
    record2 内存节省的大小需要在直接模型和环境中测试。

共享运行时内存
-----------------------

多个 NetWork 不同时运行时候，可以共享他们的 Runtime 内存，这样可以有效减少多个模型的 Inference 的场景下内存的峰值。大概原理为，MegEngine
会为多个模型分配同一个 Runtime 内存，这个内存不是所有共享模型的 Runtime 内存的和，而是取他们的最大值。

配置方法：
调用 MegEngine Lite 的 Runtime API 接口，参考 :ref:`share_runtime_memory_with`。

.. code-block:: cpp

    Config config;
    std::shared_ptr<Network> network_src = std::make_shared<Network>(config);
    std::shared_ptr<Network> network_dst = std::make_shared<Network>(config);
    network_src->load_model(model_path_src);
    Runtime::share_runtime_memory_with(network_dst, network_src);
    network_dst->load_model(model_path_dst);

.. code-block:: python

    network_src = LiteNetwork()
    network_src.load(model_path_src)

    network_dst = LiteNetwork()
    network_dst.share_runtime_memroy(src_network)
    network_dst.load(model_path_dst)

上面是在 CPP 和 python 中使用共享运行时内存优化的代码片段，主要步骤为：

* 创建源 Network 之后，可以进行正常的加载模型等操作。
* 创建目的 Network。
* 调用 share_runtime_memroy 接口配置目的 Network 的 Runtime 内存将从源 Network 中共享。
* 目的 Network 进行模型加载等操作。

.. warning::

    * 目的 Network 设置共享 Runtime 内存之前不能加载模型，源 Network 没有任何限制。
    * 目的 Network 和源 Network 不能同时运行，否则出现计算错误，这需要 **用户自己保证**。


多 Network 共享权重
-----------------------------

当需要对同一个模型创建多个 Network 时候，如果每个 Network 都载入一次模型，这样势必会将模型的权重拷贝多次，导致存放模型权重的内存成倍的增大。
MegEngine Lite 支持这种情况下多个 Network 共享同一份模型的权重，这样将有效的减少内存的用量，典型的应用场景是，同一个模型希望在多个线程中
同时并行进行 Inference，这时就可以让这些不同线程里面的 Network 共享同一份权重数据，但是他们 Network 是不同的。

配置方法：
调用 MegEngine Lite 的 Runtime API 接口参考 :ref:`shared_weight_with_network`。

.. code-block:: cpp

    std::shared_ptr<Network> network_src = std::make_shared<Network>();
    network->load_model(model_path);
    std::shared_ptr<Network> network_dst = std::make_shared<Network>(config);
    Runtime::shared_weight_with_network(network_dst, network_src);

    network_src->forward();
    network_src->wait();
    network_dst->forward();
    network_dst->wait();

.. code-block:: python

    network_src = LiteNetwork()
    network_src.load(model_path_src)

    network_dst = LiteNetwork()
    network_dst.share_weights_with(src_network)

    network_src.forward()
    network_src.wait()

    network_dst.forward()
    network_dst.wait()

上面是在 CPP 和 python 中多个模型共享权重的代码片段，主要步骤为：

* 创建源 Network 之后，创建目的 Network。
* 源 Network 进行模型加载，推理等操作
* 调用 share_weights_with 接口将目的 Network 的权重内存从源 Network 中共享过去。
* 目的 Network 进行模型加载等操作。

.. note::

    * 目的 Network 不需要再载入模型
    * 目的 Network 和源 Network 可以同时运行

用户注册自定义内存池
-----------------------------

用户可以在 MegEngine Lite 中为模型运行时的内存分配注册回调函数，这样 Runtime 期间 MegEngine Lite 申请的内存都可以被外部回调函数接管。
某些场景下，用户可以在外部实现一个内存池，可以使得 MegEngine Lite 申请的内存和外部代码申请的内存进行复用，减少用量内存的峰值，目前该接口只能
在 C++ 接口中使用。

MegEngine Lite 中定义了 Allocator 的基类，用户可以继承这个基类，然后 override 它的 allocate 和 free 接口。

* **allocate 接口**：给定了需要分配内存所在的设备类型，设备的 ID，以及需要分配内存的长度，内存对齐的要求，需要以 void* 的形式返回申请好的内存。
* **free 接口**：传递进来需要释放内存的设备类型，设备 ID，以及需要释放的内存指针。

.. code-block:: cpp

    class Allocator {
    public:
        virtual ~Allocator() = default;

        //! allocate memory of size in the given device with the given align
        virtual void* allocate(
                LiteDeviceType device_type, int device_id, size_t size, size_t align) = 0;

        //! free the memory pointed by ptr in the given device
        virtual void free(LiteDeviceType device_type, int device_id, void* ptr) = 0;
    };

下面是使用 MegEngine Lite 进行推理，并使用用户自定义的内存分配器进行 Runtime 的内存分配的 example

.. code-block:: cpp

    class MyAllocator : public lite::Allocator {
    public:
        //! allocate memory of size in the given device with the given align
        void* allocate(LiteDeviceType, int, size_t size, size_t align) override {
            return memalign(align, size);
        };

        //! free the memory pointed by ptr in the given device
        void free(LiteDeviceType, int, void* ptr) override {
            free(ptr);
        };
    };

    auto allocator = std::make_shared<MyAllocator>();

    //! create and load the network
    std::shared_ptr<Network> network = std::make_shared<Network>();
    Runtime::set_memory_allocator(network, allocator);
    network->load_model(network_path);

    //! forward
    network->forward();
    network->wait();

为了完成自定义内存申请，用户需要：

* 自定义一个 MyAllocator 并继承自 lite::Allocator。
* 创建一个 std::shared_ptr<MyAllocator> 的智能指针。
* 创建完成 Network 之后，调用 Runtime 的 set_memory_allocator 将自定义的 MyAllocator 注册 MegEngine Lite。
* 正常运行模型。

.. warning::

    MegEngine Lite 中只有 Runtime 阶段使用的内存才可能使用到用户自定义的分配器。