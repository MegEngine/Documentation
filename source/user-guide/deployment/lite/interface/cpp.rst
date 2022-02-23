.. _cpp-interface:

================================
MegEngine Lite C++ 接口介绍
================================

Tensor 相关 API
---------------------

Layout
^^^^^^^^^^^
Layout的 C++ 接口实现如下

.. code-block:: cpp

    struct Layout {
        static constexpr uint32_t MAXDIM = 7;
        size_t shapes[MAXDIM];
        size_t ndim;
        LiteDataType data_type;
    };

主要成员为： layout 维度信息 ndim （最大维度为7），每一维度的具体信息 shapes，以及数据类型，MegEngine Lite 中包含的数据类型有：

.. code-block:: cpp

        typedef enum {
        LITE_FLOAT = 0,
        LITE_HALF = 1,
        LITE_INT = 2,
        LITE_INT16 = 3,
        LITE_INT8 = 4,
        LITE_UINT8 = 5,
        LITE_UINT = 6,
        LITE_UINT16 = 7,
        LITE_INT64 = 8,
    } LiteDataType;

Tensor 创建
^^^^^^^^^^^^^^^^
创建 Tensor 时候用户可以指定一些 Tensor 的信息，包括创建 Tensor 的设备类型，是否是设备的
`锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_，以及 Layout 信息等

.. code-block:: cpp

    Tensor(LiteDeviceType device_type, bool is_pinned_host = false);
    Tensor(LiteDeviceType device_type, const Layout& layout,
           bool is_pinned_host = false);
    Tensor(int device_id, LiteDeviceType device_type, const Layout& layout = {},
           bool is_pinned_host = false);

参数：

*  LiteDeviceType ：指定创建的 Tensor 所在的设备，默认是：LITE_CPU，目前主持的设备有：

    .. code-block:: cpp

        typedef enum {
            LITE_CPU = 0,
            LITE_CUDA = 1,
            LITE_ATLAS = 3,
            LITE_NPU = 4,
            LITE_DEVICE_DEFAULT = 5,
        } LiteDeviceType;

* device_id：指明 Tensor 创建的设备号
* is_pinned_host：表示该 Tensor 是否为 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_，默认为 false。

Tensor 信息获取
^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    LiteDeviceType get_device_type() const;
    int get_device_id() const;
    Layout get_layout() const;
    bool is_pinned_host() const;
    size_t get_tensor_total_size_in_byte() const;
    bool is_continue_memory() const;

* get_device_type：返回 Tensor 所在的设备类型
* get_device_id：返回 Tensor 所在的设备 id
* get_layout：返回 Tensor 的 layout 信息
* is_pinned_host：判断该 Tensor 的内存是否是 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_
* get_tensor_total_size_in_byte：获取这个 Tensor 总的内存大小，单位为字节
* is_continue_memory：获取这个 Tensor 的内存是否是连续的

get_memory_ptr
^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void* get_memory_ptr() const;
    void* get_memory_ptr(const std::vector<size_t>& idx) const;

* 无参数 get_memory_ptr：将以 void* 的形式返回 Tensor 的内存地址，如果 Tensor 没有申请内存，将会申请内存
* 有参数 get_memory_ptr：返回指定 index 的内存地址， **参数 const std::vector<size_t> 从 Tensor 高维到低维的 shape 索引**，其长度可以小于 Tensor 中 Layout 的维度，但是需要从高维度到低维度，中间不能有跳跃

示例：

.. code-block:: cpp

    Layout layout{{1, 3, 224, 224}, 4};
    Tensor tensor(LiteDeviceType::LITE_CPU, layout);
    // 获取 tensor 起始地址
    void* start_ptr = tensor.get_memory_ptr();
    // 获取 tensor 中 index 为（0，1，100，10）的地址
    void* start_ptr = tensor.get_memory_ptr({0, 1, 1000, 10});
    // 获取 tensor 中 index 为（0，1，100，0）的地址
    void* index_ptr = tensor.get_memory_ptr({0, 1, 1000});

reset
^^^^^^^^^
设置用户自己管理的内存地址到 Tensor 中

.. code-block:: cpp

    void reset(void* prepared_data, size_t data_length_in_byte);
    void reset(void* prepared_data, const Layout& layout);

参数：

* prepared_data：用户自己管理的内存， **用户需要确保 prepared_data 生命周期大于 Tensor 持有这段 prepared_data 内存的生命周期**，Tensor 中不会对这段内存进行管理
* data_length_in_byte：这段 prepared_data 内存的长度，单位是字节
* layout：这段 prepared_data 的 Layout 信息

reshape
^^^^^^^^^^

.. code-block:: cpp

    void reshape(const std::vector<int>& shape);

改变这个 Tensor 的 Layout 中的 shapes 为新的 shape，其中 **新的 shape 中元素个数需要和老的 shape 里面的元素个数相等**

slice
^^^^^^^^
.. code-block:: cpp

    std::shared_ptr<Tensor> slice(
        const std::vector<size_t>& start, const std::vector<size_t>& end,
        const std::vector<size_t>& step = {});

对 Tensor 进行切片，返回一个新的 Tensor，新的 Tensor 和原来 Tensor 共享内存， **新的 Tensor 可能不连续**

参数： **start，end 的长度必须相等，长度可以小于 Tensor 的 Layout 的维度，如果传递了 step，则 step 也需要和 start，end 的长度相等**

* start：Tensor 每一维度的起始 index 组成的数组，从高维到低维
* end：Tensor 每一维度的结束 index 组成的数组，从高维到低维
* step：Tensor 每一维度切片的间距，从高维到低维，默认为1

返回值：返回一个新的 Tensor，类型是一个 std::shared_ptr<Tensor>

示例：

.. code-block:: cpp

    Layout layout{{20, 20}, 2};
    Tensor tensor(LiteDeviceType::LITE_CPU, layout);
    // 对 Tensor 进行切片，返回 Tensor 为原来 Tensor 的 [0:1:20,0:1:10]
    auto slice0 = tensor.slice({0, 0}, {20, 10});
    // 对 Tensor 进行切片，返回 Tensor 为原来 Tensor 的 [0:1:20,10:1:10]
    auto slice1 = tensor.slice({0, 10}, {20, 20});
    // 对 Tensor 进行切片，返回 Tensor 为原来 Tensor 的 [0:2:20,:]
    auto slice1 = tensor.slice({0}, {20}, {2});

fill_zero
^^^^^^^^^^^^^

.. code-block:: cpp

   void fill_zero();

将 Tensor 内存里面的数据全部设置为 0

copy_from
^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void copy_from(const Tensor& src);

从 src Tensor 中 copy 数据到自己内存中， **如果 src 和自己的 layout 不相同时，会更改自身 Layout 信息为 src Layout**

share_memory_with
^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    // share memory with other tensor
    void share_memory_with(const Tensor& src_tensor);

将会和 src_tensor 共享内存数据， **如果 src_tensor 和自己的 Tensor 信息（layout，device_type，device_id等）不相同时，会更改自身信息为 src 的信息**

Network 相关 API
---------------------

.. _option_config:

创建 Network
^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    Network(const Config& config = {}, const NetworkIO& networkio = {});
    Network(const NetworkIO& networkio, const Config& config = {});

根据用户配置的 Config，以及用户配置的 NetworkIO 信息创建 Network

参数：

* config：可以不指定，不指定为默认值，Config 结构如下：

    .. code-block:: cpp

        struct Config {
            // 模型是否使用 lite 的方式压缩
            bool has_compression = false;
            // Network 的 device_id 和 device_type 信息
            int device_id = 0;
            LiteDeviceType device_type = LiteDeviceType::LITE_CPU;
            // MegEngine 默认为 LITE_DEFAULT
            LiteBackend backend = LiteBackend::LITE_DEFAULT;
            // 如果模型加密，模型加密算法名字
            std::string bare_model_cryption_name = {};
            // 优化选项
            Options options = {};
        };

    * bare_model_cryption_name：目前 MegEngine Lite 中写了三种加密算法，分别是："AES_default"，"RC4_default" 和 "SIMPLE_FAST_RC4_default"
    * options 定义了对 Network 进行优化的各种参数：

        .. code-block:: cpp

            struct Options {
                bool weight_preprocess = false;
                bool fuse_preprocess = false;
                bool fake_next_exec = false;
                bool var_sanity_check_first_run = true;
                bool const_shape = false;
                bool force_dynamic_alloc = false;
                bool force_output_dynamic_alloc = false;
                bool force_output_use_user_specified_memory = false;
                bool no_profiling_on_shape_change = false;
                uint8_t jit_level = 0;
                uint8_t comp_node_seq_record_level = 0;
                uint8_t graph_opt_level = 2;
                uint16_t async_exec_level = 1;
                //! layout transform options
                bool enable_nchw44 = false;
                bool enable_nchw44_dot = false;
                bool enable_nchw88 = false;
                bool enable_nhwcd4 = false;
                bool enable_nchw4 = false;
                bool enable_nchw32 = false;
                bool enable_nchw64 = false;
            };

    * weight_preprocess：在推理时候，部分 Kernel 执行前需要对权重进行转换，或者 Relayout，开启这个选项之后，将权重处理放到 Kernel 执行之前，优化 Kernel 执行时间，但是 Network 初始化时间变长
    * fuse_preprocess：开启该选项之后，模型中的部分前后处理 Operator 将会被融合在一起，优化模型执行的性能
    * fake_next_exec：下一次执行 Inference 时候，是否为假的执行：仅仅完成内存分配等和计算无关的操作。这次假的执行完成之后将被设置为 false
    * var_sanity_check_first_run：第一次执行 Inference 时候是否需要对每一个 Operator 的输入输出 Tensor 的正确性进行检查，默认为 true
    * const_shape：指定 Network 的输入 shape 不会变化，这样不用在后面的执行时检查是否需要重新分配内存等操作
    * force_dynamic_alloc：强制要求所有的 Tensor 都是运行时动态分配，且不进行内存优化，MegEngine 默认所有的 Tensor 都是执行前进行内存优化并静态申请
    * force_output_dynamic_alloc：强制最后输出的 Tensor 的内存为动态申请，这样输出 Tensor 不用 copy 到用户的内存中，可以直接代理到返回内存给用户
    * force_output_use_user_specified_memory：强制让输出 Tensor 的内存由用户指定，这样输出 Tensor 将不需要 copy 到用户内存，在最后一个 Kernel 计算时就写到了用户的内存地址中
    * no_profiling_on_shape_change：当 Network 的输入 Tensor 的 shape 改变的时候，这时候 fast-run 将不会进行重新搜索最优的 kernel 算法实现
    * jit_level：JIT 的级别，设置为 0 时：将关闭 JIT，设置为 1 时：仅仅只开启基本的 elemwise 的 JIT，当设置为 2 时：将开启 elemwise 和 reduce Operator 的 JIT
    * comp_node_seq_record_level：设置 MegEngine 的录制模式，当设置为 0 时：将不开启录制模式，设置为 1 时：将开启录制模式，不会析构这个计算图结构，当设置为 2 时：将开启录制模式，并释放掉整个计算图
    * graph_opt_level：设置图优化等级，当设置为 0 时：关闭图优化，当设置为 1 时：算术计算 inplace 优化，当设置为 2 时：在 1 的基础上在加上全局优化，当设置为 3 时：在 2 的基础上再使能 JIT
    * enable_xxxx：开启对应的 layout 转换优化，不同的平台上不同的 layout 性能差异较大，见下表：

+-------------------+----------------------------------------------------+-------------+
| 参数              | 作用                                               | 适用平台    |
+===================+====================================================+=============+
| enable-nchw88     | 将输入nchw layout的模型转为nchw88 layout的模型     | X86 avx256  |
+-------------------+----------------------------------------------------+-------------+
| enable-nchw44     | 将输入nchw layout的模型转为nchw44 layout的模型     | Arm float32 |
+-------------------+----------------------------------------------------+-------------+
| enable-nchw44-dot | 将输入nchw layout的模型转为nchw44-dot layout的模型 | Arm V8.2    |
+-------------------+----------------------------------------------------+-------------+
| enable-nchw4      | 将输入nchw layout的模型转为nchw4 layout的模型      | CUDA        |
+-------------------+----------------------------------------------------+-------------+
| enable-chwn4      | 将输入nchw layout的模型转为chwn4 layout的模型      | CUDA        |
+-------------------+----------------------------------------------------+-------------+
| enable-nchw32     | 将输入nchw layout的模型转为nchw32 layout的模型     | CUDA        |
+-------------------+----------------------------------------------------+-------------+
| enable-nhwcd4     | 将输入nchw layout的模型转为nhcw4 layout的模型      | 移动平台GPU |
+-------------------+----------------------------------------------------+-------------+


* networkio：配置 Network 的输入输出信息，主要配置输入 Tensor 的来源从 CPU 还是 device，输出 Tensor 保存在 CPU 端还是 device 端，默认输入输出都在 CPU 端

    .. code-block:: cpp

        struct IO {
            // 输入输出 Tensor 的名字
            std::string name;
            // 是否来自、输出到 device 端
            bool is_host = true;
            // 最后需要的是 Value 还是 Shape
            LiteIOType io_type = LiteIOType::LITE_IO_VALUE;
            // 该输入输出对应的 layout，不设置，Network 会使用模型
            Layout config_layout = {};
        };
        struct NetworkIO {
            // 所有的输入配置
            std::vector<IO> inputs = {};
            // 所有的输出配置
            std::vector<IO> outputs = {};
        };

示例：

.. code-block::

    std::string model_path = "./shufflenet.mge";
    std::string input_name = "data";

    NetworkIO IO;
    bool is_host = false;
    // 输入 tensor ”data“ 数据来自 device
    IO.inputs.push_back({input_name, is_host});

    Config config;
    //! 配置 config
    config.options.var_sanity_check_first_run = false;
    config.options.comp_node_seq_record_level = 1;
    //! 构造 Network
    std::shared_ptr<Network> network = std::make_shared<Network>(IO，config);

load_model
^^^^^^^^^^^^^^^

.. code-block:: cpp

    //! load the model form memory
    void load_model(void* model_mem, size_t size);
    //! load the model from a model path
    void load_model(std::string model_path);

Network 加载模型，可以从一个指定的路径 **model_path**，或者一段内存 **model_mem** 和其对应的 size 进行加载

获取 Network 基本信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // 获取模型的所有输入 Tensor 名字
    std::vector<std::string> get_all_input_name() const;
    // 获取模型的所有输出 Tensor 名字
    std::vector<std::string> get_all_output_name() const;
    // 获取模型的第 index 输入 Tensor 的名字
    std::string get_input_name(size_t index) const;
    // 获取模型的第 index 输入 Tensor 的名字
    std::string get_output_name(size_t index) const;
    // 通过名字获取输入或者输出 Tensor，如果输入输出名字有重复的情况，需要指定是输入还是输出：LiteTensorPhase
    std::shared_ptr<Tensor> get_io_tensor(
        std::string io_name, LiteTensorPhase phase = LiteTensorPhase::LITE_IO);
    // 获取模型的第 index 输入 Tensor
    std::shared_ptr<Tensor> get_input_tensor(size_t index);
    // 获取模型的第 index 输入 Tensor
    std::shared_ptr<Tensor> get_output_tensor(size_t index);
    // 获取 Network 的设备类型，device id，stream id
    LiteDeviceType get_device_type() const;
    int get_device_id() const;
    int get_stream_id() const;

设置 Network 基本信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // 设置 Network 运行的 device id 和 stream id
    Network& set_device_id(int device_id);
    Network& set_stream_id(int stream_id);
    // 设置模型异步执行时候的回调函数
    Network& set_async_callback(const AsyncCallback& async_callback);
    // 设置模型开始执行的回调函数
    Network& set_start_callback(const StartCallback& start_callback);
    // 设置模型完成执行的回调函数
    Network& set_finish_callback(const FinishCallback& finish_callback);

Network 执行
^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    void forward();
    void wait();

执行该 Network 的推理，并等待推理结束

compute_only_configured_output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void compute_only_configured_output();

配置模型只计算创建 Network 时候指定的 output tensor，其他 Tensor 不计算，不设置 Network 默认计算所有输出 Tensor 的值

get_static_memory_alloc_info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void get_static_memory_alloc_info(const std::string& log_dir = "logs/test") const;

获取 Network 运行该 模型时候的内存使用信息，该信息将以 json 文件形式保存在指定的 log_dir 中

enable_profile_performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    void enable_profile_performance(std::string profile_file_path);

测量 Network 运行该模型时候的每个 Op 的耗时信息，该信息将以 json 文件形式保存在指定的 profile_file_path 中

.. _get_model_extra_info:

get_model_extra_info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cpp

    const std::string& get_model_extra_info();

如果 MegEngine Lite 模型在打包模型时候设置了额外的 information，将通过这个接口获得，返回一段 json 字符串，用户自己解析，如果没有额外
 information 否则将返回空字符串

Network Runtime 配置
------------------------------------

模型的一部分配置在创建 Network 时候的 Config 中进行配置，另外 Runtime 相关配置封装在 Runtime 类型中，都是 Runtime 的静态函数

get/set_cpu_threads_number
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static size_t get_cpu_threads_number(std::shared_ptr<Network> dst_network);
    static void set_cpu_threads_number(
        std::shared_ptr<Network> dst_network, size_t nr_threads);

获取或者设置 dst_network 运行时候的线程数量，dst_network 必须是运行在 CPU 上面

set_runtime_thread_affinity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void set_runtime_thread_affinity(
            std::shared_ptr<Network> network,
            const ThreadAffinityCallback& thread_affinity_callback);

设置 dst_network 多线程运行时候，绑核的回调函数

set_cpu_inplace_mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static bool is_cpu_inplace_mode(std::shared_ptr<Network> dst_network);
    static void set_cpu_inplace_mode(std::shared_ptr<Network> dst_network);

获取或者设置 dst_network 运行在 CPU 的 **inplace** 模式，inplace 模式为：运行模型时候只有一个线程，这个线程发送 Kernel 任务的同时，inplace 地将
 kernel 执行计算任务。非 inplace 模式：将有2个线程，一个线程发送 Kernel 任务，一个线程执行 Kernel 任务。在一些单核处理器。
 或者低端 cpu 上，设置 **inplace 模式性能会好一些**。

use_tensorrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void use_tensorrt(std::shared_ptr<Network> dst_network);

设置 dst_network 使用 TensorRT 引擎进行推理

.. _set_network_algo_policy_cpp:

set_network_algo_policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void set_network_algo_policy(
            std::shared_ptr<Network> dst_network, LiteAlgoSelectStrategy strategy,
            uint32_t shared_batch_size = 0, bool binary_equal_between_batch = false);

设置 dst_network 模型运行时候选择算法的策略

参数：

* strategy： 选择算法的策略，MegEngine Lite 中支持以下策略：

    .. code-block:: cpp

        typedef enum {
            // 经验性的选择算法
            LITE_ALGO_HEURISTIC = 1 << 0,
            // 通过实际运行，选择最优的算法
            LITE_ALGO_PROFILE = 1 << 1,
            // 选择多次执行没有差别的算法
            LITE_ALGO_REPRODUCIBLE = 1 << 2,
            // 从具有优化的算法中选择算法
            LITE_ALGO_OPTIMIZED = 1 << 3,
        } LiteAlgoSelectStrategy;

    其中上面的策略在不冲突的情况下，可以进行组合

* binary_equal_between_batch： 多个 batch 同时进行计算时，如果输入完全一样，保证所有 batch 的计算结果完全一样
* shared_batch_size：binary_equal_between_batch 的时候，选择最优算法所依据的 batch 大小，设置 0 将使用模型默认的 batch size

set_network_algo_workspace_limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void set_network_algo_workspace_limit(
            std::shared_ptr<Network> dst_network, size_t workspace_limit);

设置 dst_network 运行选择算法时候，算法能够允许的最大 workspace，超过最大 workspace 的算法将不会选择

set_memory_allocator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void set_memory_allocator(
            std::shared_ptr<Network> dst_network,
            std::shared_ptr<Allocator> user_allocator);

设置 dst_network 运行时，使用用户自定义的内存分配器

.. _share_runtime_memory_with:

share_runtime_memory_with
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void share_runtime_memory_with(
            std::shared_ptr<Network> dst_network, std::shared_ptr<Network> src_network);

设置 dst_network 运行和 src_network 共享运行时候的内存， **这时 dst_network 和 src_network 不能同时执行**，
运行时内存指：除了保存模型 weights 和图结构以外的所有需要的运行时内存

.. _shared_weight_with_network:

shared_weight_with_network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void shared_weight_with_network(
            std::shared_ptr<Network> dst_network, std::shared_ptr<Network> src_network);

设置 dst_network 运行和 src_network 共享同一份权重，但是可以对不同的输入数据进行推理，这两个 Network 可以同时运行

enable_io_txt_dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void enable_io_txt_dump(
            std::shared_ptr<Network> dst_network, std::string io_txt_out_file);

将 dst_network 运行时候的所有 IO tensor 输出到文本文件 io_txt_out_file 中。

enable_io_bin_dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    static void enable_io_bin_dump(
            std::shared_ptr<Network> dst_network, std::string io_bin_out_dir);

将 dst_network 运行时候的所有 IO tensor 以二进制的形式保存在 io_bin_out_dir 文件夹中。

Global 配置
-----------------

register_decryption_and_key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    bool register_decryption_and_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

向 MegEngine Lite 中注册 decrypt_name 名字的解密算法，该解密算法的方法为 func，秘钥为 key

参数：

* decrypt_name： 新注册的解密算法的名字，字符串
* func：新注册的解密算法的方法，函数指针
* key：新注册的解密算法的秘钥，uint8_t 的数组

示例：

.. code-block:: cpp

    std::vector<uint8_t> decrypt_model(
        const void* model_mem, size_t size, const std::vector<uint8_t>& key) {
        if (key.size() == 1) {
            std::vector<uint8_t> ret(size, 0);
            const uint8_t* ptr = static_cast<const uint8_t*>(model_mem);
            uint8_t key_data = key[0];
            for (size_t i = 0; i < size; i++) {
                ret[i] = ptr[i] ^ key_data ^ key_data;
            }
            return ret;
        } else {
            printf("the user define decrypt method key length is wrong.\n");
            return {};
        }
    }
    // 注册 "just_for_test" 的加密算法，解密算法是 decrypt_model，秘钥是 15
    register_decryption_and_key("just_for_test", decrypt_model, {15});

.. _update_decryption_or_key:

update_decryption_or_key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    bool update_decryption_or_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

更新 MegEngine Lite 中注册的 decrypt_name 名字的解密算法，如果 func 不为空，则将之前的解密算法的方法更新为 func，
如果 key 的长度大于0，则将解密算法的秘钥更新为 key

示例：

.. code-block:: cpp

    std::vector<uint8_t> key(32, 0);
        for (size_t i = 0; i < 32; i++) {
            key[i] = 31 - i;
        }
    // 更新 "AES_default" 加密算法的秘钥为 key，解密 func 保持不变
    update_decryption_or_key("AES_default", nullptr, key);

.. _register_parse_info_func:

register_parse_info_func
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    bool register_parse_info_func(
        std::string info_type, const ParseInfoFunc& parse_func);

向 MegEngine Lite 中注册 info_type 名字的模型信息解析方法，该模型信息解析方法的执行函数为 parse_func

try_coalesce_all_free_memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void try_coalesce_all_free_memory();

配置 MegEngine Lite 将释放所有没有用到的内存，减少内存用量

set_loader_lib_path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void set_loader_lib_path(const std::string& loader_path);

设置使用 loader 对应的库文件路径为 loader_path

set_persistent_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    void set_persistent_cache(
        const std::string& cache_path, bool always_sync = false);

设置模型运行时候使用到的算法 cache，设置之后运行模型将直接从 cache 中获取对应算法，或者将选择的算法信息保存到该文件中

参数

* cache_path： 这个 fast-run cache 文件
* always_sync：是否这个 cache 文件时刻保持同步，如果是则：每次写 cache 都将写到文件中

dump_persistent_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

     void dump_persistent_cache(const std::string& cache_path);

将内存中的 fast-run cache 写到 cache_path 中

TensorRT cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

     void set_tensor_rt_cache(std::string tensorrt_cache_path);
     void dump_tensor_rt_cache();

设置或者保存 TensorRT 的 cache 文件


基本信息
^^^^^^^^^^^

.. code-block:: cpp

    // 获取 MegEngine Lite 的版本信息
    void get_version(int& major, int& minor, int& patch);
    // 设置 MegEngine Lite 的 log 级别
    void set_log_level(LiteLogLevel level);
    // 获取 MegEngine Lite 的 log 级别
    LiteLogLevel get_log_level();
    // 获取指定设备类别的设备数量
    size_t get_device_count(LiteDeviceType device_type);

物理地址和虚拟地址设置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

    // 全局设置 vir_ptr, phy_ptr 对到 MegEngine 对应的 device 和 backend 中
    bool register_memory_pair(
            void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
            LiteBackend backend = LiteBackend::LITE_DEFAULT);
    // 全局清除 MegEngine 对应的 device 和 backend 中的 vir_ptr, phy_ptr 对 
    bool clear_memory_pair(
            void* vir_ptr, void* phy_ptr, LiteDeviceType device,
            LiteBackend backend = LiteBackend::LITE_DEFAULT);
    // 通过虚拟地址查询对应 device 和 backend 中的物理地址，并返回
    void* lookup_physic_ptr(void* vir_ptr, LiteDeviceType device, LiteBackend backend);

部分设备上有虚拟地址和物理地址的概念，这里提供用户操作虚拟地址和物理地址的接口，主要有：
* 设置全局的物理地址和虚拟地址对
* 清除这些地址对
* 通过虚拟地址查询物理地址