.. _cpp-advanced:

================================
MegEngine Lite C++ 进阶功能介绍
================================

在 example 目录中，除了 :src:`basic.cpp <lite/example/mge/basic.cpp>` 介绍了一些基本用法之外，还有其他一些例子演示了用 lite 的接口做推理相关更进阶的功能。比如：


CPU上的模型加载和推理
---------------------

主要介绍了用 lite 来完成基本的 inference 功能，load 模型使用默认的配置，进行 forward 之前将输入数据 copy 到输入 tensor 中，完成 forward 之后，再将数据从输出 tensor 中 copy 到用户的内存中，输入 tensor 和输出 tensor 都是从Network 中通过 name 来获取的，输入输出 tensor 的 layout 也可以从对应的 tensor中直接获取获取，

.. warning::
    
    输出 tensor 的 layout 必须在 forward 完成之后获取才是正确的。

着重强调两种加载模型的方式： *通过模型文件加载模型* （ ``basic_load_from_path()`` ）和 *通过内存加载模型* （ ``basic_load_from_memory()`` ），请着重对比两个函数调用 ``network->load_model()`` 时参数的不同。详细实现在文件 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中。


OpenCL后端设备上的模型加载和推理
---------------------------------

在以OpenCL为后端的设备上，有两种加载并推理模型的方式： *首次推理的同时搜索最优算法并将搜索结果存为文件* （ ``load_from_path_use_opencl_tuning()`` ）和 *以算法搜索结果文件中的算法推理模型* （ ``load_from_path_run_opencl_cache_and_policy()`` ）。前者首次推理的速度较慢，可以看做是为后者做的准备。后者的运行效率才是更贴近工程应用水平的。详细实现在文件 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中。


指定输入输出的内存
-------------------

在以CPU为后端的设备上
~~~~~~~~~~~~~~~~~~~~~~~

- 实现在 :src:`reset_io.cpp <lite/example/mge/reset_io.cpp>` 中，包括 ``reset_input()`` 和 ``reset_input_output()`` 两个函数。

- 两个函数演示了输入 tensor 的内存为用户指定的内存（该内存中已经保存好输入数据），输出 tensor 也可以是用户指定的内存，这样 Network 完成 Forward 之后就会将数据保存在指定的输出内存中。如此减少不必要的 memory copy 的操作。

- tensor 中的 reset 接口可以重新指定 tensor 的内存和对应的layout，如果 layout 没有指定，默认为 tensor 中原来的 layout。

.. warning::

    该方法中由于内存是用户申请，需要用户提前知道输入，输出 tensor 对应的 layout，然后根据 layout 来申请内存。另外，通过 reset 设置到 tensor 中的内存的生命周期不由 tensor管理，而是由外部用户来管理。


在N卡设备上
~~~~~~~~~~~~~~

- **指定输入输出的显存**

实现在 :src:`device_io.cpp <lite/example/mge/device_io.cpp>` 中的 ``device_input()`` 和 ``device_input_output()`` 两个函数。示例中，模型运行在 device(CUDA) 上，并且使用用户提前申请的 device 上的内存作为模型运行的输入和输出。这需要在 Network 构建的时候指定输入输出的在 device 上（如不设置，则默认在 CPU 上），其他地方和**输入输出为用户指定的内存**的使用相同。可以通过 tensor 的 ``is_host()`` 接口来判断该 tensor 在 device 端还是 host 端。


- **申请 pinned host 内存作为输入**

实现在 :src:`device_io.cpp <lite/example/mge/device_io.cpp>` 中的 ``pinned_host_input()`` 。示例中的模型运行在 device(CUDA) 上，但是输入输出在 CPU 上，为了加速 host2device 的copy，将 CPU 上的 input tensor 的内存指定提前申请为 cuda pinned 内存。目前如果输出output tensor 不是 device 上的时候，默认就是 pinned host 的。申请 pinned host 内存的方法是：构建 tensor 的时候指定 device，layout，以及 is_host_pinned参数，这样申请的内存就是 pinned host 的内存。

.. code-block:: cpp

     bool is_pinned_host = true;
     auto tensor_pinned_input =
             Tensor(LiteDeviceType::LITE_CUDA, input_layout, is_pinned_host);


用户指定内存分配器
--------------------

- 实现在 :src:`user_allocator.cpp <lite/example/mge/user_allocator.cpp>` 中的 ``config_user_allocator()`` 。

- 这个例子中使用用户自定义的 CPU 内存分配器演示了用户设置自定义的 Allocator 的方法，用户自定义内存分配器需要继承自 lite 中的 Allocator 基类，并实现 allocate 和 free 两个接口。目前在 CPU上验证是正确的，其他设备上有待测试。

- 设置自定定义内存分配器的接口为 Network 中如下接口：

.. code-block:: cpp

    Network& set_memory_allocator(std::shared_ptr<Allocator> user_allocator);


多个 Network 共享同一份模型 weights
-----------------------------------

- 实现在 :src:`network_share_weights.cpp <lite/example/mge/network_share_weights.cpp>` 中的 ``network_share_same_weights()`` 。

- 很多情况用户希望多个 Network 共享同一份 weights，因为模型中 weights 是只读的，这样可以节省模型的运行时内存使用量。这个例子主要演示了 lite 中如何实现这个功能，首先创建一个新的 Network，用户可以指定新的 Config 和 NetworkIO 以及其他一些配置，使得新创建出来的 Network 完成不同的功能。

- 通过已有的 NetWork load 一个新的 Network 的接口为 Network 中如下接口：

.. code-block:: cpp

        static void shared_weight_with_network(
            std::shared_ptr<Network> dst_network,
            const std::shared_ptr<Network> src_network);


**dst_network** 指新 load 出来的Network。**src_network** 指已经 load 的旧的 Network。


CPU 绑核
----------

- 实现在 :src:`cpu_affinity.cpp <lite/example/mge/cpu_affinity.cpp>` 中的 ``cpu_affinity()`` 。

- 该 example 之中指定模型运行在 CPU 多线程上，然后使用 Network 中的 ``set_runtime_thread_affinity()`` 来设置绑核回调函数。该回调函数中会传递当前线程的 id 进来，用户可以根据该 id 决定具体绑核行为，在多线程中，如果线程总数为 n，则 id 为 n-1 的线程为主线程。


用户注册自定义解密算法和 key
-----------------------------

- 实现在 :src:`user_cryption.cpp <lite/example/mge/user_cryption.cpp>` 中的 ``register_cryption_method()`` 和 ``update_aes_key()`` 。

- 这两个示例主要使用 lite 自定义解密算法和更新解密算法的接口，实现了使用用户自定的解密算法实现模型的 load 操作。在这个 example 中，自定义了一个解密方法，(其实没有做任何事情，将模型两次异或上 key 之后返回，等于将原始模型直接返回)，然后将其注册到 lite 中，后面创建 Network 时候在其config中的bare_model_cryption_name指定具体的解密算法名字。在第二个 example 展示了对其key 的更新操作。目前 lite 里面定义好了几种解密算法：

    * **AES_default** : 其 key 是由 32 个 unsighed char 组成，默认为0到31
    * **RC4_default** : 其 key 由 hash key 和 enc_key 组成的8个 unsigned char，hash
      key 在前，enc_key 在后。
    * **SIMPLE_FAST_RC4_default**: 其key组成同RC4_default。大概命名规则为：前面大写是具体算法的名字，'_'后面的小写，代表解密 key。具体的接口为：

.. code-block:: cpp

    bool register_decryption_and_key(std::string decrypt_name, 
    								const DecryptionFunc& func,
                                    const std::vector<uint8_t>& key);
    bool update_decryption_or_key(std::string decrypt_name,
                                    const DecryptionFunc& func,
                                    const std::vector<uint8_t>& key);

register 接口中必须要求三个参数都是正确的值，update中 decrypt_nam 必须为已有的解密算法，
将使用 func 和 key 中不为空的部分对 decrypt_nam 解密算法进行更新


异步执行模式
--------------

- 实现在 :src:`basic.cpp <lite/example/mge/basic.cpp>` 中的 ``async_forward()`` 。

- 用户通过接口注册异步回调函数将设置 Network 的 Forward 模式为异步执行模式，目前异步执行模式只有在 CPU 和 CUDA 10.0 以上才支持，在inference时异步模式，主线程可以在工作线程正在执行计算的同时做一些其他的运算，避免长时间等待，但是在一些单核处理器上没有收益。


纯 C example
--------------

- 实现在 :src:`lite_c_interface.cpp <lite/example/mge/lite_c_interface.cpp>` 中的 ``basic_c_interface()``， ``device_io_c_interface()`` 和 ``async_c_interface()`` 。

- Lite 完成对 C++ 接口的封装，对外暴露了纯 C 的接口，用户如果不是源码依赖 Lite 的情况下，应该使用纯 C 接口来完成集成。
- 纯 C 的所有接口都是返回一个 int，如果这个 int 的数值不为 0，则又错误产生，需要调用 ``LITE_get_last_error`` 来获取错误信息。
- 纯 C 的所有 get 函数都需要先定义一个对应的对象，然后将该对象的指针传递进接口，Lite 会将结果写入到 对应指针的地址里面。

