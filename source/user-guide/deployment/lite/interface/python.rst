.. _python-interface:

================================
MegEngine Lite python 接口介绍
================================

LiteTensor 相关 API
---------------------

LiteLayout
^^^^^^^^^^^^^^^^

.. code-block:: python

    def __init__(self, shape=None, dtype=None)

指定 shape 和 dtype，将构造出一个 LiteLayout。

参数：

* shape：LiteLayout 中的 shape 信息。
* dtype：LiteLayout 中的数据类型，这个 dtype 可以为如下类型：
  
    * 字符："int32"，"float32"，"uint8"，"int8"，"int16"，"uint16"，"float16"。
    * LiteDataType：LITE_FLOAT，LITE_HALF，LITE_INT，LITE_INT16，LITE_INT8，LITE_UINT8，LITE_UINT16。
    * numpy dtype 实例：np.dtype("int32")，np.dtype("float32")，np.dtype("uint8")，np.dtype("int8")，np.dtype("int16")，np.dtype("uint16")，np.dtype("float16")。
    * numpy dtype：numpy.int32，numpy.float32，numpy.uint8，numpy.int8，numpy.int16，numpy.uint16，numpy.float16。

LiteTensor
^^^^^^^^^^^^^^^^

.. code-block:: python

    def __init__(
        self,
        layout=None,
        device_type=LiteDeviceType.LITE_CPU,
        device_id=0,
        is_pinned_host=False,
        shapes=None,
        dtype=None,
    ):

构造 LiteTensor 时候可以指定 layout， device_type，device_id，以及是否内存是 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_，另外
如果也可以通过传递关键词参数 shapes 和 dtype 在 MegEngineLite 内部构造 layout，让后再构造 Tensor。

参数

* layout：LiteTensor 对应的 layout 信息。
* device_type：LiteDeviceType 对应的数据类型，包含：

    .. code-block:: python

        class LiteDeviceType(IntEnum):
        LITE_CPU = 0
        LITE_CUDA = 1
        LITE_ATLAS = 3
        LITE_NPU = 4
        LITE_DEVICE_DEFAULT = 5

* device_id：LiteTensor 所在设备的 id。
* is_pinned_host：LiteTensor 是否为 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_。

.. warning::

    LiteTensor 构造之后，内存没有立即申请，只有当 LiteTensor 需要用到内存时候才会申请。

示例：

    .. code-block:: python
        
        # 直接从 shapes 创建 LiteTensor
        tensor_cuda2 = LiteTensor(shapes=[4,16], dtype="float32", device_type=LiteDeviceType.LITE_CUDA)
        # 从 layout 创建 LiteTensor
        layout = LiteLayout([4, 16], "float32")
        tensor = LiteTensor(layout, LiteDeviceType.LITE_CPU)
        tensor_cuda = LiteTensor(layout=layout, device_type=LiteDeviceType.LITE_CUDA)

LiteTensor 信息获取
^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # 获取或者设置 LiteTensor 的 layout 信息
    @property
    def layout(self):
    @layout.setter
    def layout(self, layout): 

    # 获取 LiteTensor 是否是锁页内存
    @property
    def is_pinned_host(self):

    # 获取 LiteTensor 所在的设备类型
    @property
    def device_type(self):

    # 获取 LiteTensor 所在的设备 id
    @property
    def device_id(self):

    # 获取 LiteTensor 的内存是否是连续的 
    @property
    def is_continue(self):

    # 获取 LiteTensor 的内存的大小，单位是字节
    @property
    def nbytes(self):

.. note::
    
    上面 LiteTensor 的 layout 信息具有装饰器 @property 和 @layout.setter，可以直接作为成员一样访问和赋值，
    其他信息都具有 @property 的装饰器，因此都可以通过成员一样的访问。

get_ctypes_memory
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_ctypes_memory(self)

* get_ctypes_memory：将返回 ctypes.c_void_p 类型，其指向 Tensor 的内存地址，如果 Tensor 没有申请内存，将会申请内存。

reshape
^^^^^^^^^^

.. code-block:: python

    def reshape(self, shape):

改变这个 LiteTensor 的 LiteLayout 中的 shape 为新的 shape，其中 **新的 shape 中元素个数需要和老的 shape 里面的元素个数相等**。

slice
^^^^^^^^
.. code-block:: python

    def slice(self, start, end, step=None):
 
对 LiteTensor 进行切片，返回一个新的 LiteTensor，新的 LiteTensor 和原来 LiteTensor 共享内存， **新的 LiteTensor 可能不连续**

参数： **start，end 的长度必须相等，长度可以小于 Tensor 的 Layout 的维度，如果传递了 step，则 step 也需要和 start，end 的长度相等**。

* start：Tensor 每一维度的起始 index 组成的数组，从高维到低维。
* end：Tensor 每一维度的结束 index 组成的数组，从高维到低维。
* step：Tensor 每一维度切片的间距，从高维到低维，默认为1。

返回值：返回一个新的 LiteTensor。

示例：

.. code-block:: python

    layout = LiteLayout([4, 8], "int32")
    tensor1 = LiteTensor(layout)

    tensor1.set_data_by_copy([i for i in range(32)])
    real_data_org = tensor1.to_numpy()

    tensor2 = tensor1.slice([1, 4], [3, 8])
    assert tensor2.layout.shapes[0] == 2
    assert tensor2.layout.shapes[1] == 4
    assert tensor2.is_continue == False

    real_data = tensor2.to_numpy()
    for i in range(8):
        row = i // 4
        col = i % 4
        assert real_data[row][col] == real_data_org[row + 1][col + 4]

fill_zero
^^^^^^^^^^^^^

.. code-block:: python

   def fill_zero(self):

将 LiteTensor 内存里面的数据全部设置为 0。

copy_from
^^^^^^^^^^^^^^^^

.. code-block:: python

    def copy_from(self, src_tensor):

从 src_Tensor 中拷贝数据到自己内存中， **如果 src_tensor 和自己的 layout 不相同时，会更改自身 Layout 信息为 src layout**。

share_memory_with
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def share_memory_with(self, src_tensor):

将会和 src_tensor 共享内存数据， **如果 src_tensor 和自己的 LiteTensor 信息（layout，device_type，device_id等）不相同时，会更改自身信息为 src 的信息**。

示例：

.. code-block:: python

    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    tensor2 = LiteTensor(layout)

    tensor1.set_data_by_copy([i for i in range(32)])
    tensor2.share_memory_with(tensor1)
    real_data = tensor2.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i

update
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def update(self):

将 LiteTensor 底层的信息更新到 python 中的 LiteTensor 中，包括 LiteTensor 的设备，设备 id，layout等信息。

set_data_by_copy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def set_data_by_copy(self, data, data_length=0, layout=None):

将用户指定的 data 以 **复制的方式** 到该 LiteTensor 中。

参数：

* data： data 可以是 list 或者 numpy ndarray 或者 ctypes 的 c_void_p。

  * 当 data 类型为 list 时候，LiteTensor 的 Layout 不会被修改，用户需要保证 tensor 的内存大小大于 list 的长度。
  * 当 data 为 numpy ndarray 时候，如果 data 的长度和 LiteTensor 的内存大小不等时，将修改 LiteTensor 的 layout 为 data 的 layout。
  * 当 data 为 ctypes 的 c_void_p 时候，用户要么设置 data_length 并且必须 data_length LiteTensor 的长度相等，要么设置新的 Layout。

* data_length：当用户输入的 data 为 ctypes 的 c_void_p 时候，指明数据长度。
* layout 当需要改变 LiteTensor 的 layout 时，可以通过这个接口传递新的 layout。

.. warning::
    
    * LiteTensor 必须是 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_ 或者是 CPU 上的内存

示例：
            
.. code-block:: python

    layout = LiteLayout([2, 16], "int8")
    tensor = LiteTensor(layout)
                                   
    data = [i for i in range(32)]         
    tensor.set_data_by_copy(data)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

set_data_by_share
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def set_data_by_share(self, data, length=0, layout=None):

将用户传递进来的 data 通过 **共享的方式** 保存在 LiteTensor 中，避免 copy 带来的性能影响。

参数：

* data： data 可以是 numpy ndarray 或者 ctypes 的 c_void_p。

  * 当 data 为 numpy ndarray 时候，如果 data 的长度和 LiteTensor 的内存大小不等时，将修改 LiteTensor 的 layout 为 data 的 layout。
  * 当 data 为 ctypes 的 c_void_p 时候，用户要么设置 data_length 并且必须 data_length LiteTensor 的长度相等，要么设置新的 Layout。

* data_length：当用户输入的 data 为 ctypes 的 c_void_p 时候，指明数据长度。
* layout 当需要改变 LiteTensor 的 layout 时，可以通过这个接口传递新的 layout。

.. warning::
    
    * 当 data 为 numpy ndarray 时候，LiteTensor 要么是 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_ 要么是 CPU 上的内存
    * 当 data 为 ctypes 的 c_void_p 时候，对 LiteTensor 没有要求，这时候需要用户自己保证内存的设备属性。
    

示例：

.. code-block:: python

    layout = LiteLayout([2, 16], "int8")
    tensor = LiteTensor(layout)
    arr = np.ones([2, 16], "int8")
    for i in range(32):
        arr[i // 16][i % 16] = i
    tensor.set_data_by_share(arr)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

get_data_by_share
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_data_by_share(self):

将 LiteTensor 中的数据以 **共享** 的方式构建一个 numpy 的数组，并返回给用户， **当这个 LiteTensor 中内存数据被修改时候，返回的这个 numpy 数组中的
数据也将会被修改**。

返回值：

* 返回一个和 LiteTensor 共享内存的 numpy ndarray。

.. warning::
    
    * 当这个 LiteTensor 中内存数据被修改时候，返回的这个 numpy 数组中的数据也将会被修改，如：
    * 当第一次 Network Forward 之后通过输出 LiteTensor 获得的这个 numpy 数组，在下一次 Network Forward 的时候会被修改

示例：

.. code-block:: python

    layout = LiteLayout([4, 32], "int16") 
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 4 * 32 * 2

    arr = np.ones([4, 32], "int16")
    for i in range(128):
        arr[i // 32][i % 32] = i
    tensor.set_data_by_copy(arr)
    test_data = tensor.get_data_by_share()

    for i in range(128):
        assert test_data[i // 32][i % 32] == i

    arr[1][18] = 5
    arr[3][7] = 345
    tensor.set_data_by_copy(arr)
    assert test_data[1][18] == 5
    assert test_data[3][7] == 345

to_numpy
^^^^^^^^^^^^^^^^^

.. code-block:: python

    def to_numpy(self):

将 LiteTensor 中数据 copy 到一个 numpy 的 ndarray 中，可以方便查看 LiteTensor 中的数据。

.. note::
    
    * 当 LiteTensor 是 `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_ 或者是 CPU 上的 LiteTensor，则会直接 copy 到 numpy ndarray 中
    * 当 LiteTensor 在其他设备上，这时会先 copy 到 CPU LiteTensor 中，再从新的 LiteTensor copy 到 numpy ndarray 中，所以可能有 **性能问题**。

LiteOptions
^^^^^^^^^^^^^^^^

.. code-block:: python

    _fields_ = [
        ("weight_preprocess", c_int),
        ("fuse_preprocess", c_int),
        ("fake_next_exec", c_int),
        ("var_sanity_check_first_run", c_int),
        ("const_shape", c_int),
        ("force_dynamic_alloc", c_int),
        ("force_output_dynamic_alloc", c_int),
        ("force_output_use_user_specified_memory", c_int),
        ("no_profiling_on_shape_change", c_int),
        ("jit_level", c_int),
        ("comp_node_seq_record_level", c_int),
        ("graph_opt_level", c_int),
        ("async_exec_level", c_int),
        # layout transform options
        ("enable_nchw44", c_int),
        ("enable_nchw44_dot", c_int),
        ("enable_nchw88", c_int),
        ("enable_nhwcd4", c_int),
        ("enable_nchw4", c_int),
        ("enable_nchw32", c_int),
        ("enable_nchw64", c_int),
    ]

LiteOptions 是一个包含 MegEngine Network 优化选项集合的结构体，每个选项的解释如下：

* weight_preprocess：在推理时候，部分 Kernel 执行前需要对权重进行转换，或者 Relayout，开启这个选项之后，将权重处理放到 Kernel 执行之前， **优化 Kernel** 执行时间，但是 Network 初始化时间变长。
* fuse_preprocess：开启该选项之后，模型中的部分前后处理 Operator 将会被融合在一起，优化模型执行的性能。
* fake_next_exec：下一次执行 Inference 时候，是否为假的执行：仅仅完成内存分配等和计算无关的操作。这次假的执行完成之后将被设置为 false。
* var_sanity_check_first_run：第一次执行 Inference 时候是否需要对每一个 Operator 的输入输出 Tensor 的正确性进行检查，默认为 true。
* const_shape：指定 Network 的输入 shape 不会变化，这样不用在后面的执行时检查是否需要重新分配内存等操作。
* force_dynamic_alloc：强制要求所有的 Tensor 都是运行时动态分配，且不进行内存优化，MegEngine 默认所有的 Tensor 都是执行前进行内存优化并静态申请。
* force_output_dynamic_alloc：强制最后输出的 Tensor 的内存为动态申请，这样输出 Tensor 不用 copy 到用户的内存中，可以直接代理到返回内存给用户。
* force_output_use_user_specified_memory：强制让输出 Tensor 的内存由用户指定，这样输出 Tensor 将不需要 copy 到用户内存，在最后一个 Kernel 计算时就写到了用户的内存地址中。
* no_profiling_on_shape_change：当 Network 的输入 Tensor 的 shape 改变的时候，这时候 fast-run 将不会进行重新搜索最优的 kernel 算法实现。
* jit_level：JIT 的级别，设置为 0 时：将关闭 JIT，设置为 1 时：仅仅只开启基本的 elemwise 的 JIT，当是指为 2 时：将开启 elemwise 和 reduce Operator 的 JIT。
* comp_node_seq_record_level：设置 MegEngine 的录制模式，当设置为 0 时：将不开启录制模式，设置为 1 时：将开启录制模式，不会析构这个计算图结构，当设置为 2 时：将开启录制模式，并释放掉整个计算图。
* graph_opt_level：设置图优化等级，当设置为 0 时：关闭图优化，当设置为 1 时：算术计算 inplace 优化，当设置为 2 时：在 1 的基础上在加上全局优化，当设置为 3 时：在 2 的基础上再使能 JIT。
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

.. _lite_config:

LiteConfig
^^^^^^^^^^^^^^^^

    .. code-block:: python

        _fields_ = [
            ("has_compression", c_int),
            ("device_id", c_int),
            ("device_type", c_int),
            ("backend", c_int),
            ("bare_model_cryption_name", c_char_p),
            ("options", LiteOptions),
        ]

* has_compression： 模型是否压缩过。
* device_id： LiteNetwork 创建所在的设备 id。
* device_type：LiteNetwork 创建所在的设备类型。
* backend：指运行 LiteNetwork 的后端推理框架，目前默认是：MegEngine。
* bare_model_cryption_name：如果模型有加密，则指明加密算法的名字，如果没有加密，则不用配置。
* options 模型的优化参数，如上面所示。

.. _lite_io:

LiteIO
^^^^^^^^^^^^^^^^

    .. code-block:: python

        _fields_ = [
            ("name", c_char_p),
            ("is_host", c_int),
            ("io_type", c_int),
            ("config_layout", LiteLayout),
        ]

LiteIO 为指定模型中输入输出 LiteTensor 所在的位置，可以在 device 端，也可以配置在 CPU 端，如果不配置，默认为 CPU 端。

* name：LiteTensor 的名字，字符串。
* is_host：LiteNetwork 创建所在的设备 id。
* io_type：指定该 LiteTensor 对应的IO类型，目前支持两种类型，分别是：LITE_IO_VALUE 和 LITE_IO_SHAPE，默认为 LITE_IO_VALUE 。
* config_layout：提前配置好的 layout，不配置默认为模型中的 layout。

LiteNetworkIO
^^^^^^^^^^^^^^^^

    .. code-block:: python

        def __init__(self, inputs=None, outputs=None):

LiteNetworkIO 是 LiteNetwork 构造时候的 IO 信息的集合，包含 inputs 和 outputs，为用户指定的上述 LiteIO，另外用户可以通过 add_input，add_output
接口添加 LiteIO 到 LiteNetworkIO 中。

示例：

    .. code-block:: python

        input_io1 = LiteIO("data1", is_host=False, io_type=LiteIOType.LITE_IO_VALUE)
        input_io2 = LiteIO(
            "data2",
            is_host=True,
            io_type=LiteIOType.LITE_IO_SHAPE,
            layout=LiteLayout([2, 4, 4]),
        )
        io = LiteNetworkIO([input_io1, input_io2])

        io.add_output("out1", is_host=False)
        io.add_output("out2", is_host=True, layout=LiteLayout([1, 1000]))

        assert len(io.inputs) == 2
        assert len(io.outputs) == 2


LiteNetwork 相关 API
---------------------

LiteNetwork
^^^^^^^^^^^^^^^^

.. code-block:: python

    def __init__(self, config=None, io=None):

构造一个 LiteNetwork，可以传递两个参数分别是 config 和 io。

参数：

* config：模型优化需要的 LiteConfig 类型配置，默认为 None。
* io： LiteNetworkIO 类型，指定用户输入输出 LiteTensor 的信息。

示例：

.. code-block:: python

    config = LiteConfig()
    config.options.var_sanity_check_first_run = 0
    config.device_type = LiteDeviceType.LITE_CUDA

    ios = LiteNetworkIO(inputs=[LiteIO("data", False)])
    network = LiteNetwork(config=config, io=ios)

load
^^^^^^^^^^^^^^^^

.. code-block:: python

    def load(self, path):

指定创建 LiteNetwork 的模型路径，并解析这个模型，加载到内存中。

forward
^^^^^^^^^^^^^^^^

.. code-block:: python

    def forward(self):

对指定创建 LiteNetwork 进行 forward。

wait
^^^^^^^^^^^^^^^^

.. code-block:: python

    def wait(self):

等待指定创建 LiteNetwork 进行 forward 完成。

获取 LiteNetwork 相关信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # 获取 LiteNetwork 的运行所在的设备 id
    @property
    def device_id(self):

    # 获取 LiteNetwork 的运行所在的执行流 id
    @property
    def stream_id(self):

    # 获取 LiteNetwork 的运行在 CPU 多线程时候的线程个数
    @property
    def threads_number(self):

    # 获取 LiteNetwork 中输入 LiteTensor 中第 index 个的名字
    def get_input_name(self, index):

    # 获取 LiteNetwork 中输出 LiteTensor 中第 index 个的名字
    def get_output_name(self, index):

    # 获取 LiteNetwork 中所有输入 LiteTensor 的名字，返回一个 list
    def get_all_input_name(self):

    # 获取 LiteNetwork 中所有输出 LiteTensor 的名字，返回一个 list
    def get_all_output_name(self):

    # 获取 LiteNetwork 运行时候需要的内存信息，并将内存信息 dump 到 log_dir 指定的目录下
    def get_static_memory_alloc_info(self, log_dir="logs/test"):

    # 获取该 LiteNetwork 在 CPU 上运行时，是否为 inplace 模式
    def is_cpu_inplace_mode(self):

.. note::

    inplace 模式为：运行模型时候只有一个线程，这个线程发送 Kernel 任务的同时，inplace 地将
    kernel 执行计算任务。非 inplace 模式：将有2个线程，一个线程发送 Kernel 任务，一个线程执行 Kernel 任务。在一些单核处理器
    或者低端 cpu 上，设置 **inplace 模式性能会好一些**。

设置 LiteNetwork 相关信息
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    # 设置模型运行使用的设备 id
    @device_id.setter
    def device_id(self, device_id):

    # 设置模型运行使用的执行流 id
    @stream_id.setter
    def stream_id(self, stream_id):

    # 如果模型执行在 CPU 多线程的情况下，设置模型运行时候需要的线程数量
    @threads_number.setter
    def threads_number(self, nr_threads):

    # 如果模型在 CPU 上执行，设置模型运行模式为：inplace 模式
    def enable_cpu_inplace_mode(self):

    # 设置模型运行使用 TensorRT 进行推理
    def use_tensorrt(self):

.. warning::

    上面这些 LiteNetwork 的运行时的信息设置需要在 LiteNetwork 创建之后，模型 load 之前进行设置，否则将报错。

get_io_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def get_io_tensor(self, name, phase=LiteTensorPhase.LITE_IO):

获取 LiteNetwork 中名字为 name 的输入或者输出 LiteTensor。

参数：

* name：字符串，指定输入或者输出 LiteTensor 的名字。
* phase：当有输入和输出 LiteTensor 名字重复时候，指明获取的 LiteTensor 来自输入或者输出，可以设置为：

    * LiteTensorPhase.LITE_IO：在输入和输出的所有 LiteTensor 中寻找指定 name 的 LiteTensor，名字不会重复的情况下。
    * LiteTensorPhase.LITE_INPUT：在输入的所有 LiteTensor 中寻找指定 name 的 LiteTensor。
    * LiteTensorPhase.LITE_OUTPUT：在输出的所有 LiteTensor 中寻找指定 name 的 LiteTensor。

share_weights_with
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def share_weights_with(self, src_network):

设置 LiteNetwork 运行和 src_network 共享同一份权重，两个 LiteNetwork 可以对不同的输入数据进行推理，也可以同时运行。

share_runtime_memroy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
   def share_runtime_memroy(self, src_network):

设置 LiteNetwork 运行和 src_network 共享运行时候的内存， **这时 self 和 src_network 不能同时执行**，
运行时内存指：除了保存模型 weights 和图结构以外的所有需要的运行时内存。

async_with_callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def async_with_callback(self, async_callback):

设置模型 forward 运行在异步模式，异步模式中，主线程将不会被阻塞，当 LiteNetwork 执行完成之后将执行 async_callback，告诉主线程执行完成。

示例：

.. code-block:: python

    count = 0
    finished = False

    def async_callback():
        nonlocal finished
        finished = True
        return 0

    config = LiteConfig()
    config.options.var_sanity_check_first_run = 0

    network = LiteNetwork(config=config)
    network.load(model_path)

    network.async_with_callback(async_callback)

    network.forward()

    while not finished:
        count += 1

    assert count > 0
    output_data = output_tensor.to_numpy()

set_start_callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def set_start_callback(self, start_callback):

设置模型运行之前的回调函数，用户可以通过这个回调函数检查输入数据是否满足要求。

set_finish_callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def set_finish_callback(self, finish_callback):

设置模型运行之后的回调函数，用户可以通过这个回调函数检查输出数据是否满足要求。

示例：

.. code-block:: python

    network = LiteNetwork()
    network.load(model_path)
    finish_checked = False

    def finish_callback(ios):
        nonlocal finish_checked
        finish_checked = True
        assert len(ios) == 1
        for key in ios:
            io = key
            data = ios[key].to_numpy().flatten()
            output_data = self.correct_data.flatten()
            ...
        return 0

    network.set_finish_callback(finish_callback)
    network.forward()
    network.wait()
    assert finish_checked == True

enable_profile_performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def enable_profile_performance(self, profile_file):

模型运行时候，对模型中的各个 Operator 进行速度测试，并将测试结果写到指定的 profile_file 中，得到的这个 profile 文件为 json 文件，
可以使用 MegEngine 中指定的 tool 进行解析。

set_network_algo_workspace_limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def set_network_algo_workspace_limit(self, size_limit):

模型运行时候，模型中每一个 Operator 运行时候选择的算法最大能够用到的 workspace 大小，超过 size_limit 大小的算法将不会被选择，其中 size_limit 的单位为字节。

.. _set_network_algo_policy_python:

set_network_algo_policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def set_network_algo_policy(
        self, policy, shared_batch_size=0, binary_equal_between_batch=False
    ):

设置模型运行时候选择每个 Operator 算法的策略。

参数：

* policy 选择算法的策略，MegEngine Lite 中支持以下策略：

    .. code-block:: python

        class LiteAlgoSelectStrategy(IntEnum):
            """
            operation algorithm seletion strategy type, some operations have
            multi algorithms, different algorithm has different attribute, according to
            the strategy, the best algorithm will be selected.

            Note: These strategies can be combined

            LITE_ALGO_HEURISTIC | LITE_ALGO_PROFILE means: if profile cache not valid,
            use heuristic instead

            LITE_ALGO_HEURISTIC | LITE_ALGO_REPRODUCIBLE means: heuristic choice the
            reproducible algo

            LITE_ALGO_PROFILE | LITE_ALGO_REPRODUCIBLE means: profile the best
            algorithm from the reproducible algorithms set

            LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED means: profile the best
            algorithm form the optimzed algorithms, thus profile will process fast

            LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED | LITE_ALGO_REPRODUCIBLE means:
            profile the best algorithm form the optimzed and reproducible algorithms
            """

            LITE_ALGO_HEURISTIC = 1
            LITE_ALGO_PROFILE = 2
            LITE_ALGO_REPRODUCIBLE = 4
            LITE_ALGO_OPTIMIZED = 8

    其中上面的策略在不冲突的情况下，可以进行与操作，然后组合在一起。

* shared_batch_size：binary_equal_between_batch 的时候，选择最优算法所依据的 batch 大小，设置 0 将使用模型默认的 batch size。
* binary_equal_between_batch： 多个 batch 同时进行计算时，如果输入完全一样，保证所有 batch 的计算结果完全一样。


io_txt_dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def io_txt_dump(self, txt_file):

将 LiteNetwork 运行时候的所有 IO tensor 输出到文本文件 io_txt_out_file 中。

io_bin_dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
        
    def io_bin_dump(self, bin_dir):

将 LiteNetwork 运行时候的所有 IO tensor 以二进制的形式保存在 bin_dir 文件夹中。

全局设置相关 API 
---------------------
全局接口在 MegEngine Lite 中都封装在 LiteGlobal 中，都作为它的静态函数存在。

register_decryption_and_key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @staticmethod
    def register_decryption_and_key(decryption_name, decryption_func, key):

注册用户自定义的模型解密算法到 MegEngine Lite 中，包括解密方法和解密需要的秘钥。

参数：

* decryption_name：解密算法的名字，字符串。
* decryption_func：解密算法的方法，以及闭包函数。
* key：解密算法的秘钥。

示例：

.. code-block:: python

    @decryption_func
    def function(in_arr, key_arr, out_arr):
        if not out_arr:
            return in_arr.size
        else:
            for i in range(in_arr.size):
                out_arr[i] = in_arr[i] ^ key_arr[0] ^ key_arr[0]
            return out_arr.size

    LiteGlobal.register_decryption_and_key("just_for_test", function, [15])
    config = LiteConfig()
    config.bare_model_cryption_name = "just_for_test".encode("utf-8")

    network = LiteNetwork(config)
    model_path = os.path.join(self.source_dir, "shufflenet.mge")
    network.load(model_path)


update_decryption_key
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @staticmethod
    def update_decryption_key(decryption_name, key):

更新 MegEngine Lite 中 build-in 的解密算法的秘钥。

* decryption_name：解密算法的名字，目前 MegEngine Lite 中写了三种加密算法，分别是："AES_default"，"RC4_default" 和 "SIMPLE_FAST_RC4_default"。
* 对应的秘钥："AES_default" 为 32 字节数组，"RC4_default" 和 "SIMPLE_FAST_RC4_default" 为 16 自己数组。

示例：

.. code-block:: python

    wrong_key = [0] * 32
    LiteGlobal.update_decryption_key("AES_default", wrong_key)

    with self.assertRaises(RuntimeError):
        config = LiteConfig()
        config.bare_model_cryption_name = "AES_default".encode("utf-8")
        network = LiteNetwork(config)
        model_path = os.path.join(self.source_dir, "shufflenet_crypt_aes.mge")
        network.load(model_path)

    right_key = [i for i in range(32)]
    LiteGlobal.update_decryption_key("AES_default", right_key)

    config = LiteConfig()
    config.bare_model_cryption_name = "AES_default".encode("utf-8")
    network = LiteNetwork(config)
    model_path = os.path.join(self.source_dir, "shufflenet_crypt_aes.mge")
    network.load(model_path)


set_loader_lib_path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @staticmethod
    def set_loader_lib_path(path):

当第三方硬件以 loader 的形式接入到 MegEngine 中，该接口用于用户设置对应 loader 的执行动态库，path 为执行的动态库的路径。

set_persistent_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @staticmethod
    def set_persistent_cache(path, always_sync=False):

设置当前 MegEngine Lite 中模型运行的算法 cache，模型运行时将从这个 cache 中取出对应 Operator 的算法信息，并解析找到执行算法，并运行，
这将节省模型运行时候搜索最优的算法时候，用户可以提前搜索好对应的 cache。

dump_persistent_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @staticmethod
    def dump_persistent_cache(path):

将当前 MegEngine Lite 中模型运行的算法的 cache 从内存中 dump 到指定文件中，该方法可以用户用户提前所有最优算法的 cache。


dump_persistent_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @staticmethod
    def get_device_count(device_type):

获取指定 device_type 类型的设备数量。

try_coalesce_all_free_memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def try_coalesce_all_free_memory():

释放当前 MegEngine Lite 中所有不在需要的内存，这样将减少当前系统内存使用峰值。

tensorrt_cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def set_tensorrt_cache(path):
    def dump_tensorrt_cache():

设置以及下载 tensorRT 的 cache。

set_log_level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def set_log_level(LiteLogLevel):

    class LiteLogLevel(IntEnum):
        """
        DEBUG: The most verbose level, printing debugging info
        INFO: The default level
        WARN: Printing warnings
        ERROR: The least verbose level, printing errors only
        """

        DEBUG = 0
        INFO = 1
        WARN = 2
        ERROR = 3

设置 MegEngine Lite 的 log 级别，改函数不在 LiteGlobal 类中，是一个独立的全局函数。

物理地址和虚拟地址操作
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    
    def register_memory_pair(
            vir_ptr, phy_ptr, length, device, backend=LiteBackend.LITE_DEFAULT
        ):
    def clear_memory_pair(vir_ptr, phy_ptr, device, backend=LiteBackend.LITE_DEFAULT):
    def lookup_physic_ptr(vir_ptr, device, backend=LiteBackend.LITE_DEFAULT):

部分设备上有虚拟地址和物理地址的概念，这里提供用户操作虚拟地址和物理地址的接口，主要有：
* 设置全局的物理地址和虚拟地址对
* 清除这些地址对
* 通过虚拟地址查询物理地址

.. _lite_utils_api:

Utils API 
---------------------
MegEngine Lite 现在有一个 utils ，TensorBatchCollector，主要为方便用户在进行推理之前收集多个 batch 数据，然后将攒出来的一个多个 batch 的数据
同时放到 LiteNetwork 中进行推理，避免不必要的内存拷贝。

TensorBatchCollector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def __init__(
        self,
        shape,
        dtype=LiteDataType.LITE_INT8,
        device_type=LiteDeviceType.LITE_CUDA,
        device_id=0,
        is_pinned_host=False,
        tensor=None,
    ):

创建一个 TensorBatchCollector，这个 TensorBatchCollector 默认数据类型是 INT8，设备为 CUDA。

参数：

* shape：用户指定 TensorBatchCollector 的 shape。
* dtype：具体的数据类型，可以是 LITE_FLOAT，LITE_HALF，LITE_INT，LITE_INT16，LITE_INT8，LITE_UINT8，LITE_UINT16。
* device_type：具体的设备类型。
* device_id：TensorBatchCollector 所在的设备 id。
* is_pinned_host：该 TensorBatchCollector 申请的内存是否为： `锁页内存 <https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/>`_ 。
* tensor：可选的用户设置已经创建好的 LiteTensor 到 TensorBatchCollector 中。

collect_id
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def collect_id(self, array, batch_id):

设置该 TensorBatchCollector 中指定 batch_id 的数据为用户输入的 array。

参数：

* array：可以是 numpy 的 ndarry，也可以是 LiteTensor 类型。

    * 如果是 numpy 的 ndarry，MegEngine Lite 将调用 LiteTensor 的 set_data_by_copy 将数据 copy 到指定的 batch_id 的内存中。
    * 如果是 LiteTensor 类型，MegEngine Lite 将调用 LiteTensor 的 copy_from 完成数据 copy。

* batch_id：用户指定将要拷贝 array 数据的目标 batch。

collect_by_ctypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def collect_by_ctypes(self, data, length):

当用户的数据为 ctypes 的 c_void_p，可以调用该接口将数据设置到第一个空着的 batch 中。

collect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def collect(self, array):

当用户需要顺序的搜集batch，如从 0 一直到最大 batch，可以直接调用该接口。

free
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def free(self, indexes):

释放指定的 indexes，indexes 是一个 list。

get
^^^^^^^^

.. code-block:: python

   def get(self):

获得该 TensorBatchCollector 中内部存储数据的完整 LiteTensor。

to_numpy
^^^^^^^^

.. code-block:: python

    def to_numpy(self):

获得该 TensorBatchCollector 中的数据保存在 numpy 的 array 中，并返回。

* 示例1：顺序的进行攒 batch

    .. code-block:: python

        batch_tensor = TensorBatchCollector(
            [4, 8, 8], dtype=LiteDataType.LITE_INT, device_type=LiteDeviceType.LITE_CUDA
        )
        arr = np.ones([8, 8], "int32")
        for j in range(2):
            for i in range(4):
                batch_tensor.collect(arr)
                arr += 1
            batch_tensor.free(range(4))
        data = batch_tensor.to_numpy()
        assert data.shape[0] == 4
        assert data.shape[1] == 8
        assert data.shape[2] == 8
        for i in range(4):
            for j in range(64):
                assert data[i][j // 8][j % 8] == i + 4 + 1

* 示例2：通过指定 batch_id 进行攒 batch

    .. code-block:: python

        batch_tensor = TensorBatchCollector(
            [4, 8, 8], dtype=LiteDataType.LITE_INT, device_type=LiteDeviceType.LITE_CUDA
        )
        arr = np.ones([8, 8], "int32")
        arr += 1  # ==2
        batch_tensor.collect_id(arr, 1)
        arr -= 1  # ==1
        batch_tensor.collect_id(arr, 0)
        arr += 2  # ==3
        batch_tensor.collect_id(arr, 2)
        arr += 1  # ==4
        batch_tensor.collect_id(arr, 3)

        data = batch_tensor.to_numpy()
        batch_tensor.free(range(4))
        assert data.shape[0] == 4
        assert data.shape[1] == 8
        assert data.shape[2] == 8
        for i in range(4):
            for j in range(64):
                assert data[i][j // 8][j % 8] == i + 1

* 示例3：通过 ctpes 进行攒 batch

    .. code-block:: python

        all_tensor = LiteTensor(
            LiteLayout([4, 6, 8], dtype=LiteDataType.LITE_INT),
            device_type=LiteDeviceType.LITE_CUDA,
        )
        batch_tensor = TensorBatchCollector([4, 6, 8], tensor=all_tensor)
        nparr = np.ones([6, 8], "int32")
        for j in range(2):
            for i in range(4):
                batch_tensor.collect(nparr)
                nparr += 1
            batch_tensor.free(range(4))
        data = batch_tensor.to_numpy()
        assert data.shape[0] == 4
        assert data.shape[1] == 6
        assert data.shape[2] == 8
        for i in range(4):
            for j in range(48):
                assert data[i][j // 8][j % 8] == i + 4 + 1

* 示例4：通过 LiteTensor 进行攒 batch

    .. code-block:: python

        batch_tensor = TensorBatchCollector(
            [4, 6, 8], dtype=LiteDataType.LITE_INT, device_type=LiteDeviceType.LITE_CPU
        )
        nparr = np.ones([6, 8], "int32")
        tensor = LiteTensor(LiteLayout([6, 8], LiteDataType.LITE_INT))
        for j in range(2):
            for i in range(4):
                tensor.set_data_by_share(nparr)
                batch_tensor.collect(tensor)
                nparr += 1
            batch_tensor.free(range(4))
        data = batch_tensor.to_numpy()
        assert data.shape[0] == 4
        assert data.shape[1] == 6
        assert data.shape[2] == 8
        for i in range(4):
            for j in range(48):
                assert data[i][j // 8][j % 8] == i + 4 + 1