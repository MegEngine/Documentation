.. _environment-variables:

.. currentmodule:: megengine

===================
环境变量（Env）设置
===================

默认情况下，无需对环境变量进行更改，即可正常使用 MegEngine 框架。

如果你需要进行一定的改动，请确保你完全了解可能产生的影响。👻 👻 👻

注意事项
--------
.. note::

   MegEngine 中使用的环境变量可分为 “动态” 与 “非动态” 两类：
   
   * 动态环境变量可以动态地进行读取，即在代码运行过程中修改可即时生效。
   * 非动态环境变量在整个过程中只读取一次，后续对其进行修改将无法生效。
    
   对于非动态环境变量，我们会标注为 :ref:`nd-environ` 。

.. _nd-environ:

仅首次设置生效
~~~~~~~~~~~~~~

.. warning::

   需要注意，运行 ``import megengine`` 的过程中会读取部分环境变量。

   因此对于那些只读取一次的环境变量，推荐采用以下两种方法之一进行设置：

   * 在 Shell 中通过 ``export`` 进行临时设置，然后再运行代码；

   * 在需要运行的代码中进行设置，相关代码放在最开头处：

     .. code-block:: python

        os.environ['MGE_ENV_VAR']="value"  # Put this line before import megengine

        import megengine  # Read some environment variables
   

编译相关
--------
.. note::

   * 你可以在 :src:`CMakeLists.txt` 文件中找到 MegEngine 基本的 ``option`` 配置；
   * 第三方库的 cmake 配置文件可以在 :src:`cmake` 中找到。

设备相关
--------
``MGE_DEFAULT_DEVICE`` （ :ref:`nd-environ` ）
  设定默认使用计算设备，参考 :py:func:`~.set_default_device`.

``MEGENGINE_HOST_COMPUTE`` （ :ref:`nd-environ` ）
  是否允许在 Host 上进行简单的计算（即使已有 GPU 设备），默认开启。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

日志相关
--------
``MEGENGINE_LOGGING_LEVEL`` （ :ref:`nd-environ` ）
  设置 Log 等级，可选择 ``INFO``, ``DEBUG``, ``ERROR``.

``RUNTIME_OVERRIDE_LOG_LEVEL`` （ :ref:`nd-environ` ）
  设置 Runtime Log 等级，默认为 ``ERROR`` 级别，需通过数字进行设置：
  DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, NO_LOG = 4.

``MGB_DEBUG_VAR_SANITY_CHECK_LOG``
  对指定 ID 的 varnode 打印显存越界检查的 log, 默认为空，不开启。

``MGB_LOG_TRT_MEM_ALLOC``
  打印 tensorrt 申请的显存情况，默认为空，不开启。

分布式相关
----------
``MGE_PLASMA_MEMORY``
  :class:`~.DataLoader` 共享内存大小，默认单位为 B.

  设置为 ``0`` 表示不使用，为 ``100000000`` 表示 100MB, 为 ``4000000000`` 表示 4GB, 以此类推。

``MGE_DATALOADER_PLASMA_DEBUG``
  是否显示 :class:`~.DataLoader` 共享内存的输出和报错信息。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MGE_MM_OPR_DEBUG`` （ :ref:`nd-environ` ）
  是否输出多机算子的 Debug 信息，默认关闭。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

显存相关
--------
``MEGENGINE_INPLACE_UPDATE``
  是否原地修改模型参数，避免动态显存碎片，通常可以省一倍参数量的显存。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MGB_CUDA_RESERVE_MEMORY`` （ :ref:`nd-environ` ）
  是否占满所有 CUDA 显存，可能会对显存分配有一定优化。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MGB_ROCM_RESERVE_MEMORY`` （ :ref:`nd-environ` ）
  是否占满所有 ROCm 显存，可能会对显存分配有一定优化。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

Sublinear 相关
~~~~~~~~~~~~~~
.. note::

   参考 :py:class:`~.SublinearMemoryConfig` API 了解更多信息。

``MGB_SUBLINEAR_MEMORY_THRESH_NR_TRY``
  线性空间以及亚线性内存优化的当前范围搜索的样本数目，默认为 ``10``.

``MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER``
  使用遗传算法寻找最优切分策略时的迭代轮数，默认为 ``0``.

``MGB_SUBLINEAR_MEMORY_GENETIC_POOL_SIZE``
  遗传优化算法进行交叉随机选择（crossover）时所使用的样本数，默认为 ``20``.

``MGB_SUBLINEAR_MEMORY_LOWER_BOUND_MB``
  亚线性内存优化瓶颈大小的下界（以 MB 为单位），可用于在内存和速度之间进行手动权衡，默认为 ``0``.

``MGB_SUBLINEAR_MEMORY_WORKERS``
  搜索次线性内存优化最优切分策略时使用的线程数，默认为当前系统中CPU数目的一半。

``MEGENGINE_INPUT_NODE_USE_STATIC_SHAPE``
  给 InputNode 加一个 static shape 的模式，可以使更多的 var_node 是static_shape 并使用static storage，
  batch_size 可以开的更大，默认关闭。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

DTR 相关
~~~~~~~~
``MEGENGINE_ENABLE_SWAP`` （ :ref:`nd-environ` ）
  是否开启 swap 策略，默认不开启。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MEGENGINE_ENABLE_DROP`` （ :ref:`nd-environ` ）
  是否开启 drop 策略，默认不开启。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MEGENGINE_DTR_AUTO_DROP`` （ :ref:`nd-environ` ）
  是否开启自动 drop 策略，默认不开启。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

.. note::
   以上环境变量应当通过 :py:func:`~dtr.enable` 和 :py:func:`~dtr.disable` 进行控制。

``MEGENGINE_DTR_EVICTION_THRESHOLD``
  DTR 显存上限，超过后将采用自动 drop 策略，参考 :py:func:`~.eviction_threshold`.

``MEGENGINE_DTR_EVICTEE_MINIMUM_SIZE``
  应用 DTR 策略的 tensor 需要达到的大小，默认为 1048576B （1MB），参考 :py:func:`~.evictee_minimum_size`.

图机制相关
----------
``MEGENGINE_INTERP_ASYNC_LEVEL`` （ :ref:`nd-environ` ）
  动态图的执行并行度，``0`` 是完全串行，``1`` 是计算异步，``2`` 是用户代码和计算都异步（默认）。
  设置为 0 将使 MegEngine 上层的任务队列变成同步执行，即 Python 调用一个 Op, C++ 层执行一个 Op,
  没执行完前 Python 层不会走到下一句，便于定位 Python 层报错的位置，但会影响速度。

``MEGENGINE_CATCH_WORKER_EXEC`` （ :ref:`nd-environ` ）
  是否捕获动态图 worker 的异常，默认开启，Debug 时可将其关闭。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MEGENGINE_COMMAND_BUFFER_LENGTH`` （ :ref:`nd-environ` ）
  延迟执行最后面计算算子的个数（默认为 ``3`` ），用于做局部优化。

``MEGENGINE_RECORD_COMPUTING_PATH`` （ :ref:`nd-environ` ）
  是否记录 tensor 的历史计算路径，默认关闭。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。

``MEGENGINE_EXECUTION_STRATEGY`` （ :ref:`nd-environ` ）
  设置 kernel 选择策略（fast-run），影响运行速度、是否可复现以及编译时间：

  * ``HEURISTIC`` - 表示启发式选择 kernel
  * ``PROFILE`` -  表示根据 profile 时间选择 kernel
  * ``REPRODUCEABLE`` - 表示使用可复现的算法
  * ``OPTIMIZED`` - 表示使用经过优化的算法

  默认为 ``HEURISTIC``, 参考 :py:func:`~.set_execution_strategy` 了解更多。

``MGB_CONV_PROFILING_TIMEOUT`` （ :ref:`nd-environ` ）
  Profile 超时阈值，超时则直接杀掉 Kernel 运行，默认为 0 表示不做限制。

``MGB_PROFILE_ONLY_WAIT`` （ :ref:`nd-environ` ）
  Prifile 时只选择有 wait 行为的算子，默认为空，不开启。

``CUDA_BIN_PATH`` （ :ref:`nd-environ` ）
  设置 CUDA 编译器 nvcc 的路径，用于编译 fuse kernel.

  默认从 ``PATH``, ``LIBRARY_PATH`` 环境变量中寻找，也可人为指定路径如 ``"/data/opt/cuda/bin/"``.

``MGB_JIT_BACKEND`` 
  jit fuse kernel 的编译后端选项，可设置为 ``HALIDE``, ``NVRTC``, ``MLIR``.

``MGB_JIT_KEEP_INTERM`` （ :ref:`nd-environ` ）
  是否保存 jit 产生的临时文件，默认为空，不保存。

``MGB_JIT_WORKDIR`` （ :ref:`nd-environ` ）
  jit 产生的临时文件目录路径，默认为 ``/tmp/mgbjit-XXXXXX``. 

``MGB_DUMP_INPUT`` （ :ref:`nd-environ` ）
  是否在 Dump 的同时导出每个算子时输入值，默认为空，不开启。

调试相关
--------
``MGB_WAIT_TERMINATE`` （ :ref:`nd-environ` ）
  在 MegEngine 崩溃的时候进入等待，此时可以用 gdb attch 进行 debug, 默认为空，不开启。

``MGB_THROW_ON_FORK`` （ :ref:`nd-environ` ）
  是否在 Fork 进程中抛出异常，默认为空，不开启。

``MGB_THROW_ON_SCALAR_IDX`` （ :ref:`nd-environ` ）
  Tensor index 的下标如 果是scalar, 则可以使用 subtensor, 选择是否抛出异常，默认关闭。

  设置为 ``0`` 表示关闭，为 ``1`` 表示开启。



