.. _load-and-run-py:

===============================
如何使用 Load and Run（Python）
===============================

Python 版 Load and Run 的主要功能和用法与 :ref:`C++ 版本 <load-and-run>` 基本一致。

命令行中输入以下命令可以看到全部参数的列表与介绍：

.. code-block:: shell

   python -m megengine.tools.load_network_and_run --help

参数列表及简介如下：

``net``
  **必需参数** ，指定 mge graph 文件路径

``--output-name [OUTPUT_NAME [OUTPUT_NAME ...]]``
  指定用于测速的网络输出数据节点的名字，可用空格分隔指定多个，不指定则为网络编译 dump 时的输出

``--load-input-data LOAD_INPUT_DATA``
  指定用作输入的 inputs data 文件路径，内容应该为 pickle 化的 numpy array 或者含 np array 的 dict，key 为 inputs 节点名

``--input-desc INPUT_DESC``
  对于未指定 --load-input-data 的情况，会根据的 INPUT_DESC 指定的 shape 来随机生成数据，如果不指定，则会根据图中节点的 shape 来生成

``--batchsize BATCHSIZE``
  类似 --input-desc，但是只指定 batchsize

``--seed SEED``
  指定生成随机 inputs 数据的随机数种子

``--rng RNG``
  指定生成随机数的配置，包括范围、大小等，也可用 python function 指定，指定大小时应确保与 inputs shape 一致

``--profile PROFILE``
  开启后使用 GraphProfiler 记录 profile 信息，并将结果的 json 内容写到 PROFILE 文件路径中，可后续用于 profile_analyze.py 分析

``--focused-nvprof``
  会在最后额外跑一个用 pycuda.driver profiler 包起来的 iter，用于外部 nvprof 进行测速

``--warm-up``
  在开始测速前，先跑一个迭代，减少设备缓存等因素带来的性能影响

``--iter ITER``
  正式运行测速的迭代数

``--calc-output-rms``
  在运行日志中附带 outputs RMS(root meam square) 值的结果，用于快速比较两次输出结果是否一致

``--device``
  （目前无效）指定 mge graph 加载时使用的 device，等同于 MGE_DEFAULT_DEVICE 环境变量

``--fast-run``
   设置网络中 conv、matmul 等支持修改 execution strategy 选项的算子的执行算法，开启后会对当前平台的多个算法进行运行测速，选出最快的算法

``--reproducible``
  影响 --fast-run 选项中的算法选择，开启后只选择带"reproducible"标签的算法进行比较

``--optimize-for-inference 以及配套的 --enable-xxx 参数``
  对 mge graph 进行优化，会导致图节点被替换、修改，具体选项参考 MegEngine 的 megbrain_graph.optimizer_for_inference 接口

``--embed-input``
  是否将 inputs data 作为 SharedDeviceTensor 嵌入网络中替换 Host2Device 节点，以兼容 C++ 版 load-and-run，
  不开启时 h2d 会被替换为 InputCallback 节点以支持 set_value 

``--dump-cpp-model DUMP_CPP_MODEL``
  在依次做完fast-run修改、optimize、embed-inputs等操作后，添加output callback前，将网络进行 dump

``--verbose, -v``
  设置 log level 为 DEBUG，输出更多信息（包括 fast-run 测速过程等）

``--log LOG``
  指定 log 输出的保存路径，不指定则不保存
