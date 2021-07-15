.. _stats:

========================
参数和计算量统计与可视化
========================

借助一些工具，我们可以统计 MegEngine 模型的参数量和计算量，目前的实现方式有两种：

* 基于 :py:mod:`~.module` 的实现——

  * 优点：可以在 Python 代码中嵌入调用，随时可以看统计信息
  * 缺点：只能统计 :py:mod:`~.module` 的信息，无法统计 :py:mod:`~.functional` 的调用

* 基于 :py:meth:`~.trace.dump` 的实现——

  * 优点：可以覆盖所有的算子
  * 缺点：需要先进行 :py:meth:`~.trace.dump` 操作

基于 module 的统计
------------------

实现在 :py:func:`~.module_stats` 中, 可以支持 float32/qat/qint8 模型的统计，使用方式很简单：

.. code-block::

   from megengine.hub import load
   from megengine.utils.module_stats import module_stats

   # 构建一个 net module，这里从 model hub 中获取 resnet18 模型
   net = load("megengine/models", "resnet18", pretrained=True)

   # 指定输入
   input_data = np.random.rand(1, 3, 224, 224).astype("float32")

   # Float model.
   total_stats, stats_details = module_stats(
       model,
       inputs=input_data,
       cal_params=True,
       cal_flops=True,
       cal_activations=True,
       logging_to_stdout=True,
   )
   print("params {} flops {} acts {}".format(total_stats.param_dims, total_stats.flops, total_stats.act_dims))

可以通过 ``cal_params`` 、 ``cal_flops`` 和 ``cal_activations`` 来控制是否计算parameter、flops和activations信息，
通过 ``logging_to_stdout`` 来控制是否将计算的细节信息打印出来，返回总的统计信息和详细统计信息的namedtuple，可以查看每个统计量的总量和每个模块的分量。

基于 dump 图的可视化与统计
--------------------------

基于 Python Graph 的图结构解析功能实现：

* 输入 mge 格式的 dump 模型路径以及 log 存储目录
* 可将图结构信息存成 TensorBoard 可读的格式。

命令行调用
~~~~~~~~~~

.. code-block:: shell

   python -m megengine.tools.network_visualize ./resnet18.mge --log_path ./log --load_input_data data.pkl --cal_flops --cal_params --cal_activations --logging_to_stdout

其中各个参数说明如下：

``./resnet18.mge`` （第一个参数）
   **必填参数** ，指定模型文件名。

``./log`` （第二个参数）
  **必填参数** ，指定 log 存储目录。

``--load_input_data``
   指定输入数据文件路径，文件内容应该为 pickle 化的 numpy array 或者含 numpy array 的 dict，key 为 inputs 节点名。

``--cal_flops``
   指定统计 FLOPs 信息。
  
``--cal_params``
   指定统计 Parameters 信息。

``--cal_activations``
   指定统计 activations 信息。

``--logging_to_stdout``
   指定当前屏打印出所有统计量的信息。

Python 中调用
~~~~~~~~~~~~~

以下代码等效于上方的命令行调用方式：

.. code-block:: python

   from megengine.tools.network_visualize import visualize

   input_data = np.random.rand(1, 3, 224, 224).astype("float32")
   total_stats, stats_details = visualize(
       "./resnet18.mge", 
       "./log",
       input=input_data,
       cal_flops=True,
       cal_params=True,
       cal_activations=True,
       logging_to_stdout=True
   )
   print("params {} flops {} acts {}".format(total_stats.param_dims, total_stats.flops, total_stats.act_dims))

进行可视化
~~~~~~~~~~

完成上面的步骤后，再在对应目录（例子中为 ``./log`` ）启动 tensorboard, 即可在本机打开 tensorboard 进程：

.. code-block:: shell

   tensorboard --logdir ./log

.. note::

   TensorBoard 的安装和使用请参考 `TensorBoard 官网 <https://www.tensorflow.org/tensorboard>`_ 。 

如果启动服务器为远程 ssh 登陆，可用以下命令映射端口到本地（可使用 sshconfig 中的服务器名缩写）：

.. code-block:: shell

   ssh <user>@<host_name> -L 6006:0.0.0.0:6006 -N
