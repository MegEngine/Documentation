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

   # 指定输入 shape
   input_shape = (1, 3, 224, 224)

   # Float model.
   total_params, total_flops = module_stats(
       net, input_shape, log_params=True, log_flops=True
   )
   print("params {} flops {}".format(total_params, total_flops))

可以通过 ``log_params`` 和 ``log_flops`` 参数来控制是否输出 parameter 和 flops 细节信息，返回总的参数量和计算量。

基于 dump 图的可视化与统计
--------------------------

基于 Python Graph 的图结构解析功能实现：

* 输入 mge 格式的 dump 模型路径以及 log 存储目录
* 可将图结构信息存成 TensorBoard 可读的格式。

命令行调用
~~~~~~~~~~

.. code-block:: shell

   python3 -m megengine.tools.network_visualize ./resnet18.mge ./log --log_flops --log_params

其中各个参数说明如下：

``./resnet18.mge`` （第一个参数）
   **必填参数** ，指定模型文件名。

``./log`` （第二个参数）
  **必填参数** ，指定 log 存储目录。

``--log_flops``
   指定当前屏打印出 FLOPs 信息。
  
``--log_params``
   指定当前屏打印出 Parameters 信息。

Python 中调用
~~~~~~~~~~~~~

以下代码等效于上方的命令行调用方式：

.. code-block:: python

   from megengine.tools.network_visualize import visualize

   total_params, total_flops = visualize(
       "./resnet18.mge", "./log"
   )
   print("params {} flops {}".format(total_params, total_flops))

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
