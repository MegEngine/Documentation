.. _module-stats:

==================
参数量与计算量统计
==================

有时候我们经常需要统计模型的参数量和计算量，目前的实现方式有两种：

* 基于 :py:class:`~.module` 的实现

  * 优点：可以在 Python 代码中嵌入调用，随时可以看统计信息
  * 缺点：由于是利用 module 的 hook 实现的，只能统计 module 的信息，无法统计 functional 的调用，如 F.matmul

* 基于 :py:meth:`~.trace.dump` 的实现

  * 优点：可以覆盖所有的 op，包括 matmul 或 batched matmul 等
  * 缺点：需要先进行 dump 操作

基于 module 的统计
------------------

实现在 :py:func:`module_stats`, 可以支持 float32/qat/qint8 模型的统计，使用方式很简单：

.. code-block::

   # 构建一个 net module，这里从 model hub 中获取 resnet18 模型
   net = load("megengine/models", "resnet18", pretrained=True)

   # 指定输入shape
   input_shape = (1, 3, 224, 224)

   # float model.
   total_params, total_flops = module_stats(
       net, input_shape, log_params=True, log_flops=True
   )
   print("params {} flops {}".format(total_params, total_flops))

可以通过 log_params 和 log_flops 来控制是否输出细节 parameter 和 flops 信息，返回总的参数量和计算量。

基于 dump 图的可视化与统计
--------------------------

基于 Python Graph 的图结构解析功能，输入 mge 格式的 dump 模型路径，
以及 log 存储目录，可将图结构信息存成 TensorBoard 可读的格式。

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

   TensorBoard 的安装和使用请参考 `TensorBoard 官网 <https://www.tensorflow.org/tensorboard>`_ 

如果启动服务器为远程 ssh 登陆，可用以下命令映射端口到本地（可使用 sshconfig 中的服务器名缩写）：

.. code-block:: shell

   ssh <user>@<host_name> -L 6006:0.0.0.0:6006 -N
