.. _profiler-guide:

==================================
模型性能数据生成与分析（Profiler）
==================================

.. note::

   由于实现限制，:ref:`动态图与静态图 <dynamic-and-static-graph>` 下的 Profiler 接口并不一致，
   侧重点也不相同，下面将分别介绍。

动态图下的性能分析
------------------
假设我们写好了一份动态图代码，其中训练部分代码如下：

.. code-block:: python

   def train_step(data, label, *, optimizer, gm, model)
       with gm:
           logits = model(data)
           loss = F.loss.cross_entropy(logits, label)
           gm.backward(loss)
           optimizer.step().clear_grad()
       return loss

生成性能数据
~~~~~~~~~~~~
.. warning::

   挂载 Profiler 会拖慢模型的运行速度（大概在 8% 左右）。

想要使用 Profiler 生成性能数据，存在两种写法（任选其一即可）：

* 使用 :py:data:`~megengine.utils.profiler.profile` 装饰器 （profile 是 Profiler 别名）
* 使用 with :py:class:`~.utils.profiler.Profiler` 语法

示例代码如下：

.. code-block:: python

   from megengine.utils.profiler import profile, Profiler

   # 装饰器写法
   @profile()
   def train_step(data, label, *, optimizer, gm, model)
       with gm:
           logits = model(data)
           loss = F.loss.cross_entropy(logits, label)
           gm.backward(loss)
           optimizer.step().clear_grad()
       return loss

   # with 写法
   profiler = Profiler()
   def train_step(data, label, *, optimizer, gm, model)
       with profiler:
          with gm:
              logits = model(data)
              loss = F.loss.cross_entropy(logits, label)
              gm.backward(loss)
              optimizer.step().clear_grad()
       return loss

这样在每次进到对应代码块里时，MegEngine 会对区域里的代码单独做一次 Profiling.

程序结束时（准确地说，是在Profiler析构时），将会在运行目录下生成 ``JSON`` 文件，用于接下来的性能分析。

参数说明
^^^^^^^^

:py:class:`~.utils.profiler.Profiler` 的构造函数支持如下参数：

``path``
  profile数据的存储路径，默认为当前路径下的 ``profile`` 文件夹.

``format``
  输出数据的格式，默认为 ``chrome_timeline.json`` ，是Chrome支持的一种标准格式，以时间线的形式展现profiling结果.
  可选项还有 ``有memory_flow.svg`` ，以时间x地址空间的形式展示内存使用情况.

``formats``
  若需要的输出格式不止一种，可以在formats参数里列出.

``sample_rate``
  若该项不为零，则每隔n个op会统计一次显存信息，分析数据时可以绘制显存占用曲线，默认为0.

``profile_device``
  是否记录gpu耗时，默认为True.

分析性能数据
~~~~~~~~~~~~
可以使用 `Perfetto <https://ui.perfetto.dev/>`_
工具加载上一步生成的 ``JSON`` 文件：

#. 打开 `Perfetto 网页 <https://ui.perfetto.dev/>`_ ；
#. 点击 ``Open trace file`` 按钮加载数据；
#. 展开内容。

此时可以在窗口里看到数个线程，每个线程都按时间顺序显示历史调用栈。
横坐标是时间轴，色块的左右边缘是事件的起始与终止时间。
纵坐标代表事件所属的线程（其中 channel 为 python 主线程）。
例如，当我们在模型源代码里的 ``self.conv1(x)`` 被执行时，
channel 线程上会有一个对应的 ``conv1`` 块，而其他线程上同样的 ``conv1`` 块会滞后一些。
而 worker 的主要工作是发送 kernel, 而真正执行计算的是 gpu  线程。
gpu 线程上的事件密度明显比 channel 和 worker 高。

.. note::

   * 一般来说，GPU 线程越繁忙，说明模型的 GPU 利用率越高。
   * 频繁使用 :py:meth:`.Tensor.shape` , :py:meth:`.Tensor.numpy` 
     操作都可能导致需要做数据同步，降低 GPU 的利用率。

以下操作会在 Performance 界面里默认以色块的形式呈现：

* :py:meth:`.GradManager.backward`
* :py:meth:`.Optimizer.step`
* :py:meth:`.Optimizer.clear_grad`
* :py:meth:`.Module.forward`

通过观察事件的持续时间，可以评估模型的性能瓶颈。
在timeline的上方还会有一些曲线这些曲线与下方的事件共用同一条时间轴，展示了对应数据的变化过程。


静态图下的性能分析
------------------
假设我们写好了一份静态图代码，其中训练部分代码如下：

.. code-block:: python

   @trace(symbolic=True)
   def train_step(data, label, *, optimizer, gm, model)
       with gm:
           logits = model(data)
           loss = F.loss.cross_entropy(logits, label)
           gm.backward(loss)
           optimizer.step().clear_grad()
       return loss

生成性能数据
~~~~~~~~~~~~
只需要在 :py:class:`~.jit.trace` 接口中传入 ``profiling=True``,
然后再调用 :py:meth:`~.trace.get_profile` 方法即可得到性能数据。

修改后的代码如下：

.. code-block:: python

   @trace(symbolic=True, profiling=True)
   def train_step(data, label, *, optimizer, gm, model)
       with gm:
           logits = model(data)
           loss = F.loss.cross_entropy(logits, label)
           gm.backward(loss)
           optimizer.step().clear_grad()
       return loss

    ... # 训练代码，调用了 train_step()

    # 得到性能数据
   prof_result = train_func.get_profile()

   # 保存结果为 JSON 格式
   with open("profiling.json", "w") as fout:
       json.dump(prof_result, fout, indent=2)

这样我们将获得一个 ``JSON`` 文件，可用于下面的性能分析。

.. _profile-analyze:

分析性能数据
~~~~~~~~~~~~
在前一步中保存的 ``JSON`` 文件可以使用 MegEngine 在 ``tools`` 
目录下提供的 ``profile_analyze.py`` 脚本进行分析，示例代码如下：

.. code-block:: bash

    # 输出详细帮助信息
    python3 -m megengine.tools.profile_analyze -h

    # 输出前 5 慢的算子
    python3 -m megengine.tools.profile_analyze ./profiling.json -t 5

    # 输出总耗时前 5 大的算子的类型
    python3 -m megengine.tools.profile_analyze ./profiling.json -t 5 --aggregate-by type --aggregate sum

    # 按 memory 排序输出用时超过 0.1ms 的 ConvolutionForward 算子
    python3 -m megengine.tools.profile_analyze ./profiling.json -t 5 --order-by memory --min-time 1e-4  --type ConvolutionForward

输出将是一张表格，每列的含义如下：

``device self time``
  算子在计算设备上（例如 GPU ）的运行时间

``cumulative``
  累加前面所有算子的时间

``operator info``
  打印算子的基本信息

``computation``
  算子需要的浮点数操作数目

``FLOPS`` 
  算子每秒执行的浮点操作数目，由 ``computation`` 除以 ``device self time`` 并转换单位得到

``memory``
  算子使用的存储（例如 GPU 显存）大小

``bandwidth``
  算子的带宽，由 ``memory`` 除以 ``device self time`` 并转换单位得到

``in_shapes``
  算子输入张量的形状

``out_shapes``
  算子输出张量的形状

