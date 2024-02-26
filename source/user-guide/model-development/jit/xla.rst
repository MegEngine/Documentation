.. _xla_jit:

=================================
使用 XLA 作为编译后端加速模型训练
=================================

.. note::

   * XLA 作为 jit 后端目前属于实验特性，可能存在一些 bug
   * 算子支持还不完备，可能存在部分算子不支持/算子 lowering 不正确的问题
   * 相关接口不稳定， 后续版本可能会有 API 变动


XLA 编译后端
--------------------
`XLA <https://github.com/openxla/xla>`_ 是 google 推出的可用于模型训练加速的编译器。 MegEngine 接入了 XLA 编译器用于提高训练性能。

在 MegEngine 中使用 XLA 编译器需要用户安装 mge_xlalib 包。 

.. note::
   
    * 仅支持在 MegEngine 1.13 及以上版本使用 XLA 编译后端。
    * 安装的 mge_xlalib 包的 cuda 版本需要与用户使用的 MegEngine 包相同。 
    * mge_xlalib 目前支持的 python 版本是 3.8 到 3.10， cuda 版本是 11.1 11.4 11.8 。
    * mge_xlalib cuda 11.8 加速效果最好， cuda 11.1 由于 cudnn 版本较低 XLA 加速效果较差。 

mge_xlalib 安装命令如下：

.. code-block:: shell

   # cuda 11.1
   python3 -m pip install mge-xlalib==0.4.7+cuda11010.cudnn804 -f https://www.megengine.org.cn/whl/mge.html
   # cuda 11.4
   python3 -m pip install mge-xlalib==0.4.7+cuda11040.cudnn821 -f https://www.megengine.org.cn/whl/mge.html
   # cuda 11.8
   python3 -m pip install mge-xlalib==0.4.7+cuda11080.cudnn860 -f https://www.megengine.org.cn/whl/mge.html

xla 在编译优化时需要使用 nvptx 等工具进行运行时编译，所以我们需要在环境中安装相关依赖等，对于 cuda 11.8，nvidia 已经支持 pip 安装

.. code-block:: shell

   pip install "nvidia-cuda-cupti-cu11>=11.8" "nvidia-cuda-nvcc-cu11>=11.8" "nvidia-cuda-runtime-cu11>=11.8"

对于 cuda 11.1 和 cuda 11.4，则需要手动自行安装 cuda，并把 cuda/bin 等目录加入 PATH 中。故而从性能和使用便利性上来说，如果想使用 mge-xla，更推荐使用 cuda 11.8。

XLA 编译器的使用方式与 MegEngine graph runtime 自带编译器类似， 需要用MegEngine提供的装饰器 (xla_trace)
对训练函数进行包装。 函数执行第一遍时会记录算子执行序列，以捕获静态图。 后续执行会把静态图用XLA编译， 并调用编译好的
XLA executable 加速训练过程。

由于 xla_trace 实现和 XLA 编译器自身的特性， 目前该功能的使用还有如下限制：

    * 所有（非常量的）外部 Tensor /包含外部 Tensor 的对象（如 Module， Optimizer等）需要作为被装饰器包装的训练函数的参数传入。外部 Tensor 指所有不是由 Op 计算产生的 Tensor（对应静态图中的输入节点）。
    * 由于 XLA 目前对动态 Shape 支持较差，需要保证所有外部 Tensor 的 Shape 是固定的。
    * 需要用户保证被 trace 函数是静态的（函数不会在输入不同时执行不同的分支）。


具体使用示例见 :ref:`xla_trace` 和 :ref:`partial_trace`


.. _xla_trace:

使用 xla_trace 装饰器
----------------------

您可以使用 jit.xla_trace 装饰器对需要加速的函数进行包装， 代码示例如下。 

.. code-block:: python

   import numpy as np

   import megengine as mge
   import megengine.functional as F
   from megengine.jit import partial_trace, xla_trace


   def softmax(inp):
       offset = inp.max(axis=-1, keepdims=True).detach()
       cached = F.exp(inp - offset)
       down = F.sum(cached, axis=-1, keepdims=True)
       return cached / down


   @xla_trace
   def xla_fused_softmax(inp):
       offset = inp.max(axis=-1, keepdims=True).detach()
       cached = F.exp(inp - offset)
       down = F.sum(cached, axis=-1, keepdims=True)
       return cached / down

   inp = mge.tensor(np.random.randn(256, 1000, 1000))
   xla_fused_softmax(inp) # run in imperative runtime, trace op sequence
   print (softmax(inp))
   print (xla_fused_softmax(inp)) # run in xla
    

如果我们想看到 mge 和 xla 优化的一些中间 IR 表示，可以通过设置环境变量 MGE_VERBOSE_XLA_IR 来打印相关结果。MGE_VERBOSE_XLA_IR 为 1 时，会打印 mge trace 出来的图 IR，MGE_VERBOSE_XLA_IR 为 2 时，会打印xla 的 hlo 图结构，在 MGE_VERBOSE_XLA_IR 为 3 时会打印 xla 编译优化后的图结构。如果我们 export MGE_VERBOSE_XLA_IR=1 后再执行上述代码，则可以看到：

.. code-block:: python

   please_realize_func_name_system_1(
       0%:<256x1000x1000,f32>
   ) {
       1%:<256x1000x1000,f32> = io_mark_var(0%:<256x1000x1000,f32>)
       2%:<256x1000x1,f32> = ReduceMAX(1%:<256x1000x1000,f32>)
       3%:<256x1000x1000,f32> = SUB(1%:<256x1000x1000,f32>, 2%:<256x1000x1,f32>)
       4%:<256x1000x1000,f32> = EXP(3%:<256x1000x1000,f32>)
       5%:<256x1000x1,f32> = ReduceSUM(4%:<256x1000x1000,f32>)
       6%:<256x1000x1000,f32> = TRUE_DIV(4%:<256x1000x1000,f32>, 5%:<256x1000x1,f32>)
       7%:<256x1000x1000,f32> = io_mark_var(6%:<256x1000x1000,f32>)
       return 1 7%:<256x1000x1000,f32>
   }

当模型训练迭代（Iteration）完全静态的情况下， 您也可以使用 jit.xla_trace 装饰器将训练迭代全部交由XLA执行。
需要将 optimizer， module 作为train_func 参数传入，同时 train_func 中需包含包含模型前向、 反向
、 参数更新等代码，
代码示例如下：

.. code-block:: python

   :emphasize-lines: 44-51, 58

   from functools import partial
   import numpy as np

   import megengine
   import megengine.autodiff as autodiff
   import megengine.functional as F
   import megengine.module as M
   from megengine import distributed as dist
   from megengine.jit import partial_trace, xla_trace
   from megengine.optimizer import AdamW

   class ConvNet(M.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = M.Conv2d(3, 6, 5, bias=False)
           self.bn1 = M.BatchNorm2d(6)
           self.conv2 = M.Conv2d(6, 16, 5, bias=False)
           self.bn2 = M.BatchNorm2d(16)
           self.fc1 = M.Linear(16 * 5 * 5, 120)
           self.fc2 = M.Linear(120, 84)
           self.classifier = M.Linear(84, 10)
           self.pool = M.AvgPool2d(2, 2)

       def forward(self, x):
           x = self.pool(self.bn1(self.conv1(x)))
           x = self.pool(self.bn2(self.conv2(x)))
           x = F.flatten(x, 1)
           x = self.fc1(x)
           x = self.fc2(x)
           x = self.classifier(x)
           return x

   @dist.launcher(n_gpus=2, device_type="gpu")
   def worker():
       def runner():
           model = ConvNet()
           model.train()
           dist.bcast_list_(model.tensors())

           cblist = [dist.make_allreduce_cb("mean")]
           gm = autodiff.GradManager().attach(model.parameters(), callbacks=cblist)
           optimizer = AdamW(model.parameters(), lr=0.01)

           @xla_trace(without_host=True, capture_as_const=True)
           def func(model, optimizer, timage, tlabel):
               with gm:
                   score = model(timage)
                   loss = F.nn.cross_entropy(score, tlabel)
                   gm.backward(loss)
                   optimizer.step().clear_grad()
               return loss

           image = np.random.randn(3, 8, 3, 32, 32)
           label = np.random.randint(0, 10, (3, 8,))
           for i in range(6):
               timage = megengine.Tensor(image[i % 3])
               tlabel = megengine.Tensor(label[i % 3])
               loss = func(model, optimizer, timage, tlabel)
               print(loss)

       runner()

   worker()

.. _partial_trace:

使用 partial_trace 装饰器
---------------------------

模型训练迭代中存在动态执行逻辑的情况下， 无法将整个计算交由 XLA 执行。
这种情况下可以使用 jit.patrial_trace 装饰器对其中静态的部分进行加速。

被 partial_trace 包装部分的前向/反向会使用 XLA 执行, 其他部分仍由 MegEngine 执行。
代码示例如下：

.. code-block:: python

   :emphasize-lines: 12-27

   @dist.launcher(n_gpus=2, device_type="gpu")
   def worker():
       def runner():
           model = ConvNet()
           model.train()
           dist.bcast_list_(model.tensors())

           cblist = [dist.make_allreduce_cb("mean")]
           gm = autodiff.GradManager().attach(model.parameters(), callbacks=cblist)
           optimizer = AdamW(model.parameters(), lr=0.01)

           model.forward = partial(
               partial_trace(
                   func=type(model).forward,
                   backend="xla",
                   capture_as_const=True,
               ),
               model,
           )
           optimizer._updates = partial(
               partial_trace(
                   func=type(optimizer)._updates,
                   backend="xla",
                   capture_as_const=True,
               ),
               optimizer,
           )

           image = np.random.randn(3, 8, 3, 32, 32)
           label = np.random.randint(0, 10, (3, 8,))
           for i in range(6):
               timage = megengine.Tensor(image[i % 3])
               tlabel = megengine.Tensor(label[i % 3])
               with gm:
                   score = model(timage)
                   loss = F.nn.cross_entropy(score, tlabel)
                   gm.backward(loss)
                   optimizer.step().clear_grad()
                   print(loss)

       runner()

   worker()   

