.. _module-guide:

========================
使用 Module 定义模型结构
========================
.. toctree::
   :hidden:
   :maxdepth: 1 

   design


神经网络模型由对输入数据执行操作的各种层（Layer），或者说模块（Module）组成。

.. figure:: ../../../_static/images/Simplified-illustration-of-the-AlexNet-architecture.pbm
   :align: center

上图为经典的 AlexNet 模型结构
（ `图片来源 <https://www.researchgate.net/figure/Simplified-illustration-of-the-AlexNet-architecture_fig2_329790469>`_ ），
其中有经典的卷积层 ``conv`` 以及全连接层 ``fc`` 模块...

在 MegEngine 的 :mod:`megengine.module` 命名空间中提供了对这种结构的抽象 ——

* 该命名空间中实现了常见的神经网络模块接口如 :py:class:`~.module.Conv2d` ，方便用户快速搭建模型结构；
* 其中所有的模块都是 :class:`~.module.Module` 的子类， 可参考 :ref:`module-design` ；
* 另外还提供了 :class:`~.module.Sequential` 顺序容器，在定义复杂结构时会很有帮助。

.. warning::

   MegEngine 中首字母大写形式的 ``Module`` 指的是在设计模型结构时频繁被我们使用的基类，
   需要把它和 Python 中小写 ``module`` 的概念进行区分，后者指的是可被导入的文件。
   而语句 ``import megengine.module as M`` 实际上是导入名为 ``module.py`` 这个文件模块（通常缩写为 ``M`` ）。

.. seealso::

   这一章节主要介绍默认使用的 Float32 类型的 ``Module`` 以及参数初始化 ``init`` 模块。
   关于量化模型中所用到的 ``QAT Module`` 和 ``Quantized Module``, 将在 :ref:`quantization-guide` 章节中介绍。

基本用法示例
------------

下面这段代码展现了如何借助 ``Module`` 中的基本组件快速地设计一个卷积神经网络结构：

* 所有网络结构都源自基类 ``M.Module``. 
  在构造函数中，一定要先调用 ``super().__init__()``.
* 在构造函数中，声明要使用的所有层/模块；
* 在 ``forward`` 函数中，定义模型将如何运行，从输入到输出。

.. code-block:: python

   import megengine.functional as F
   import megengine.module as M

   class ConvNet(M.Module):
        def __init__(self):
            # this is the place where you instantiate all your modules
            # you can later access them using the same names you've given them in
            # here
            super().__init__()
            self.conv1 = M.Conv2d(1, 10, 5)
            self.pool1 = M.MaxPool2d(2, 2)
            self.conv2 = M.Conv2d(10, 20, 5)
            self.pool2 = M.MaxPool2d(2, 2)
            self.fc1 = M.Linear(320, 50)
            self.fc2 = M.Linear(50, 10)
   
        # it's the forward function that defines the network structure
        # we're accepting only a single input in here, but if you want,
        # feel free to use more
        def forward(self, input):
            x = self.pool1(F.relu(self.conv1(input)))
            x = self.pool2(F.relu(self.conv2(x))) 
            x = F.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            return x

注意以下几点： 

* ``ConvNet`` 本身也是一个模块 ``Module``, 与 ``Conv2d``, ``Linear`` 一样，这意味着它可以作为其它模块的子结构。
  ``Module`` 之间这种灵活的嵌套机制允许用户使用比较简单的方式设计出非常复杂的模型结构。
* 在定义模型的过程中，可以使用任意的 Python 代码来组织模型结构，
  条件和循环控制流语句是完全合法的，能够被自动微分机制很好地处理。
  你甚至可以在前传过程中造一个循环，里面重复使用相同的模块。

让我们创建一个实例试试看：

>>> net = ConvNet()
>>> net
ConvNet(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(10, 20, kernel_size=(5, 5))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
)

.. note::

   所有的 ``Module`` 只支持小批量样本作为输入，而不是单个样本。

   例如，:py:class:`~.module.Conv2d` 输入为 ``nSamples x nChannels x Height x Width`` 的 4 维 Tensor.

   如果你有单个的样本，则需要使用 :py:func:`~.expand_dims` 来增加一个纬度。

我们创建一个包含单个样本的小批量数据（即 ``batch_size=1`` ），并发送给 ``ConvNet``:

>>> input = megengine.Tensor(np.random.randn(1, 1, 28, 28))
>>> out = net(input)
>>> out.shape
(1, 10)

``ConvNet`` 的输出是一个 Tensor, 我们可以用它和目标标签来计算损失，再利用自动求导来完成反向传播过程。
不过默认情况下所有的 Tensor 是不需要被求导的，因此在此之前我们需要有一个梯度管理器来绑定 ``Module`` 参数，
并在前向计算的过程中记录梯度信息，想要了解这个流程，请参考 :ref:`autodiff-guide` 。

更多使用情景
------------

.. seealso::

   :py:class:`~.module.Module` 接口中提供了许多有用的属性和方法，可以方便在不同的情景下使用，比如：

   * 使用 ``.parameters()`` 可以方便地获取参数的迭代器，可被用来追踪梯度，方便进行自动求导；
   * 每个 ``Module`` 有自己的名字 ``name``, 通过 ``.named_module()`` 可以获取到名字和对应的 ``Module``;
   * 使用 ``.state_dict()`` 和 ``.load_state_dict()`` 获取和加载状态信息...

   有关于更多内容的介绍，请参考 :ref:`module-design` 等页面。

