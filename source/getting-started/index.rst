.. _getting-started:

========
新手入门
========

.. toctree::
   :maxdepth: 2
   :hidden:

   beginner/index
   more-tutorials

安装
----

天元 MegEngine 可以使用 Python 包管理器 ``pip`` 直接进行安装：

.. code-block:: shell

   pip3 install megengine -f https://megengine.org.cn/whl/mge.html

如果想要安装特定的版本，或需要从源码进行编译？ :ref:`了解更多安装方式<install>` 。

.. note::

   MegEngine 安装包中集成了使用 GPU 运行代码所需的 CUDA 环境，不用区分 CPU 和 GPU 版。
   如果想要运行 GPU 程序，请确保机器本身配有 GPU 硬件设备并安装好驱动。

   如果你想体验在云端 GPU 算力平台进行深度学习开发的感觉，欢迎访问 `MegStudio <https://studio.brainpp.com/>`_ 平台。

一个最简单的使用样例
--------------------

以下代码仅简单展示了使用 MegEngine 进行模型开发的基础流程，并未展示所有特性

我们使用线性分类器来对 MNIST 手写数字进行分类：

.. code-block:: python

   import numpy as np
   import megengine as mge
   import megengine.functional as F
   from megengine.data.dataset import MNIST
   from megengine.data import SequentialSampler, RandomSampler, DataLoader
   from megengine.autodiff import GradManager
   import megengine.optimizer as optim

   MNIST_DATA_PATH = "/data/datasets/MNIST/"

   # 设置超参数
   bs = 64
   lr = 1e-6
   epochs = 5

   # 读取原始数据集
   train_dataset = MNIST(root=MNIST_DATA_PATH, train=True, download=False)
   nums = len(train_dataset)
   num_features = 784   # (28, 28, 1) Flatten -> 784
   num_classes = 10

   # 训练数据加载与预处理
   train_sampler = SequentialSampler(dataset=train_dataset, batch_size=bs)
   train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler)

   # 初始化参数
   W = mge.Parameter(np.zeros((num_features, num_classes)))
   b = mge.Parameter(np.zeros((num_classes,)))

   # 定义模型
   def linear_cls(X):
       return F.matmul(X, W) + b

   # 定义求导器和优化器
   gm = GradManager().attach([W, b])
   optimizer = optim.SGD([W, b], lr=lr)

   # 模型训练
   for epoch in range(epochs):
       total_loss = 0
       for batch_data, batch_label in train_dataloader:
           batch_data = F.flatten(mge.tensor(batch_data), 1)
           batch_label = mge.tensor(batch_label)
           with gm:
               pred = linear_cls(batch_data)
               loss = F.loss.cross_entropy(pred, batch_label)
               gm.backward(loss)
           optimizer.step().clear_grad()
           total_loss +=  loss.item()
       print("epoch = {}, loss = {:.3f}".format(epoch, total_loss / len(train_dataloader)))

* epoch = 0, loss = 0.533
* epoch = 1, loss = 0.362
* epoch = 2, loss = 0.335
* epoch = 3, loss = 0.322
* epoch = 4, loss = 0.313

可以看见 loss 在不断下降，接下来测试该模型的性能：

.. code-block::

   test_dataset = MNIST(root=MNIST_DATA_PATH, train=False, download=False)
   test_sampler = RandomSampler(dataset=test_dataset, batch_size=100)
   test_dataloader = DataLoader(dataset=test_dataset, sampler=test_sampler)

   nums_correct = 0
   for batch_data, batch_label in test_dataloader:
       batch_data = F.flatten(mge.tensor(batch_data), 1)
       batch_label = mge.tensor(batch_label)
       logits = linear_cls(batch_data)
       pred = F.argmax(logits, axis=1)
       nums_correct += (pred == batch_label).sum().item()
   print("Accuracy = {:.3f}".format(nums_correct / len(test_dataloader)))
   
我们只训练了 5 个周期的线性分类器，在测试集的 10,000 张图片中，就有 9,170 张图片被正确地预测了分类。

接下来做什么
------------

如果你感觉上面的流程比较陌生（深度学习新手？不知道 MNIST 数据集？），可以参考我们的 :ref:`beginner` .