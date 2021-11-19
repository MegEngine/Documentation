.. _distributed-guide:

==================================
分布式训练（Distributed Training）
==================================

本章我们将介绍如何在 MegEngine 中高效地利用多 GPU 进行分布式训练。
分布式训练是指同时利用一台或者多台机器上的 GPU 进行并行计算。
在深度学习领域，最常见的并行计算方式是在数据层面进行的，
即每个 GPU 各自负责一部分数据，并需要跑通整个训练和推理流程。
这种方式叫做 **数据并行** 。

目前 MegEngine 开放的接口支持单机多卡和多机多卡的数据并行方式。


单机多卡
--------

单机多卡是最为常用的方式，比如单机四卡、单机八卡，足以支持我们完成大部分模型的训练。

本节我们按照以下顺序进行介绍：

#. 如何启动一个单机多卡的训练
#. 如何在多进程环境中将模型保存与加载

如何启动一个单机多卡的训练
~~~~~~~~~~~~~~~~~~~~~~~~~~

我们提供了一个单机多卡的启动器。代码示例：

.. code-block::

    import numpy as np
    import megengine as mge
    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim
    from megengine.data.dataset.vision import MNIST
    from megengine.data.dataloader import DataLoader
    from megengine.data.sampler import SequentialSampler
    from megengine import functional as F
    from megengine import module as M

    # pre download MNIST data
    MNIST()

    @dist.launcher
    def main():
        rank = dist.get_rank()

        # 设置超参数
        bs = 100
        lr = 1e-6
        epochs = 5

        num_features = 784   # (28, 28, 1) Flatten -> 784
        num_classes = 10

        # 定义单层线性分类网络
        class Linear(M.Module):
            def __init__(self):
                # 初始化参数
                self.w = mge.Parameter(np.zeros((num_features, num_classes)))
                self.b = mge.Parameter(np.zeros((num_classes,)))

            def forward(self, data):
                data = f.flatten(data, 1)
                return F.matmul(data, self.w) + self.b

        # 初始化模型
        linear_cls = Linear()

        # 同步模型参数，默认全局同步，可以给bcast_list_加上group参数在指定group之间同步
        dist.bcast_list_(linear_cls.tensors())

        gm = ad.GradManager()
        gm.attach(linear_cls.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
        opt = optim.SGD(linear_cls.parameters(), lr=lr)

        data = MNIST()
        sampler = SequentialSampler(data, batch_size=bs)
        data_loader = DataLoader(data, sampler=sampler)

        for epoch in range(epochs):
            total_loss = 0
            for data, label in data_loader:
                data = mge.tensor(data)
                label = mge.tensor(label)
                with gm:
                    pred = linear_cls(data)
                    loss = F.nn.cross_entropy(pred, label)
                    gm.backward(loss)
                opt.step().clear_grad()
                loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
                total_loss += loss.item()
            if rank == 0:
                print("epoch = {}, loss = {:.3f}".format(epoch, total_loss / len(data_loader)))

    main()

    # 期望结果
    # epoch = 0, loss = 0.618
    # epoch = 1, loss = 0.392
    # epoch = 2, loss = 0.358
    # epoch = 3, loss = 0.341
    # epoch = 4, loss = 0.330


和单卡训练相比，单机多卡的训练代码只有几行代码的不同

* @dist.launcher
* dist.bcast_list_(linear_cls.tensors())
* gm.attach(linear_cls.parameters(), callbacks=[dist.make_allreduce_cb("sum")])

下面我会逐一解释这几句话分别有什么含义

.. code-block::

    @dist.launcher

:class:`~.distributed.launcher` 将一个 function 包装成一个多进程运行的 function (默认根据机器上的 device 数量开启多进程)，
每个进程会在最开始根据 rank 设定默认 deivce, 假如是一台 8 卡机器，那么就会开启 8 个进程，rank 分别为 0 到 8 ，device 为 gpu0 到 gpu7.

.. code-block::

    dist.bcast_list_(linear_cls.tensors())

:func:`~.distributed.bcast_list_` 用于同步各个进程之间的参数，默认在全局范围（所有计算设备）同步，可以设置group参数在特定的group之间同步

.. warning::

    注意，这里使用的API是 :func:`module.Module.tensors`而不是 :func:`module.Module.parameters`，这是因为不仅参数需要同步，
    有些时候模型里还会存在一些统计量，比如 :class:`~module.BatchNorm2d` 里的均值和方差

.. code-block::

    gm.attach(linear_cls.parameters(), callbacks=[dist.make_allreduce_cb("sum")])

在数据并行的情况下，由于每张卡只负责一部分数据，所以求导之后只会有部分导数，
在GradManager中注册对于梯度的回调函数，在对应参数的导数求完之后，
做一个 :func:`~.distributed.all_reduce_sum` 操作进行全局求和，这样同步各个计算设备的导数来保证参数更新的一致性

.. note::

    在 :class:`~.data.dataloader.DataLoader` 内部对多机训练有特殊支持，会自动给每个进程分配不重叠的数据进行训练，所以在数据供给方面没有做特殊处理，
    如果没有使用 :class:`~.data.dataloader.DataLoader` ，则需要自己手动给不同 rank 的设备分配不重叠的数据进行训练
    就像下面这样

    .. code-block::

        mnist_datasets = MNIST() # 下载并读取 MNIST 数据集

        size = ceil(len(mnist_datasets) / num_devices) # 将所有数据划分为 num_devices 份
        l = size * rank # 得到本进程负责的数据段的起始索引
        r = min(size * (rank + 1), len(mnist_datasets)) # 得到本进程负责的数据段的终点索引
        data, label = mnist_datasets[l:r] # 得到本进程的数据和标签
        data = np.concatenate([*data]).reshape(r-l, 28, 28, 1) # data 的数据类型为 list of nparray，需要拼接起来作为模型的输入

模型保存与加载
~~~~~~~~~~~~~~

在 MegEngine 中，依赖于上面提到的状态同步机制，我们保持了各个进程状态的一致，
因此可以很容易地实现模型的保存和加载。

对于加载，我们只要在主进程（rank 0 进程）中加载模型参数，
然后调用 :func:`~.distributed.bcast_list_` 对各个进程的参数进行同步，就保持了各个进程的状态一致。

对于保存，由于我们在梯度计算中插入了 callback 函数对各个进程的梯度进行累加，
所以我们进行参数更新后的参数还是一致的，可以直接保存。

可以参考以下示例代码实现：

.. code-block::

    # 加载模型参数
    if rank == 0:
        net.load_state_dict(checkpoint['net'])
    dist.bcast_list_(net.tensors())
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    gm = GradManager().attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum")])

    # 训练
    for epoch in range(epochs):
        for data, label in data_loader:
            data = mge.tensor(data)
            label = mge.tensor(label)
            with gm:
                pred = net(data)
                loss = F.nn.cross_entropy(pred, label)
                gm.backward(loss)
            opt.step().clear_grad()

    # 保存模型参数
    if rank == 0:
        checkpoint = {
            'net': net.state_dict(),
            'acc': best_acc,
        }
        mge.save(checkpoint, path)

.. _dist_dataloader:

多机多卡
--------

在 MegEngine 中，我们能很方便地将上面单机多卡的代码修改为多机多卡，
只需修改传给 :class:`~.megengine.distributed.launcher` 的参数就可以进行多机多卡训练，其他部分和单机多卡一样。

.. code-block::

    @dist.launcher(world_size=world_size, 
                   n_gpus=n_gpus, 
                   rank_start=rank_start, 
                   master_ip=master_ip, 
                   port=port)

参数含义

.. list-table::
    :widths: 10 10 35
    :header-rows: 1

    * - 参数名
      - 数据类型
      - 实际含义
    * - world_size
      - int
      - 训练的用到的总卡数
    * - n_gpus
      - int
      - 运行时这台物理机的卡数
    * - rank_start
      - int
      - 这台机器的 rank 起始值
    * - master_ip
      - str
      - rank 0 所在机器的 IP 地址
    * - port
      - int
      - 分布式训练 master server 使用的端口号

流水线并行
----------

在 MegEngine 中，也支持流水线的方式来做训练。

最简单的流水线并行就是把一个模型拆分成上下两个部分来做，在 MegEngine 中可以简单的实现。

下面是一个简单的例子来展示怎么写一个流水线的训练：

.. code-block::

    import megengine as mge
    import numpy as np
    import megengine.module as M
    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim

    @dist.launcher(n_gpus=2)
    def main():

        rank = dist.get_rank()
        # client 用于各个 rank 之间互相通信
        client = dist.get_client()
        if rank == 0:
            layer1 = M.Linear(1, 1) # 模型上半部分

            x = mge.tensor(np.random.randn(1))
            gm = ad.GradManager()
            opt = optim.SGD(layer1.parameters(), lr=1e-3)
            gm.attach(layer1.parameters())

            with gm:
                feat = layer1(x)
                dist.functional.remote_send(feat, dest_rank=1)
                gm.backward([])
                print("layer1 grad:", layer1.weight.grad)
                opt.step().clear_grad()
        else:
            layer2 = M.Linear(1, 1) # 模型下半部分

            gm = ad.GradManager()
            opt = optim.SGD(layer2.parameters(), lr=1e-3)
            gm.attach(layer2.parameters())

            with gm:
                feat = dist.functional.remote_recv(src_rank=0)
                loss = layer2(feat)
                gm.backward(loss)
                print("layer2 grad:", layer2.weight.grad)
                opt.step().clear_grad()

    main()

    # 期望输出
    # layer2 grad: Tensor([[-2.4756]], device=gpu1:0)
    # layer1 grad: Tensor([[-0.7784]], device=gpu0:0)

常见问题
--------

Q：为什么在多机多卡训练开始前还正常，进入多卡训练之后就报错 ``cuda init error`` ?

A：请确保在进入多机多卡训练之前主进程没有进行 cuda 相关操作，cuda 在已经初始化的状态下进行 fork 操作会导致 fork 的进程中 cuda 不可用，
参考 `这里 <https://stackoverflow.com/questions/22950047/cuda-initialization-error-after-fork>`_ . 建议用 numpy 数组作为输入输出来使用 launcher 包装的函数。

Q：为什么我自己用 :py:mod:`multiprocessing` 写多机多卡训练总是卡住？

A：可以在函数结束前调用 :func:`~.distributed.group_barrier` 来避免卡死的情况

   * 在 MegEngine 中，为了保证性能，会异步执行相应的 cuda kernel，所以当 python 代码执行完毕时，相应的 kernel 执行还没有结束。
   * 为了保证 kernel 全部执行完毕，MegEngine 初始化时在 :py:mod:`atexit` 里注册了全局的同步，但是 multiprocess 默认的 fork 模式在进程退出的时候，不会执行 :py:mod:`atexit` 注册的函数，导致 kernel 没有执行完。
   * 如果有进程间需要通信的算子，而又有几个进程提前退出，那么剩下的进程就会一直等待其他进程导致卡死（如果你某个进程比如 rank0 需要取参数的值）。

