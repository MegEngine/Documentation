.. _sublinear-memory:

==============
亚线性内存优化
==============

使用大 batch size 通常能够提升深度学习模型性能。
然而，我们经常遇到的困境是有限的 GPU 内存资源无法满足大 batch size 模型训练。
为了缓解这一问题， MegEngine 提供了亚线性内存 ( sublinear memory ) 优化技术用于降低网络训练的内存占用量。
该技术基于 `Gradient Checkpointing <https://arxiv.org/abs/1604.06174>`_ 算法，
通过事先搜索最优的计算图节点作为前向传播和反向传播检查点（ checkpoints ），
省去其它中间结果存储，大幅节约了内（显）存使用。

用户在编译静态图时使用 :class:`~jit.sublinear_memory_config.SublinearMemoryConfig` 
设置 :class:`~jit.tracing.trace` 的参数 ``sublinear_memory_config`` ，就可以打开亚线性内存优化。

.. testcode::

    from megengine.jit import trace, SublinearMemoryConfig

    config = SublinearMemoryConfig()

    @trace(symbolic=True, sublinear_memory_config=config)
    def train_func(data, label, * , net, optimizer, gm):
         ...


使用亚线性内存在编译计算图和训练模型时有少量的额外时间开销，但是可以大幅减少显存的开销。
下面我们以 ResNet50 为例，说明如何使用亚线性内存优化技术，突破显存瓶颈来训练更大 batch size 的模型。

.. testcode::

    import os
    from multiprocessing import Process

    def train_resnet_demo(batch_size, enable_sublinear, genetic_nr_iter=0):
        import megengine as mge
        import megengine.functional as F
        import megengine.hub as hub
        import megengine.optimizer as optim
        from megengine import tensor
        from megengine.jit import trace, SublinearMemoryConfig
        from megengine.autodiff import GradManager
        import numpy as np

        print(
            "Run with batch_size={}, enable_sublinear={}, genetic_nr_iter={}".format(
                batch_size, enable_sublinear, genetic_nr_iter
            )
        )
        # 使用GPU运行这个例子
        assert mge.is_cuda_available(), "Please run with GPU"
        import resnet as models
        resnet = models.__dict__['resnet50'](pretrained=False)

        optimizer = optim.SGD(resnet.parameters(), lr=0.1,)
        gm = GradManager().attach(resnet.parameters())
        config = None
        if enable_sublinear:
            config = SublinearMemoryConfig(genetic_nr_iter=genetic_nr_iter)

        @trace(symbolic=True, sublinear_memory_config=config)
        def train_func(data, label, * , net, gm):
            with gm:
                pred = net(data)
                loss = F.loss.cross_entropy(pred, label)
                gm.backward(loss)

        resnet.train()
        for i in range(10):
            batch_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            batch_label = np.random.randint(1000, size=(batch_size,)).astype(np.int32)
            train_func(tensor(batch_data), tensor(batch_label), net=resnet, gm=gm)
            optimizer.step()
            optimizer.clear_grad()

    # 以下示例结果在 2080Ti GPU 运行得到，显存容量为 11 GB

    # 不使用亚线性内存优化，允许的 batch_size 最大为 100 左右
    p = Process(target=train_resnet_demo, args=(100, False))
    p.start()
    p.join()
    
    # 报错显存不足
    p = Process(target=train_resnet_demo, args=(200, False))
    p.start()
    p.join()

    # 使用亚线性内存优化，允许的 batch_size 最大为 200 左右
    p = Process(target=train_resnet_demo, args=(200, True, 20))
    p.start()
    p.join()
