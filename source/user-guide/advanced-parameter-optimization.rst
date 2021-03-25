.. _advanced-parameter-optimization:

================
参数优化进阶配置
================

假定网络使用如下优化器进行训练：

.. testcode::

    import megengine.optimizer as optim
    optimizer = optim.SGD(
        le_net.parameters(),    # 参数列表，将指定参数与优化器绑定
        lr=0.05,                # 学习速率
    )

这个优化器对所有参数都使用同一学习速率进行优化，本章将介绍如何做到对不同的参数采用不同的学习速率。


不同参数使用不同的学习速率
--------------------------

:class:`~.megengine.optimizer.Optimizer` 支持将网络的参数进行分组，
不同的参数组可以采用不同的学习速率进行训练。 一个参数组由一个字典表示，这个字典中有

*  ``'params': param_list`` ，用来指定参数组包含的参数。此键值对必须有。

*  ``'lr': learning_rate`` ，用来指定此参数组的学习速率。此键值对有时可省略，省略后参数组的学习速率由优化器指定。

所有待优化参数组的字典会组成一个列表作为 :class:`~.megengine.optimizer.Optimizer` 实例化时的第一个参数传入。

为了更好的说明参数组，我们首先使用 :class:`~.megengine.module.Module` 提供的 
:meth:`~.megengine.module.Module.named_parameters` 函数来对网络参数进行分组。
这个函数返回一个包含网络所有参数并且以参数名字为键、参数变量为值的字典：

.. testcode::

    for (name, param) in le_net.named_parameters():
        print(name, param.shape) # 打印参数的名字和对应张量的形状

.. testoutput::

    classifer.bias (10,)
    classifer.weight (10, 84)
    conv1.bias (1, 6, 1, 1)
    conv1.weight (6, 1, 5, 5)
    conv2.bias (1, 16, 1, 1)
    conv2.weight (16, 6, 5, 5)
    fc1.bias (120,)
    fc1.weight (120, 400)
    fc2.bias (84,)
    fc2.weight (84, 120)

根据参数的名字我们可以将 ``LeNet`` 中所有卷积的参数分为一组，所有全连接层的参数分为另一组：

.. testcode::

    conv_param_list = []
    fc_param_list = []
    for (name, param) in le_net.named_parameters():
        # 所有卷积的参数为一组，所有全连接层的参数为另一组
        if 'conv' in name:
            conv_param_list.append(param)
        else:
            fc_param_list.append(param)

分组后即可根据下述代码对不同参数组设置不同的学习速率：

.. testcode::

    import megengine.optimizer as optim

    optimizer = optim.SGD(
        # 参数组列表即param_groups，每个参数组都可以自定义学习速率，也可不自定义，统一使用优化器设置的学习速率
        [
            {'params': conv_param_list},            # 卷积参数所属的参数组，未自定义学习速率
            {'params': fc_param_list, 'lr': 0.01}   # 全连接层参数所属的参数组，自定义学习速率为0.01
        ],
        lr=0.05,  # 参数组例表中未指定学习速率的参数组服从此设置，如所有卷积参数
    )

优化器中设置的参数组列表对应于 :attr:`~.megengine.Optimizer.param_groups` 属性。
我们可以通过其获取不同参数组的学习速率。

.. testcode::

    # 打印每个参数组所含参数的数量和对应的学习速率
    print(len(optimizer.param_groups[0]['params']), optimizer.param_groups[0]['lr'])
    print(len(optimizer.param_groups[1]['params']), optimizer.param_groups[1]['lr'])

.. testoutput::

    4 0.05
    6 0.01

训练中对学习速率的更改
----------------------

MegEngine 也支持在训练过程中对学习速率进行修改，比如部分参数训练到一定程度后就不再需要优化，
此时将对应参数组的学习速率设为零即可。我们修改训练代码进行示例说明。
修改后的训练代码总共训练四个epoch，我们会在第二个epoch结束时将所有全连接层参数的学习速率置零，
并在每个epoch当中输出 ``LeNet`` 中全连接层的部分参数值以显示是否被更新。

.. testcode::

    print("original parameter: {}".format(optimizer.param_groups[1]['params'][0]))
    for epoch in range(4):
        for step, (batch_data, batch_label) in enumerate(dataloader):
            _, loss = train_func(batch_data, batch_label, le_net, gm)
            optimizer.step()  # 根据梯度更新参数值
            optimizer.clear_grad() # 将参数的梯度置零

        # 输出 LeNet 中全连接层的部分参数值
        print("epoch: {}, parameter: {}".format(epoch, optimizer.param_groups[1]['params'][0]))

        if epoch == 1:
            # 将所有全连接层参数的学习速率改为0.0
            optimizer.param_groups[1]['lr'] = 0.0
            print("\nset lr zero\n")

.. testoutput::

    original parameter: Tensor([0. 0. 0. 0. 0. 0. 0. 0. 0. 0.], device=xpux:0)
    epoch: 0, parameter: Tensor([-0.0102  0.0082  0.0062 -0.0093 -0.0018  0.0132 -0.0064  0.0077 -0.0005 -0.007 ], device=xpux:0)
    epoch: 1, parameter: Tensor([-0.0094  0.008   0.0066 -0.0105 -0.0026  0.0141 -0.008   0.0073  0.0015 -0.0071], device=xpux:0)

    set lr zero

    epoch: 2, parameter: Tensor([-0.0094  0.008   0.0066 -0.0105 -0.0026  0.0141 -0.008   0.0073  0.0015 -0.0071], device=xpux:0)
    epoch: 3, parameter: Tensor([-0.0094  0.008   0.0066 -0.0105 -0.0026  0.0141 -0.008   0.0073  0.0015 -0.0071], device=xpux:0)

从输出可以看到在学习速率设为0之前参数值是在不断更新的，但是在设为0之后参数值就不再变化。

同时多数网络在训练当中会不断减小学习速率，如下代码展示了 MegEngine 是如何在训练过程中线性减小学习速率的：

.. testcode::

    total_epochs = 10
    learning_rate = 0.05 # 初始学习速率
    for epoch in range(total_epochs):
        # 设置当前epoch的学习速率
        for param_group in optimizer.param_groups: # param_groups中包含所有需要此优化器更新的参数
            # 学习速率线性递减，每个epoch调整一次
            param_group["lr"] = learning_rate * (1 - float(epoch) / total_epochs)

不同参数使用不同的优化器
------------------------

对于不同的参数，也可以使用不同的优化器对它们分别优化。
对参数的梯度置零（ :meth:`~.megengine.optimizer.Optimizer.clear_grad` ）
和更新（ :meth:`~.megengine.optimizer.Optimizer.step` ）操作，
如果所有优化器都是同时进行的，可以定义一个 ``MultipleOptimizer`` 类。
在初始化时声明多个不同的优化器，在调用置零函数和更新函数时对所有优化器执行对应操作。

.. code-block:: python

    class MultipleOptimizer(object):
        def __init__(*opts):
            self.opts = opts

        def clear_grad(self):
            for opt in self.opts:
                opt.clear_grad()

        def step(self):
            for opt in self.opts:
                opt.step()

假设想用 :class:`~.megengine.optimizer.SGD` 优化所有卷积参数，
用 :class:`~.megengine.optimizer.adam.Adam` 优化所有全连接层参数。
可以按照如下方式定义优化器，不需要改变训练代码就可以达到不同的参数使用不同的优化器优化的效果。

.. code-block:: python

    optimizer = MultipleOptimizer(
        optim.SGD(conv_param_list, lr=0.05), optim.Adam(fc_param_list, lr=0.01)
    )        

如果不同的参数梯度置零和更新不是同时进行的，你只需要定义多个优化器，在不同的时间调用对应的函数即可。

固定部分参数不优化
------------------

除了将不训练的参数分为一组并将学习速率设为零外，
MegEngine 还提供了其他途径来固定参数不进行优化：
仅将需要优化的参数与求导器和优化器绑定即可。
如下代码所示，仅对 ``LeNet`` 中的卷积参数进行优化：

.. testcode::

    import megengine.optimizer as optim
    from megengine.autodiff import GradManager
    
    le_net = LeNet()
    param_list = []
    for (name, param) in le_net.named_parameters():
        if 'conv' in name: # 仅训练LeNet中的卷积参数
            param_list.append(param)

    optimizer = optim.SGD(
        param_list, # 参数
        lr=0.05,    # 学习速率
    )

    gm = GradManager().attach(param_list)

下述代码将上面的设置加入到了具体训练当中，能够更加直观的看到各个参数的梯度差异：

.. testcode::

    learning_rate = 0.05
    total_epochs = 1 # 为了减少输出，本次训练仅训练一个epoch
    for epoch in range(total_epochs):
        # 设置当前epoch的学习速率
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * (1 - float(epoch) / total_epochs)

        total_loss = 0
        for step, (batch_data, batch_label) in enumerate(dataloader):
            batch_data = tensor(batch_data)
            batch_label = tensor(batch_label)
            _, loss = train_func(batch_data, batch_label, le_net, gm)
            optimizer.step()  # 根据梯度更新参数值
            optimizer.clear_grad() # 将参数的梯度置零
            total_loss += loss.numpy().item()

        # 输出每个参数的梯度
        for (name, param) in le_net.named_parameters():
            if param.grad is None:
                print(name, param.grad)
            else:
                print(name, param.grad.sum())

.. testoutput::

    classifier.bias None
    classifier.weight None
    conv1.bias Tensor([-0.0432], device=xpux:0)
    conv1.weight Tensor([0.1256], device=xpux:0)
    conv2.bias Tensor([0.0147], device=xpux:0)
    conv2.weight Tensor([5.0205], device=xpux:0)
    fc1.bias None
    fc1.weight None
    fc2.bias None
    fc2.weight None

从输出可以看到除了卷积参数有梯度外其余参数均没有梯度也就不会更新。
