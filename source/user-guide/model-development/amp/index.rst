.. _amp-guide:

===================
自动混合精度（AMP）
===================

混合精度（Mix Precision）训练是指在训练时，对网络不同的部分采用不同的数值精度，对追求速度的算子（比如 conv、matmul）可以用较低精度（比如 float16）从而获得明显的性能提升，而对其它更看重精度的算子（比如 log、softmax）则保留较高的精度（比如 float32）。

得益于 NVIDIA TensorCore 的存在（需要 Volta, Turing, Ampere 架构 GPU），对于 conv、matmul 占比较多的网络，混合精度训练一般能使网络整体的训练速度有较大的提升（2-3X)。

接口介绍
--------

在 MegEngine 中，使用 :class:`~.amp.autocast` 接口可以实现对网络中相关 op 数据类型的自动转换：

.. code-block::

    import numpy as np
    import megengine as mge
    from megengine import amp
    from megengine.hub import load
    net = load("megengine/models", "resnet18", pretrained=False)
    inp = mge.tensor(np.random.normal(size=(1, 3, 224, 224)), dtype="float32")

    with amp.autocast():    # 使用 autocast context 接口
        oup = net(inp)
    print(oup.dtype)

上面样例中是把 :class:`~.amp.autocast` 作为 context manager 使用，也可以使用装饰器的写法：

.. code-block::

    @amp.autocast()
    def train_func(inp):    # 使用 autocast 装饰器
        oup = net(inp)
        return oup

    oup = train_func(inp)
    print(oup.dtype)

或者使用单独的开关：

.. code-block::

    amp.enabled = True
    oup = net(inp)
    amp.enabled = False
    print(oup.dtype)


在开启 autocast 后，网络中间结果的数值类型会变成 float16，其对应的梯度自然也是 float16。而由于 float16 的数值范围比 float32 要小，所以如果遇到特别小的数（比如 loss、gradient），float16 就难以精确表达，这时候一般需要进行梯度放缩（Gradient Scaling）。做法是对网络的 loss 进行放大，从而使反向传播时网络中间结果对应的梯度也得到相同的放大，减少精度的损失，而梯度反传到参数时，仍会是 float32 的类型，在等比缩小之后，并不会影响参数的更新。

在 MegEngine 中使用梯度放缩，可以通过 :class:`~.amp.GradScaler` 接口进行。

.. code-block::

    import megengine.functional as F
    from megengine.autodiff import GradManager
    from megengine.optimizer import SGD

    gm = GradManager().attach(net.parameters())
    opt = SGD(net.parameters(), lr=0.01)
    scaler = amp.GradScaler()           # 使用 GradScaler 进行梯度缩放

    image = mge.tensor(np.random.normal(size=(1, 3, 224, 224)), dtype="float32")
    label = mge.tensor(np.zeros(1), dtype="int32")

    @amp.autocast()
    def train_step(image, label):
        with gm:
            logits = net(image)
            loss = F.nn.cross_entropy(logits, label)
            scaler.backward(gm, loss)   # 通过 GradScaler 修改反传行为
        opt.step().clear_grad()
        return loss

    train_step(image, label)

上面的样例中，通过替换 ``gm.backward(loss)`` 为 ``scaler.backward(gm, loss)``，可以实现对 loss、gradient 的自动放缩，实际上包含三个步骤：

* 修改 :meth:`.GradManager.backward` 的 ``dy`` 参数，使从 loss 反传的梯度都乘以一个常数 ``scale_factor``；
* 调用 :meth:`.GradScaler.unscale` 对 :meth:`.GradManager.attached_tensors` 的梯度进行修改，乘以 ``scale_factor`` 的倒数；
* 调用 :meth:`.GradScaler.update` ，更新内部统计量，以及视情况更新 ``scale_factor`` 。

所以如果需要更加精细的操作，比如累积多个 iter 的梯度，那么可以使用以下等价形式：

.. code-block::

    @amp.autocast()
    def train_step(image, label):
        with gm:
            logits = net(image)
            loss = F.nn.cross_entropy(logits, label)
            gm.backward(loss, dy=mge.tensor(scaler.scale_factor))   # 对应步骤一
        # 这里可以插入对梯度的自定义操作
        scaler.unscale(gm.attached_tensors())                       # 对应步骤二
        scaler.update()                                             # 对应步骤三
        opt.step().clear_grad()
        return loss

    train_step(image, label)

我们可以形象地把上面两种方式分别称为自动挡和手动挡。

通过以上接口，就可以在无需修改模型代码的条件下，只修改训练代码实现混合精度训练，大幅提升网络的训练速度了。