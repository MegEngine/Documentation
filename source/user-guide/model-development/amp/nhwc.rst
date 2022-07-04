.. _amp-guide-nhwc:

============================
使用 NHWC 格式进一步提速
============================

.. note::

    该功能目前仍不完善，支持的模型种类有限，欢迎积极反馈遇到的问题~

在 :ref:`amp-guide` 文档中，我们介绍了基础的 AMP 加速功能，这是一种代价低、效果好的训练速度提升方法，在大部分卷积神经网络上都能有可观的作用。本文则会在此基础上介绍一种代价稍高、限制更大，但效果能更好的提速方法。

我们知道目前在训练大部分视觉领域的卷积神经网络时，输入都是 NCHW 格式的， 然而在推理部署时，往往会采用 NHWC、NHWCD4 等等完全不同的输入格式，这是因为硬件和软件算法层面都能从这些格式上获得更多的性能加速。所以我们为什么不能在训练时也用上这些格式呢？

目前已经较为成熟的方案是采用 NHWC 代替 NCHW 成为训练时的内存格式进行计算，一方面需要对模型做的改动比较小，一方面性能的提升效果也足够不错，在 fp16 AMP 的基础上一般能再获得 22% 以上的性能提升 [1]_。

.. [1] https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html

接口介绍
--------

在接口使用上其实非常简单，主要需要做两件事：

* 将模型的 NCHW 输入转换成 NHWC 格式；
* 将模型中可以使用 NHWC 加速的 Module 的参数从 NCHW 转换成 NHWC。

具体来说相比原先的 AMP 用法，需要多修改两条语句：

.. code-block::

    import megengine.functional as F
    import megengine.amp as amp
    from megengine.autodiff import GradManager
    from megengine.optimizer import SGD

    # 将输入数据转成 NHWC
    image = mge.tensor(np.random.normal(size=(1, 3, 224, 224)), dtype="float32", format="nhwc")
    label = mge.tensor(np.zeros(1), dtype="int32")

    # 将模型转为 NHWC，需要在 gm 和 opt 之前
    model = amp.convert_module_format(model)

    gm = GradManager().attach(net.parameters())
    opt = SGD(net.parameters(), lr=0.01)
    scaler = amp.GradScaler()

后面的训练部分和基础的 AMP 用法没有区别：

.. code-block::

    @amp.autocast
    def train_step(image, label):
        with gm:
            logits = net(image)
            loss = F.nn.cross_entropy(logits, label)
            scaler.backward(gm, loss)   # 通过 GradScaler 修改反传行为
        opt.step().clear_grad()
        return loss

    train_step(image, label)

在实际计算时，会根据算子输入 Tensor 的 format 类型做相应的处理，并给输出加上正确的 format，从而使得网络能一步步始终使用 NHWC 格式进行计算。

目前已知的问题主要有：

* 网络中如果存在把 Tensor 变为 numpy 数组再计算的中间过程，会导致 format 信息丢失，进而影响后续的计算过程；
* NHWC 格式在部分网络中可能存在计算卡死的现象，还在修复中。

如果你在使用中有什么疑问或者发现一些具体的错误样例，欢迎提 `issue <https://github.com/MegEngine/MegEngine/issues>`_ ~