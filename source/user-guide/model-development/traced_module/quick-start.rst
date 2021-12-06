.. _quick-start:

=====================
快速上手 TracedModule
=====================

**TracedModule 本质是一个 Module**，其由普通的 Module 转换而来。
本文以 resnet18 为例，讲解如何使用 TracedModule 进行模型开发与部署。

Module 转换为 TracedModule
==========================

本文使用的 resnet18 模型为 MegEngine 预训练的模型，来自 `模型中心 <https://megengine.org.cn/model-hub>`_ 。
我们利用 :py:func:`~.trace_module` 方法将 ``resnet18`` 这个 Module 转换 TracedModule。

.. code-block:: python

    import numpy as np
    import megengine.functional as F
    import megengine.module as M
    import megengine as mge
    import megengine.traced_module as tm

    resnet = mge.hub.load("megengine/models", "resnet18", pretrained=True)

    data = mge.Tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))
    
    traced_resnet = tm.trace_module(resnet, data)

:py:func:`~.trace_module` 方法所返回的 ``traced_resnet`` 是一个 TracedModule ，但其本质是一个 Module。

>>> isinstance(traced_resnet, M.Module)
True

我们可以像普通 Module 一样使用 TracedModule。例如将其作为一个 Module 的子 Module。

.. code-block:: python

    class MyModule(M.Module):
        def __init__(self, backbone: M.Module):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            x = self.backbone(x)
            return x

    new_net = MyModule(traced_resnet)

对于任意一个 Module 我们都可以将其转换为 TracedModule，包括 TracedModule 也可以被再次转换
（再次 ``trace`` 后的模型结构不会发生变换）。

>>> traced_resnet = tm.trace_module(traced_resnet, data)
>>> traced_new_net = tm.trace_module(new_net, data)

TracedModule 序列化
===================

可以直接使用 MegEngine 提供的序列化接口 ``mge.save`` 将 TracedModule 模型序列化到文件中。

>>> mge.save(traced_resnet, "traced_resnet18.pkl")

也可以直接使用 python 内置的 pickle 模块将 TracedModule 序列化到文件中。

>>> pickle.dump(traced_resnet, "traced_resnet18.pkl")

对应的，可以分别使用 ``mge.load`` 或 ``pickle.load`` 将序列化的 TracedModule 再 load 回来。
在脱离模型源码的环境中 load 得到的 TracedModule，依然可以正常被解析与运行。

>>> mge.load("traced_resnet18.pkl")
>>> pickle.load("traced_resnet18.pkl")

TracedModule 图手术
===================

TracedModule 提供了一些方便的图手术接口来修改 TracedModule 的执行逻辑。图手术的接口可以直接阅读 :ref:`tracedmodule-graph-transform-method`，
每一个接口下都提供了如何使用该接口的例子。

我们提供了一些常见的图手术例子在 :ref:`graphsurgeon-example` 中，可以了解如何完成对 TracedMdoule 执行逻辑的的修改。

同时，我们提供了一些内置的图手术实现来优化模型结构，包括：

* FuseConvBn：将 BatchNorm 融合到 Convolution 中
* FuseAddMul：融合连续的常量加法或常量乘法
* BackwardFoldScale：将卷积之后的常量乘法融合到卷积中

使用这些优化的接口为 ``tm.optimize``, 具体用法请参考 :ref:`tracedmodule_graph_optimize`。

TracedModule 模型部署
=====================

使用 MegEngine 进行模型推理部署与普通 Module 部署方法一致，可参考 :ref:`trace <trace>` & :ref:`dump <dump>` 将模型转为 c++ 静态图模型，
同时在 ``dump`` 时可以对 c++ 模型进行 Inference 相关的优化。或参考 《:ref:`megengine-lite`》 进行模型部署。


对于第三方平台部署，我们开发了基于 TracedModule 的转换器 `mgeconvert <https://github.com/MegEngine/mgeconvert>`__ ，
可以方便地将 TracedModule 所描述的模型结构和参数转换至其它推理框架，例如：caffe, tflite 和 onnx 等，
未来也将支持更多第三方推理框架的转换。
