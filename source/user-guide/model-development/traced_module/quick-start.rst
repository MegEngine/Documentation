.. _tracedmodule-quick-start:
.. _quick-start:

=====================
快速上手 TracedModule
=====================

**TracedModule 本质是一个 Module**, 其由普通的 Module 转换而来。
本文以 shufflenet_v2 为例，讲解如何使用 TracedModule 进行模型开发与部署。

Module 转换为 TracedModule
==========================

本文使用的 shufflenet_v2 模型为 MegEngine 预训练的模型，来自 `模型中心 <https://megengine.org.cn/model-hub>`_ 。
我们利用 :py:func:`~.trace_module` 方法将 ``shufflenet`` 这个 Module 转换 TracedModule.

.. code-block:: python

    import numpy as np
    import megengine.functional as F
    import megengine.module as M
    import megengine as mge
    import megengine.traced_module as tm

    shufflenet = mge.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
    shufflenet.eval()

    data = mge.Tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))
    
    traced_shufflenet = tm.trace_module(shufflenet, data)

:py:func:`~.trace_module` 方法所返回的 ``traced_shufflenet`` 是一个 TracedModule, 但其本质是一个 Module.

>>> isinstance(traced_shufflenet, M.Module)
True

我们可以像普通 Module 一样使用 TracedModule, 例如将其作为一个 Module 的子 Module.

.. code-block:: python

    class MyModule(M.Module):
        def __init__(self, backbone: M.Module):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            x = self.backbone(x)
            return x

    new_net = MyModule(traced_shufflenet)

对于任意一个 Module 我们都可以将其转换为 TracedModule, 包括 TracedModule 也可以被再次转换
（再次 ``trace`` 后的模型结构不会发生变化）。

>>> traced_shufflenet = tm.trace_module(traced_shufflenet, data)
>>> traced_new_net = tm.trace_module(new_net, data)

TracedModule 序列化
===================

.. note::

   * 推荐使用 ``.tm`` 作为 TracedModule 序列化文件的后缀名
   * 推荐使用 :func:`megengine.save` 和 :func:`megengine.load` 保存和加载 TracedModule

可以直接使用 MegEngine 提供的序列化接口 :func:`megengine.save` 将 TracedModule 模型序列化到文件中。

>>> mge.save(traced_shufflenet, "traced_shufflenet.tm")

也可以直接使用 python 内置的 pickle 模块将 TracedModule 序列化到文件中。

>>> with open("traced_shufflenet.tm", "wb") as f:
...     pickle.dump(traced_shufflenet, f)

对应的，可以分别使用 :func:`megengine.load` 或 :func:`pickle.load` 将序列化的 TracedModule 再加载回来。
在脱离模型源码的环境中加载得到的 TracedModule, 依然可以被正常解析与运行。

>>> traced_resnet = mge.load("traced_shufflenet.tm")

>>> with open("traced_shufflenet.tm", "rb") as f:
...     traced_resnet = pickle.load(f)

TracedModule 图手术
===================

TracedModule 提供了一些方便的图手术接口来修改 TracedModule 的执行逻辑。
图手术的接口可以直接阅读 :ref:`tracedmodule-graph-transform-method`,
每一个接口下都提供了如何使用该接口的例子。

我们提供了一些常见的图手术例子在 :ref:`graphsurgeon-example` 中，可以了解如何完成对 TracedMdoule 执行逻辑的的修改。

同时，我们提供了一些内置的图手术实现来优化模型结构，包括：

* FuseConvBn：将 BatchNorm 融合到 Convolution 中
* FuseAddMul：融合连续的常量加法或常量乘法
* BackwardFoldScale：将卷积之后的常量乘法融合到卷积中

使用这些优化的接口为 ``tm.optimize``, 具体用法请参考 :ref:`tracedmodule_graph_optimize`.

TracedModule 模型部署
=====================

使用 MegEngine 进行部署
------------------------

**TracedModule 本质是一个 Module**, 使用 MegEngine 进行模型部署与普通 Module 部署方法一致，
可参考 :ref:`trace <trace>` & :ref:`dump <dump>` 将模型转为 c++ 静态图模型。

以 shufflenet_v2 为例，我们可以直接使用上面被转换为 TracedModule 的 shufflenet_v2 模型 ``traced_shufflenet``,
也可以在脱离 shufflenet_v2 源码的环境中直接加载被序列化后并保存的 ``traced_shufflenet.tm``.

>>> traced_shufflenet = mge.load("traced_shufflenet.tm")

然后调用 :py:class:`~.jit.trace` 方法将 ``traced_shufflenet`` 转换为 MegEngine 静态图，
调用 :py:meth:`~.jit.trace.dump` 方法将静态图保存为 c++ 模型，同时在 ``dump`` 时可以开启 Inference 相关的优化。

.. code:: python
    :emphasize-lines: 3, 10, 12

    import megengine.jit as jit

    @jit.trace(symbolic=True, capture_as_const=True)
    def fun(data, *, net):
        pred = net(data)
        return pred

    data = mge.Tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))
    
    fun(data, net=traced_shufflenet)

    fun.dump("shufflenet_v2.mge", arg_names=["data"], optimize_for_inference=True)

将模型序列化为 c++ 模型并保存后，可以参考 :ref:`MegEngine Lite C++ 部署模型快速上手 <lite-quick-start-cpp>` 在 c++ 环境中进行模型部署，
或参考 :ref:`MegEngine Lite Python 部署模型快速上手 <lite-quick-start-python>` 在 python 环境中进行模型部署。

使用第三方平台部署
----------------------

对于第三方平台部署，我们开发了基于 TracedModule 的转换器 `mgeconvert <https://github.com/MegEngine/mgeconvert>`__ ，
可以方便地将 TracedModule 所描述的模型结构和参数转换至其它推理框架，例如：caffe, tflite 和 onnx 等，
未来也将支持更多第三方推理框架的转换。
