.. _runtimeopr:

===================
RuntimeOpr 使用说明
===================

RuntimeOpr 指通过 MegEngine 将其它硬件厂商支持的离线模型作为一个算子嵌入到 MegEngine Graph 中。

.. warning::

   包含 RuntimeOpr 的模型无法通过 :py:func:`megengine.save` 保存权重，
   只能通过 :py:meth:`.trace.dump` 直接保存为模型。用法见 :ref:`runtimeopr-dump` 。

目前支持 RuntimeOpr 的类型有 TensorRT、Atlas 和 Cambricon 三种，
包含 RuntimeOpr 的模型需要在对应的硬件平台上才能执行推理任务。
下面以 Atlas 为例展示用法（TensorRT、Cambricon 的接口与之类似）：

模型只包含一个 RuntimeOpr
-------------------------

.. code-block:: python

   import numpy as np
   import megengine as mge
   from megengine.module.external import AtlasRuntimeSubgraph

   with open("AtlasRuntimeOprTest.om", "rb") as f:
     data = f.read()

   m = AtlasRuntimeSubgraph(data)
   inp = mge.tensor(np.ones((4, 3, 16, 16)).astype(np.float32), device="atlas0")

   y = m(inp)

.. note::

   #. 硬件厂商的模型文件需要以字节流形式打开
   #. RuntimeOpr 输入的所属设备应该是该类设备，本例中 inp 的 device 为 “atlas0”

RuntimeOpr 作为模型的一部分
---------------------------

.. code-block:: python

   import megengine as mge
   import megengine.module as M
   import megengine.functional as F

   class Net(M.Module):
    def __init__(self, data):
        super().__init__()
        self.runtimeopr = AtlasRuntimeSubgraph(data)

    def forward(self, x):
        out = F.relu(x)
        # out = out.astype(np.float16)
        out = F.copy(out, "atlas0")
        out = self.runtimeopr(out)[0]
        out = F.copy(out, "cpux")
        out = F.relu(out)
        return out

   m = Net(data)
   inp = Tensor(np.ones(shape=(1, 64, 32, 32)).astype(np.float32), device="cpux")
   y = m(inp)

.. note::

   #. 在 RuntimeOpr 前后必须使用 :py:func:`~.copy` 把 Tensor 从 CPU 拷贝到 Atlas,
      或者从 Atlas 拷贝到 CPU, 不然会因为 CompNode 不符合规范而报错；
   #. 如果需要转变数据类型，请在 CPU 上完成（参考上面的代码）；
   #. 只能从 CPU 拷贝到其他设备或者反之，各类设备之间不能直接拷贝，比如 GPU 到 Atlas.

.. _runtimeopr-dump:

序列化与反序列化
----------------
参考下面的代码：

.. code-block:: python

   import io
   from megengine.jit import trace
   import megengine.utils.comp_graph_tools as cgtools

   def func(inp):
     feature = m(inp)
     return feature

   traced_func = trace(func, symbolic=True, capture_as_const=True)
   y2 = traced_func(inp)
   file = io.BytesIO()
   traced_func.dump(file)
   file.seek(0)
   infer_cg = cgtools.GraphInference(file)
   y3 = list((infer_cg.run(inp.numpy())).values())[0]
   np.testing.assert_almost_equal(y2.numpy(), y3)

