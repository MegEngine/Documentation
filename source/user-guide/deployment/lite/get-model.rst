.. _get-model:

===================================
获得用于 MegEngine Lite 推理的模型
===================================
MegEngine 训练时候使用的动态图进行训练，当模型训练完成之后，需要将动态图转化为静态图，才能在 MegEngine Lite 中进行推理。目前有两种方法可以从
训练的模型转换到推理模型：

* 使用 trace_module 方式：通过 MegEngine 的 :ref:`Traced Module <traced_module-guide>` 将动态图转换为 traced_module IR，在这个 IR 基础上可以进行图手术等，参考 :ref:`graphsurgeon-example`，最后再转化为可以在 MegEngine Lite 上运行的静态图模型。
* 直接 dump 方式：通过使用 MegEngine 的 :class:`~.jit.trace` 和 :meth:`~.trace.dump` 功能将动态图转为静态图。

如下图所示：

.. mermaid::

   graph LR

   training_code[训练代码] ==> |tm.trace_module| tm_file[.tm 文件]
   training_code .-> |dump| mge_file
   tm_file ==> |dump| mge_file[.mge 文件]

   mge_file ==> |load| litepy[Lite Python 运行时]
   mge_file ==> |load| lite[Lite C++ 运行时]

使用 trace_module 方式
----------------------------
参考 :ref:`quick-start` ，下面是对 `模型中心 <https://megengine.org.cn/model-hub>`_ 已经训练好的 resnet18 转换为 trace_module ，然后再 dump 成为
MegEngine Lite 可以加载的静态模型的示例。

.. code-block:: python

   import numpy as np
   import megengine.functional as F
   import megengine.module as M
   import megengine as mge
   import megengine.traced_module as tm
   from megengine import jit, tensor

   # 用户需要将这里的模型替换为自己已经训练好的模型
   resnet = mge.hub.load("megengine/models", "resnet18", pretrained=True)

   data = mge.Tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

   traced_resnet = tm.trace_module(resnet, data)
   # 可以在这里进行基于 trace_module 的图手术，以及模型转换
   traced_resnet.eval()

   @jit.trace(symbolic=True, capture_as_const=True)
   def fun(data, *, net):
      pred = net(data)
      pred_normalized = F.softmax(pred)
      return pred_normalized

   fun(data, net=traced_resnet)
   fun.dump("resnet18.mge", arg_names=["data"])

上面代码完成了如下步骤：

* 首先通过从 MegEngine 的 `模型中心 <https://megengine.org.cn/model-hub>`_ 下载了 ``resnet18`` 的预训练模型， **用户可以用自己预训练的模型代替** 。
* 将 resnet18 转换为 ``trace_module`` 的模型 ``traced_resnet``，用户可以在 ``traced_resnet`` 中做一些图手术以及模型转换，图手术参考 :ref:`graphsurgeon-example`，模型转换参考 `mgeconvert <https://github.com/megengine/mgeconvert>`_ ， **上面的示例没有做任何图手术和模型转换** 。
* 将 ``traced_resnet`` 通过 :class:`~.jit.trace` 和 :meth:`~.trace.dump` 将模型序列化到文件 ``resnet18.mge`` 中。

.. note::

   如果需要 dump 自己的模型而不是 `模型中心 <https://megengine.org.cn/model-hub>`_ 的模型，这时候可以通过 MegEngine 中 :ref:`serialization-guide`
   来加载和序列化已经训练好的模型，然后替换上面的 ``resnet`` 即可。

直接 dump 的方式
-------------------------
直接 dump 过程比上面使用 ``trace_module`` 方式仅仅少了转换为 trace_module 的过程，省掉这个过程，将牺牲掉对模型做图手术和模型转换的能力，参考下面的示例。

.. code-block:: python

   import numpy as np
   import megengine.functional as F
   import megengine.hub
   from megengine import jit, tensor

   if __name__ == "__main__":

      # 这里需要替换为自己训练的模型，或者 trace_module 之后的模型。
       net = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
       net.eval()

       @jit.trace(symbolic=True, capture_as_const=True)
       def fun(data, *, net):
           pred = net(data)
           pred_normalized = F.softmax(pred)
           return pred_normalized

       data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

       fun(data, net=net)
       fun.dump("shufflenet_v2.mge", arg_names=["data"])

上面代码将从 `模型中心 <https://megengine.org.cn/model-hub>`_ 下载 ``shufflenet_v2_x1_0`` 模型并
进行 :class:`~.jit.trace` 和 :meth:`~.trace.dump` 完成从动态图模型到静态图模型装换。

.. note::

   同样如果需要 dump 自己的模型而不是 `模型中心 <https://megengine.org.cn/model-hub>`_ 的模型，这时可以通过 MegEngine 中 :ref:`serialization-guide`
   来加载已经训练好的模型，或者使用 :ref:`Traced Module <traced_module-guide>` 中的方法得到模型，然后替换上面的 ``net`` 即可。