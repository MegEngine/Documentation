.. _dump:

==========================
导出序列化模型文件（Dump）
==========================

.. note::

   * 序列化操作依赖于 :py:class:`~.jit.trace` 进行的 :ref:`将动态图转为静态图 <trace>` 操作；
   * 与此同时，需要在 :py:class:`~.jit.trace` 中指定 ``capture_as_const = True`` 以将参数固化下来。

考虑到推理部署需求，使用 :py:meth:`~.jit.trace.dump`, 即可将训练好的模型序列化到一个文件或文件对象中：

我们以 `ResNet50 <https://github.com/MegEngine/Models/tree/master/official/vision/classification/resnet>`_
为例子，参考代码片段如下：

.. code-block:: python

   import numpy as np
   import megengine.functional as F
   import megengine.hub
   from megengine import jit, tensor

   if __name__ == "__main__":
       net = megengine.hub.load("megengine/models", "resnet50", pretrained=True)
       net.eval()

       @jit.trace(symbolic=True, capture_as_const=True)
       def fun(data, *, net):
           pred = net(data)
           pred_normalized = F.softmax(pred)
           return pred_normalized

       data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

       fun(data, net=net)
       fun.dump("resnet50.mge", arg_names=["data"])

执行脚本，并完成模型转换后，我们就获得了 MegEngine C++ API 可识别的预训练模型文件 ``resnet50.mge`` .

Dump 常用参数说明
-----------------

使用 :meth:`~.jit.trace.dump` 时，可传入多个参数，其中最常用的有如下两个：

``arg_names``
  在序列化的时候统一设置模型输入 Tensor 的名字。由于不同的模型的差异，会导致输入 Tensor 的名字千差万别。
  为了减少理解和使用难度，可使用此参数统一设置模型输入为诸如 ``arg_0``, ``arg_1``, ...

``optimize_for_inference``
  训练出的模型往往在部署时不能发挥最优的性能，
  而我们提供 ``optimize_for_inference`` 来保证序列化出的模型是经特定优化的。
  详细的键值参数可见下方的 :ref:`optimieze-for-inference-options` . 

.. warning::

   ``optimize_for_inference`` 参数默认是 ``True`` ，
   所以即使不给任何键值优化参数，仍然会做一些基础的优化操作，
   这会导致序列化出来的模型相较之前的定义有细微的差别。

Dump 带有测试数据的模型文件
---------------------------

使用 :meth:`~.jit.trace.dump` 时，设置如下参数：

``input_data``
   这是一个字符串列表，列表中的每个字符串都代表一组测试数据。

   字符串支持三种格式：

   - ``var0:file0;var1:file1...`` 指定每个输入变量对应的文件名。文件可以是能被 opencv 加载的图片，也可以是 numpy.ndarray 的 pickle 文件。如果只有一个输入，输入的变量名可以省略。

   - ``var0:#rand(min, max, shape);var1:#rand...`` 指定每个输入变量的数据随机生成方式，形状是 ``shape``，值域是 ``[min, max)``。例如 ``rand(0, 255)``, ``rand(0, 255, 1, 3, 224, 224)`` 或 ``#rand(0, 255, 1, ...)`` （其中 ``...`` 表示 shape 的剩余部分）。如果形状没有被指定，就会使用网络中输入张量的形状。如果只有一个输入，输入的变量名可以省略。

   - ``@filename`` 指定输入文件名，文件中的每一行都是一个符合上面两种格式的字符串。

更多相关的参数设置，请参考 :meth:`~.jit.trace.dump`.

.. _optimieze-for-inference-options:

推理优化选项表
--------------

``--enable-io16xc32``
  采用 float16 作为算子之间的数据传输类型，使用 float32 作为计算类型。

``--enable-ioc16``
  采用 float16 作为算子之间的数据传输类型以及计算类型。

``--enable-fuse-conv-bias-nonlinearity``
  是否融合 conv+bias+nonlinearity。

``--enalbe-hwcd4``
  采用 hwcd4 数据布局。

``--enable-nchw88``
  采用 nchw88 数据布局。

``--enable-nchw44``
  采用 nchw44 数据布局。

``--enable-nchw44-dot``
  采用 nchw44_dot 数据布局。

``--enable-nchw32``
  采用 nchw32 数据布局。

``--enable-chwn4``
  采用 chwn4 数据布局。

``--enable-fuse-conv-bias-with-z``
  仅在使用 GPU 平台下可用，把 conv，bias (elemwise add)，z(elemwise add) 融合成一个算子。

