.. _dump:

==================
导出模型序列化文件
==================

MegEngine 依赖 :class:`~.jit.trace` 来序列化（:meth:`~.jit.trace.dump` ）一个训练好的模型。
并且为了把一些参数（比如卷积层的卷积核等）固化下来，
需要在 :class:`~.jit.trace` 中多指定一项 ``capture_as_const = True`` .
之后调用 :meth:`~.jit.trace.dump` 方法即可把模型序列化到一个文件或者文件对象中。如：

.. code-block::

    from megengine import jit, tensor

    @jit.trace(capture_as_const=True)
    def f(x):
        return exp(x)

    f(tensor(5.0))
    f.dump("test.mge")


常见参数配置方法
----------------

:meth:`~.jit.trace.dump` 函数可接受多个参数，其中最常用的有如下两个：

``arg_names``
  在序列化的时候统一设置模型输入 Tensor 的名字。由于不同的模型的差异，会导致输入 Tensor 的名字千差万别。
  为了减少理解和使用难度，可使用此参数统一设置模型输入为诸如 ``arg_0``, ``arg_1``, ...

``optimize_for_inference``
  训练出的模型往往在部署时不能发挥最优的性能，
  而我们提供 ``optimize_for_inference`` 来保证序列化出的模型是经特定优化的。
  详细的键值参数可见 :ref:`optimieze-for-inference-options` . 

使用上面的例子，通过指定 ``enable_io16xc32`` 来设置模型输入输出的 Tensor 的精度为 float16，但是运算的 Tensor 精度为 float32 .

.. code-block::

    from megengine.core.tensor import megbrain_graph as G

    f.dump("test.mge", enable_io16xc32=True)

    res = G.load_graph("test.mge")
    computing_input = res.output_vars_list[0].owner.inputs[0]
    assert computing_input.dtype == np.float16

值得注意的是，``optimize_for_inference`` 参数默认是 ``True`` ，
所以即使不给任何键值优化参数，仍然会做一些基础的优化操作，这会导致序列化出来的模型相较之前的定义有细微的差别。

为导出模型添加测试用例
----------------------

在上一步，我们已经得到了序列化模型，假定为 ``model`` .
在 :src:`/sdk/load-and-run/` 中，提供了 ``dump_with_testcase_mge.py`` 脚本，
可进一步输出带测试用例的模型，其基本使用方法如下：

.. code-block:: python

   python3 dump_with_testcase_mge.py model -d input_description -o model_with_testcases

其可用参数如下：

``input``
  **必须参数** ，执行需要添加输入的MegEngine模型文件地址

``-d --data``
  **必须参数** ，指定模型的输入数据，指定方法为：``<input0 name>:<data0>;<input1 name>:<data1>...`` 
  当模型只有一个输入，则可以省略 input 的名字。数据支持以下三种类型——

  #. 使用随机数据，以 "#rand" 开头：

     - 仅指定输入数据的最大最小值，其中 shape 由输入模型推出：--data #rand(0,255) 
     - 指定输入数据的最大最小值和 batchsize，其中 shape 由输入模型推出
       （注意省略号不可省略）：–data #rand(0,255,1,...)
     - 指定输入数据的全部维度：–data #rand(0,255,1,3,224,224)

  #. 使用图片或者 ``npy`` 文件：

     - 使用图片：--data image.png
     - 使用 npy：--data image.npy

  #. 使用包含多条数据的文本文件，以 "@" 开头，文件中的每一行都符合上面两种形式：--data image.txt

     image.txt里面的内容可能是这样的：

     .. code-block:: none

        var0:image0.png;va1:image1.npy
        var0:#rand(0,255);var1:image2.png

``-o --output``
  **必需参数** ，指定输出模型地址

``--repeat``
  默认值为 1，指定 -d 传递的输入数据会重复多少份，常用于性能测试。

``--silent``
  默认为 false，在启用推理正确性检查的时候，是否输出更加简洁的检查信息。比如说展示误差最大值。

``--optimize-for-inference``
  默认为 false，是否开启计算图优化，经过优化后的图结构可能会发生改变，但是可以获得更好地推理性能，
  详见 :ref:`optimieze-for-inference-options` 。

``--no-assert``
  默认为 false，是否禁用推理正确性检查，常用于性能测试。
  assert 比较的对象为：输入模型 + 输入数据的推理结果 VS 输出模型（此时数据已纳入模型中）的推理结果。

``--maxerr``
  默认为 1e-4，在开启推理正确性检查时允许的最大误差。

``--resize-input``
  默认为 false，是否采用 cv2 库把输入图片的尺寸 resize 到模型要求的输入尺寸。

``--input-transform``
  可选参数，有用户指定的一行 python 代码，用于操作输入数据。比如 ``data/np.std(data)`` .

``--discard-var-name``
  默认为 false，是否丢弃输入模型的变量 (varnode) 和参数 (param) 的名字。

``--output-strip-info``
  默认为 false，是否保存模型的输出信息到 JSON 文件，默认路径为输出模型名 + ".json" .
  文件中包含模型 hash 码，所有输出的 opr 类型和计算数据类型。

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

