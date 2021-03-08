.. _dump:

==============
导出序列化模型
==============

模型序列化
------------------------------

MegEngine 依赖 trace 来序列化（:meth:`dump `）一个训练好的模型。并且为了把一些参数（比如卷积层的卷积核等）固化下来，需要在 trace 中多指定一项 ``capture_as_const = True``。之后调用 ``dump`` 函数即可把模型序列化到一个文件或者文件对象中。如：

.. code-block::

    from megengine import jit, tensor

    @jit.trace(capture_as_const=True)
    def f(x):
        return exp(x)

    f(tensor(5.0))
    f.dump("test.mge")

``dump`` 函数可接受多个参数，其中最常用的有如下两个。

arg_names
```````````````````````
在序列化的时候统一设置模型输入 Tensor 的名字。由于不同的模型的差异，会导致输入 Tensor 的名字千差万别。
为了减少理解和使用难度，可使用此参数统一设置模型输入为诸如 ``arg_0``, ``arg_1``, ...

optimize_for_inference
```````````````````````
训练出的模型往往在部署时不能发挥最优的性能，而我们提供 ``optimize_for_inference`` 来保证序列化出的模型是经过特定优化的。详细的键值参数可见 :meth:`~.megengine.jit.tracing.trace.dump`。
使用上面的例子，通过指定 `enable_io16xc32` 来设置模型输入输出的 Tensor 的精度为 float16，但是运算的 Tensor 精度为 float32。

.. code-block::

    from megengine.core.tensor import megbrain_graph as G

    f.dump("test.mge", enable_io16xc32=True)

    res = G.load_graph("test.mge")
    computing_input = res.output_vars_list[0].owner.inputs[0]
    assert computing_input.dtype == np.float16

值得注意的是，optimize_for_inference 参数默认是 True，
所以即使不给任何键值优化参数，仍然会做一些基础的优化操作，这会导致序列化出来的模型相较之前的定义有细微的差别。