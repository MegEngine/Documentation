.. _gpu-memory:

================
显存使用常见问题
================

.. note::

   本指南只适用于 MegEngine Python 接口。在使用 C++ 接口进行推理时，显存使用和控制将更为精细和复杂。

如何正确的观察显存使用情况
--------------------------

MegEngine 使用了显存池机制来加速显存的申请和释放，程序中释放掉的显存将会存储在显存池中，不会被主动的释放给 CUDA driver. 因此使用 ``nvidia-smi`` 命令观察到的显存占用可能大于实际的使用值。

可以使用 :meth:`~megengine.get_mem_status_bytes` 来获取某个计算设备的总显存和空闲显存（此时的空闲显存包含了显存池中未分配的显存），根据两者的差值即可获知当前准确的显存占用量。


如何释放当前占用的显存
----------------------

当一个 Python 对象的生命周期结束时，显存就会被释放，例如：

* 对于 :class:`~.Tensor` 对象，当没有被任何变量、:class:`~.Module` 和 :class:`~.GradManager` 引用时，将会被析构并释放显存
* 可以通过 :code:`del` 对应的 Tensor 或 Module 对象来手动释放引用

.. note::
    
   由于 Python GC 并不保证所有对象在引用计数为 0 时会被立刻释放，因此可能会出现对象在删除后显存却没有立刻释放的情况（尤其是在多个对象循环引用时），
   此时可以通过 :py:func:`gc.collect` 来尝试强制立刻回收。

.. warning::
   
   Python 的变量生存周期与 C++ 不一样，在 for 循环结束时，有可能会遇到 Tensor 对象未释放导致的额外显存占用。
   我们可以通过显式删除变量，来立刻释放掉这部分显存，示例如下：

    .. code-block:: python

        for i in range(4):
            d = tensor(i * 2)

        print(d)    # Still exist!
        del d       # Release tensor
