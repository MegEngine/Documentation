.. py:module:: megengine
.. currentmodule:: megengine

=========
megengine
=========

.. code-block:: python3

   import megengine as mge  # NOT as torch - Even if we look like twins.

.. note::

   ``MGE`` 或 ``mge`` 是 ``MegEngine`` 的官方缩写，我们建议相关衍生库将其作为前缀。

.. warning::

   不要尝试 [ ``import megengine as torch`` ] 这是行不通的！ ( /ω＼)

   * MegEngine 的部分 API 设计借鉴了 PyTorch_, 我们相信这对开发者而言会更加友好；
   * 但二者的底层设计完全不同，MegEngine is just MegEngine.

.. _PyTorch: https://pytorch.org/

.. _tensor:

Tensor
------

.. code-block:: python

   from megengine import Tensor
   from megengine import tensor  # tensor is an alias of Tensor

MegEngine 中提供了一种名为 “张量” （:class:`Tensor` ）的数据结构，
区别于物理学中的定义，其概念与 NumPy_ 中的 :class:`~numpy.ndarray` 更加相似，
即张量是一类多维数组，其中每个元素的数据类型和空间大小一致，而数据值可以不同。

.. _Numpy: https://numpy.org

.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Tensor
   Parameter

上面 Parameter 可以被看作是一种特殊的 Tensor, 通常用来表示神经网络中的参数。

想要了解更多，请参考 :ref:`tensor-guide` 。

.. _core:

Core
----

在 :mod:`megengine.core` 子包中实现了 MegEngine 的核心功能，包括 Tensor 和 Operators 组成的计算图，自动求导机制等等。
MegEngine 用户在日常使用中无需直接调用它，因为里面的功能实现已经被其它面向用户的常用子包如进行了封装。
但出于方便 MegEngine 开发者检索的目的，我们也将 ``core`` 中的 API 列举在此处。

.. toctree::
   :hidden:
   
   core

.. warning::

   我们不承诺 core 模块中 API 的兼容性和稳定性。

.. note::

   任何 MegEngine 的用户都可以尝试成为我们的开发人员，就好像你理解了 Tensor 和 Core 之后，
   便可以尝试去理解 NVIDIA 的 TensorCore_ 了。冷笑话 +1 (￣▽￣)" 

   .. _TensorCore: https://www.nvidia.cn/data-center/tensor-cores/

.. note::

   你可能在 MegEngine 源代码中经常会看到诸如 MegBrain, MGB 等字样。
   不用担心，MegBrain 是 MegEngine 的内部代号，二者某种程度上是等价的。

模型保存与加载
--------------
.. autosummary::
   :toctree: api
   :nosignatures:

   save
   load

.. _device:

设备相关
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   is_cuda_available
   get_device_count
   get_default_device
   set_default_device
   get_mem_status_bytes
   get_cuda_compute_capability
   set_prealloc_config

.. _logger:

日志相关
--------
.. autosummary::
   :toctree: api
   :nosignatures:

   enable_debug_log
   get_logger
   set_log_file
   set_log_level

.. _version:

查询版本信息
------------
.. py:data:: __version__
   :annotation: （MegEngine 版本号）


