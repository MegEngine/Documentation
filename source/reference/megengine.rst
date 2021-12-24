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

Serialization
-------------
.. autosummary::
   :toctree: api
   :nosignatures:

   save
   load

.. _device:

Device
------
.. autosummary::
   :toctree: api
   :nosignatures:

   is_cuda_available
   get_device_count
   get_default_device
   set_default_device
   get_mem_status_bytes
   get_cuda_compute_capability
   get_allocated_memory
   get_reserved_memory
   get_max_reserved_memory
   get_max_allocated_memory
   reset_max_memory_stats
   set_prealloc_config
   coalesce_free_memory

.. _logger:

Logger
------
.. autosummary::
   :toctree: api
   :nosignatures:

   enable_debug_log
   get_logger
   set_log_file
   set_log_level

.. _version:

Version
-------
.. py:data:: __version__
   :annotation: （MegEngine 版本号）


