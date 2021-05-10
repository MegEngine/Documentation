.. _megengine-reference:

========
API 参考
========
当前页面详细介绍了 MegEngine 中包含的模块、对象和方法，描述了它们的功能和作用，以便于检索和浏览。
想要了解如何使用 MegEngine, 请参阅 :ref:`getting-started` 和 :ref:`user-guide` 页面。

Python API
----------

.. py:module:: megengine
.. currentmodule:: megengine

.. toctree::
   :caption: Python API
   :hidden:
   :maxdepth: 1

   data
   tensor
   functional
   module
   autodiff
   optimizer
   jit
   distributed
   quantization
   serialization
   random
   hub
   logger
   device
   version
   utils

.. note::

   * MegEngine 在 GitHub 上的 Python 包源码：:src:`imperative/python/megengine`
   * 为了方便用户调用，在一些 ``__init__.py`` 文件中导入了一些子模块或常用对象和方法。
     如在调用 :py:func:`.functional.add` 时，实际调用的是 ``functional.elemwise.add`` 接口。

Index
-----

* :ref:`genindex`
* :ref:`modindex`
