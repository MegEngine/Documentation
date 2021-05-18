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

Tensor
------

.. py:module:: megengine.tensor
.. currentmodule:: megengine

.. code-block:: python

   from megengine import Tensor
   from megengine import tensor  # tensor is an alias of Tensor

MegEngine 中提供了一种名为 “张量” （:class:`Tensor` ）的数据结构，
区别于物理学中的定义，其概念与 NumPy_ 中的 :class:`~numpy.ndarray` 更加相似，
即张量是一类多维数组，其中每个元素的数据类型和空间大小一致，而数据值可以不同。

Tensor 举例：三阶魔方
~~~~~~~~~~~~~~~~~~~~~

.. image:: ../_static/images/cube.svg
   :align: center
   :height: 128px

我们可以借助上面这张魔方（ `图片来源 <https://commons.wikimedia.org/wiki/File:Rubiks_cube.jpg>`_ ）来直观地理解 Tensor:

* 首先，我们假设这个魔方是“实心同质”的，是一个存在于现实世界中的 Tensor;
* 这个 Tensor 里面的每个元素的类型（:attr:`~Tensor.dtype` ）都是一致的（方方正正的形状、外加一样的做工）；
* 而且这是一个维度（:attr:`~Tensor.ndim` ）为 :math:`3` 的结构，形状（:attr:`~Tensor.shape` ）为 :math:`(3, 3, 3)` ; 
* 对应地，该 Tensor 的总元素个数（:attr:`~Tensor.size` ）是 :math:`3*3*3=27`.

如果你将每种颜色代表一个值，而每个魔方块的值可以用其具有的颜色值之和来表示（中间块只好为零了），
那么不同的魔方块就具有了各自的取值，就好像 Tensor 中的每个元素可以有自己的取值一样。
事实上，除了魔方以外，还有很多东西可以抽象成 Tensor 数据结构，意味着 MegEngine 也许能派上用场。

尽管 MegEngine 不是用来教你 `解魔方 <https://rubiks-cube-solver.com/>`_ 的... 😆 
但尝试做一下 :ref:`Tensor 计算 <general-tensor-operations>` 吧，它比魔方还要更加神奇。

.. note::

   与 NumPy 的区别之处在于，MegEngine 还支持利用 GPU 设备进行更加高效的计算。
   当 GPU 和 CPU 设备都可用时，MegEngine 将优先使用 GPU 作为默认计算设备，无需用户进行手动设定。
   
   * 如果有查看/改变默认计算设备的需求，请参考 :mod:`megengine.device` 中提供的接口。
   * 通过 :meth:`.Tensor.to` 和 :func:`.functional.copy` 可将 Tensor 拷贝到指定设备。

.. _Numpy: https://numpy.org
 
.. autosummary::
   :toctree: api
   :nosignatures:
   :template: autosummary/api-class.rst

   Tensor
   Parameter

Tensor 数据类型
~~~~~~~~~~~~~~~

MegEngine 中使用 :class:`numpy.dtype` 来表示数据类型，目前支持的数据类型如下：

========================  ======================================================
数据类型                  numpy.dtype              
========================  ======================================================
32-bit floating-point     :any:`numpy.float32` a alias of :class:`numpy.single`
16-bit floating-point     :any:`numpy.float16` a alias of :class:`numpy.half`
8-bit integer (unsigned)  :any:`numpy.uint8`
8-bit integer (signed)    :any:`numpy.int8`
16-bit integer (signed)   :any:`numpy.int16`
32-bit integer (signed)   :any:`numpy.int32`
Boolean                   :any:`numpy.bool8` a alias of :class:`numpy.bool_`
========================  ======================================================

要获取指定数据类型的 Tensor, 可以使用 :meth:`~Tensor.astype` 方法进行转化。

.. note::

   单精度浮点 ``float32`` 是 MegEngine 中默认的（也是最通用的）Tensor 数据类型。

.. note::

   我们会在 :mod:`megengine.quantization` 模块中提到对量化数据类型的支持。

.. warning::

   并不是所有的已有算子都支持上述任意数据类型的计算，还有一些情况尚未实现。

Tensor 创建与处理
~~~~~~~~~~~~~~~~~

创建 Tensor 的方式有很多种，常见操作如下：

* 如果想要使用已经存在的数据创建 Tensor, 可以将其传入 :class:`Tensor` 的构造函数：
  支持传入标量（Scalar）、Python :class:`list` 以及 NumPy :class:`~numpy.ndarray` （当然也包括 Tensor 自己）；
* 对应地，Tensor 也支持通过 :meth:`~Tensor.numpy` , :meth:`~Tensor.tolist` 和 :meth:`~Tensor.numpy` 变成其它类型。
* 如果想要根据某些规则生成特定的 Tensor, 请参考 :mod:`megengine.functional` 中的 :ref:`tensor-creation` 部分。

.. note::

   Tensor 中重载了 Python 中常见的运算符，支持直接进行 ``+`` ``-`` ``*`` ``/`` 等操作。
   但请注意，与 Tensor 相关的绝大多数计算接口，都封装在 :mod:`megengine.functional` 模块中。

.. note::

   更多时候，我们希望能够将现实中的数据（或数据集）在 MegEngine 中进行处理。
   一种普遍的方法是将其转化为 :class:`~numpy.ndarray` 后传入 :class:`Tensor` 来创建实例；
   还可以利用 MegEngine 中封装好的接口，方便进行一些预处理操作，参考 :mod:`megengine.data` 模块。

Core
----
.. warning::

   我们不承诺 core 模块中 API 的兼容性和稳定性。

Core 模块中实现了 MegEngine 的核心功能，包括 Tensor 和 Operators 组成的计算图，自动求导机制等等。
MegEngine 用户在日常使用中无需直接调用这个模块，因为里面的功能实现已经被其它常用模块进行了封装。
但出于方便 MegEngine 开发者检索的目的，我们也将 Core API 列举在此处。

.. note::

   任何 MegEngine 的用户都可以尝试成为我们的开发人员，就好像你理解了 Tensor 和 Core 之后，
   便可以尝试去理解 NVIDIA 的 TensorCore_ 了。冷笑话 +1 (￣▽￣)" 

   .. _TensorCore: https://www.nvidia.cn/data-center/tensor-cores/

.. note::

   你可能在 MegEngine 源代码中经常会看到诸如 MegBrain, MGB 等字样。
   不用担心，MegBrain 是 MegEngine 的内部代号，二者某种程度上是等价的。

.. toctree::
   :maxdepth: 3
   
   core


