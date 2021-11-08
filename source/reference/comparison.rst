.. _comparison:

=======================
与其它框架 API 进行对比
=======================

.. warning::

   MegEngine 的 API 设计遵循 :ref:`mep-0003` , 向《数组 API 标准》靠齐。

   * 在同其它框架进行对比时，同样的命名不意味着用法也完全一致；
   * 如果有新的 API 支持需求，可在 GitHub 创建相应的 Issue 或 Pull Request.

.. note::

   * 你可以利用浏览器的查找功能在当前页面查询对应的 API.
   * 当前页面并非自动生成，如发现有缺失/过时内容，欢迎编辑当前页面。

.. seealso::

   当前页面更多用于检索，具有其它框架使用经验的用户还可以参考 :ref:`transfer-to-megengine` 。

Data Structure
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~numpy.ndarray`
     - :py:class:`~torch.Tensor`
     - :py:class:`~megengine.Tensor`
     - :ref:`tensor-guide`

General tensor operations
-------------------------

Creation Functions
~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.arange`
     - :py:func:`~torch.arange`
     - :py:func:`~megengine.functional.arange`
     -
   * - :py:func:`~numpy.linspace`
     - :py:func:`~torch.linspace`
     - :py:func:`~megengine.functional.linspace`
     -
   * - :py:func:`~numpy.eye`
     - :py:func:`~torch.eye`
     - :py:func:`~megengine.functional.eye`
     -
   * - :py:func:`~numpy.zeros`
     - :py:func:`~torch.zeros`
     - :py:func:`~megengine.functional.zeros`
     -
   * - :py:func:`~numpy.zeros_like`
     - :py:func:`~torch.zeros_like`
     - :py:func:`~megengine.functional.zeros_like`
     -
   * - :py:func:`~numpy.ones`
     - :py:func:`~torch.ones`
     - :py:func:`~megengine.functional.ones`
     -
   * - :py:func:`~numpy.ones_like`
     - :py:func:`~torch.ones_like`
     - :py:func:`~megengine.functional.ones_like`
     -
   * - :py:func:`~numpy.full`
     - :py:func:`~torch.full`
     - :py:func:`~megengine.functional.full`
     -
   * - :py:func:`~numpy.full_like`
     - :py:func:`~torch.full_like`
     - :py:func:`~megengine.functional.full_like`
     -

Manipulation Functions
~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.reshape`
     - :py:func:`~torch.reshape`
     - :py:func:`~megengine.functional.reshape`
     -
   * - :py:meth:`~numpy.ndarray.flatten`
     - :py:func:`~torch.flatten`
     - :py:func:`~megengine.functional.flatten`
     -
   * - :py:func:`~numpy.broadcast_to`
     - :py:func:`~torch.broadcast_to` / :py:meth:`~torch.Tensor.expand`
     - :py:func:`~megengine.functional.broadcast_to`
     -
   * - :py:func:`~numpy.expand_dims`
     - :py:func:`~torch.unsqueeze`
     - :py:func:`~megengine.functional.expand_dims`
     -
   * - :py:func:`~numpy.squeeze`
     - :py:func:`~torch.squeeze`
     - :py:func:`~megengine.functional.squeeze`
     -
   * - :py:func:`~numpy.concatenate`
     - :py:func:`~torch.cat`
     - :py:func:`~megengine.functional.concat`
     -
   * - :py:func:`~numpy.stack`
     - :py:func:`~torch.stack`
     - :py:func:`~megengine.functional.stack`
     -
   * - :py:func:`~numpy.split`
     - :py:func:`~torch.split`
     - :py:func:`~megengine.functional.split`
     -
   * - :py:func:`~numpy.tile`
     - :py:func:`~torch.tile`
     - :py:func:`~megengine.functional.tile`
     -
   * - :py:func:`~numpy.repeat`
     - :py:func:`~torch.repeat_interleave`
     - :py:func:`~megengine.functional.repeat`
     -
   * - :py:func:`~numpy.roll`
     - :py:func:`~torch.roll`
     - :py:func:`~megengine.functional.roll`
     -

Arithmetic operations
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:data:`~numpy.add`
     - :py:func:`~torch.add`
     - :py:func:`~megengine.functional.add`
     - ``+`` operator
   * - :py:data:`~numpy.subtract`
     - :py:func:`~torch.sub`
     - :py:func:`~megengine.functional.sub`
     - ``-`` operator
   * - :py:data:`numpy.multiply`
     - :py:func:`~torch.mul`
     - :py:func:`~megengine.functional.mul`
     - ``*`` operator
   * - :py:data:`~numpy.divide`
     - :py:func:`~torch.div`
     - :py:func:`~megengine.functional.div`
     - ``/`` operator
   * - :py:data:`~numpy.floor_divide`
     - :py:func:`~torch.floor_divide`
     - :py:func:`~megengine.functional.floor_div`
     - ``//`` operator
   * - :py:data:`~numpy.negative`
     - :py:func:`~torch.neg`
     - :py:func:`~megengine.functional.neg`
     -
   * - :py:data:`~numpy.absolute`
     - :py:func:`~torch.abs`
     - :py:func:`~megengine.functional.abs`
     -
   * - :py:data:`~numpy.power`
     - :py:func:`~torch.pow`
     - :py:func:`~megengine.functional.pow`
     - ``**`` operator
   * - :py:data:`~numpy.mod`
     - :py:func:`~torch.remainder`
     - :py:func:`~megengine.functional.mod`
     - ``%`` operator
   * - :py:data:`~numpy.sqrt`
     - :py:func:`~torch.sqrt`
     - :py:func:`~megengine.functional.sqrt`
     -
   * - :py:data:`~numpy.square`
     - :py:func:`~torch.square`
     - :py:func:`~megengine.functional.square`
     -
   * - :py:data:`~numpy.sign`
     - :py:func:`~torch.sign`
     - :py:func:`~megengine.functional.sign`
     -
   * - :py:data:`~numpy.maximum`
     - :py:func:`~torch.maximum`
     - :py:func:`~megengine.functional.maximum`
     -
   * - :py:data:`~numpy.minimum`
     - :py:func:`~torch.minimum`
     - :py:func:`~megengine.functional.minimum`
     -
   * - :py:meth:`~numpy.ndarray.round`
     - :py:func:`~torch.round`
     - :py:func:`~megengine.functional.round`
     -
   * - :py:data:`~numpy.ceil`
     - :py:func:`~torch.ceil`
     - :py:func:`~megengine.functional.ceil`
     -
   * - :py:data:`~numpy.floor`
     - :py:func:`~torch.floor`
     - :py:func:`~megengine.functional.floor`
     -
   * - :py:func:`~numpy.clip`
     - :py:func:`~torch.clamp`
     - :py:func:`~megengine.functional.clip`
     -
   * - :py:data:`~numpy.exp`
     - :py:func:`~torch.exp`
     - :py:func:`~megengine.functional.exp`
     -
   * - :py:data:`~numpy.expm1`
     - :py:func:`~torch.expm1`
     - :py:func:`~megengine.functional.expm1`
     -
   * - :py:data:`~numpy.log`
     - :py:func:`~torch.log`
     - :py:func:`~megengine.functional.log`
     -
   * - :py:data:`~numpy.log1p`
     - :py:func:`~torch.log1p`
     - :py:func:`~megengine.functional.log1p`
     -

Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:data:`~numpy.sin`
     - :py:func:`~torch.sin`
     - :py:func:`~megengine.functional.sin`
     -
   * - :py:data:`~numpy.cos`
     - :py:func:`~torch.cos`
     - :py:func:`~megengine.functional.cos`
     -
   * - :py:data:`~numpy.tan`
     - :py:func:`~torch.tan`
     - :py:func:`~megengine.functional.tan`
     -
   * - :py:data:`~numpy.arcsin`
     - :py:func:`~torch.asin`
     - :py:func:`~megengine.functional.asin`
     -
   * - :py:data:`~numpy.arccos`
     - :py:func:`~torch.acos`
     - :py:func:`~megengine.functional.acos`
     -
   * - :py:data:`~numpy.arctan`
     - :py:func:`~torch.atan`
     - :py:func:`~megengine.functional.atan`
     -

Hyperbolic functions
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:data:`~numpy.sinh`
     - :py:func:`~torch.sinh`
     - :py:func:`~megengine.functional.sinh`
     -
   * - :py:data:`~numpy.cosh`
     - :py:func:`~torch.cosh`
     - :py:func:`~megengine.functional.cosh`
     -
   * - :py:data:`~numpy.tanh`
     - :py:func:`~torch.tanh`
     - :py:func:`~megengine.functional.tanh`
     -
   * - :py:data:`~numpy.arcsinh`
     - :py:func:`~torch.asinh`
     - :py:func:`~megengine.functional.asinh`
     -
   * - :py:data:`~numpy.arccosh`
     - :py:func:`~torch.acosh`
     - :py:func:`~megengine.functional.acosh`
     -
   * - :py:data:`~numpy.arctanh`
     - :py:func:`~torch.atanh`
     - :py:func:`~megengine.functional.atanh`
     -

Bit operations
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:data:`~numpy.left_shift`
     - Not Found
     - :py:func:`~megengine.functional.left_shift`
     - ``<<`` operator
   * - :py:data:`~numpy.right_shift`
     - Not Found
     - :py:func:`~megengine.functional.right_shift`
     - ``>>`` operator

Logic functions
~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:data:`~numpy.isnan`
     - :py:func:`~torch.isnan`
     - :py:func:`~megengine.functional.isnan`
     -
   * - :py:data:`~numpy.isinf`
     - :py:func:`~torch.isinf`
     - :py:func:`~megengine.functional.isinf`
     -
   * - :py:data:`~numpy.logical_and`
     - Not Found
     - :py:func:`~megengine.functional.logical_and`
     - ``&`` operator
   * - :py:data:`~numpy.logical_not`
     - Not Found
     - :py:func:`~megengine.functional.logical_not`
     - ``~`` operator
   * - :py:data:`~numpy.logical_or`
     - Not Found
     - :py:func:`~megengine.functional.logical_or`
     - ``|`` operator
   * - :py:data:`~numpy.logical_xor`
     - Not Found
     - :py:func:`~megengine.functional.logical_xor`
     - ``^`` operator
   * - :py:data:`~numpy.equal`
     - :py:func:`~torch.equal`
     - :py:func:`~megengine.functional.equal`
     -
   * - :py:data:`~numpy.not_equal`
     - :py:func:`~torch.not_equal`
     - :py:func:`~megengine.functional.not_equal`
     -
   * - :py:data:`~numpy.less`
     - :py:func:`~torch.less`
     - :py:func:`~megengine.functional.less`
     -
   * - :py:data:`~numpy.less_equal`
     - :py:func:`~torch.less_equal`
     - :py:func:`~megengine.functional.less_equal`
     -
   * - :py:data:`~numpy.greater`
     - :py:func:`~torch.greater`
     - :py:func:`~megengine.functional.greater`
     -
   * - :py:data:`~numpy.greater_equal`
     - :py:func:`~torch.greater_equal`
     - :py:func:`~megengine.functional.greater_equal`
     -

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.sum`
     - :py:func:`~torch.sum`
     - :py:func:`~megengine.functional.sum`
     -
   * - :py:func:`~numpy.prod`
     - :py:func:`~torch.prod`
     - :py:func:`~megengine.functional.prod`
     -
   * - :py:func:`~numpy.mean`
     - :py:func:`~torch.mean`
     - :py:func:`~megengine.functional.mean`
     -
   * - :py:meth:`~numpy.ndarray.min`
     - :py:func:`~torch.min`
     - :py:func:`~megengine.functional.min`
     -
   * - :py:meth:`~numpy.ndarray.max`
     - :py:func:`~torch.max`
     - :py:func:`~megengine.functional.max`
     -
   * - :py:func:`~numpy.var`
     - :py:func:`~torch.var`
     - :py:func:`~megengine.functional.var`
     -
   * - :py:func:`~numpy.std`
     - :py:func:`~torch.std`
     - :py:func:`~megengine.functional.std`
     -
   * - :py:func:`~numpy.cumsum`
     - :py:func:`~torch.cumsum`
     - :py:func:`~megengine.functional.cumsum`
     -


Linear Algebra Functions
~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.transpose`
     - :py:func:`~torch.transpose`
     - :py:func:`~megengine.functional.transpose`
     -
   * - :py:func:`~numpy.dot`
     - :py:func:`~torch.dot`
     - :py:func:`~megengine.functional.dot`
     -
   * - :py:func:`~numpy.linalg.inv`
     - :py:func:`~torch.linalg.inv`
     - :py:func:`~megengine.functional.matinv`
     -
   * - :py:data:`~numpy.matmul`
     - :py:func:`~torch.matmul`
     - :py:func:`~megengine.functional.matmul`
     -
   * - :py:func:`~numpy.linalg.svd`
     - :py:func:`~torch.linalg.svd`
     - :py:func:`~megengine.functional.svd`
     -
   * - :py:func:`~numpy.linalg.norm`
     - :py:func:`~torch.norm`
     - :py:func:`~megengine.functional.norm`
     -

Indexing Functions
~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.take_along_axis`
     - :py:func:`~torch.gather`
     - :py:func:`~megengine.functional.gather`
     -
   * - :py:func:`~numpy.put_along_axis`
     - :py:func:`~torch.scatter`
     - :py:func:`~megengine.functional.scatter`
     -
   * - :py:func:`~numpy.where`
     - :py:func:`~torch.where`
     - :py:func:`~megengine.functional.where` / :py:func:`~megengine.functional.cond_take`
     - 取决于传参情况

Searching Functions
~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.argmin`
     - :py:func:`~torch.argmin`
     - :py:func:`~megengine.functional.argmin`
     -
   * - :py:func:`~numpy.argmax`
     - :py:func:`~torch.argmax`
     - :py:func:`~megengine.functional.argmax`
     -


Sorting Functions
~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - NumPy
     - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~numpy.argsort`
     - :py:func:`~torch.argsort`
     - :py:func:`~megengine.functional.argsort`
     -
   * - :py:func:`~numpy.sort`
     - :py:func:`~torch.sort`
     - :py:func:`~megengine.functional.sort`
     -

NN Funtional Operations
-----------------------
Convolution functions
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.conv1d`
     - :py:func:`~megengine.functional.nn.conv1d`
     -
   * - :py:func:`~torch.nn.functional.conv2d`
     - :py:func:`~megengine.functional.nn.conv2d`
     -
   * - :py:func:`~torch.nn.functional.conv3d`
     - :py:func:`~megengine.functional.nn.conv3d`
     -
   * - :py:func:`~torch.nn.functional.conv_transpose1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.conv_transpose2d`
     - :py:func:`~megengine.functional.nn.conv_transpose2d`
     -
   * - :py:func:`~torch.nn.functional.conv_transpose3d`
     - :py:func:`~megengine.functional.nn.conv_transpose3d`
     -
   * - local_conv2d
     - :py:func:`~megengine.functional.nn.local_conv2d`
     -
   * - deformable_conv2d
     - :py:func:`~megengine.functional.nn.deformable_conv2d`
     -
   * - :py:func:`~torch.nn.functional.unfold`
     - :py:func:`~megengine.functional.nn.sliding_window`
     -
   * - :py:func:`~torch.nn.functional.fold`
     - :py:func:`~megengine.functional.nn.sliding_window_transpose`
     -

Pooling functions
~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.avg_pool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.avg_pool2d`
     - :py:func:`~megengine.functional.nn.avg_pool2d`
     -
   * - :py:func:`~torch.nn.functional.avg_pool3d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.max_pool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.max_pool2d`
     - :py:func:`~megengine.functional.nn.max_pool2d`
     -
   * - :py:func:`~torch.nn.functional.max_pool3d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.max_unpool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.max_unpool2d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.max_unpool3d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.lp_pool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.lp_pool2d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.adaptive_max_pool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.adaptive_max_pool2d`
     - :py:func:`~megengine.functional.nn.adaptive_max_pool2d`
     -
   * - :py:func:`~torch.nn.functional.adaptive_max_pool3d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.adaptive_avg_pool1d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.adaptive_avg_pool2d`
     - :py:func:`~megengine.functional.nn.adaptive_avg_pool2d`
     -
   * - :py:func:`~torch.nn.functional.adaptive_avg_pool3d`
     - :ref:`not-implemented`
     -

Non-linear activation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.threshold`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.relu`
     - :py:func:`~megengine.functional.nn.relu`
     -
   * - :py:func:`~torch.nn.functional.hardtanh`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.hardswish`
     - :py:func:`~megengine.functional.nn.hswish`
     -
   * - :py:func:`~torch.nn.functional.relu6`
     - :py:func:`~megengine.functional.nn.relu6`
     -
   * - :py:func:`~torch.nn.functional.elu`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.selu`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.celu`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.leaky_relu`
     - :py:func:`~megengine.functional.nn.leaky_relu`
     -
   * - :py:func:`~torch.nn.functional.prelu`
     - :py:func:`~megengine.functional.nn.prelu`
     -
   * - :py:func:`~torch.nn.functional.rrelu`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.glu`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.gelu`
     - :py:func:`~megengine.functional.nn.gelu`
     -
   * - :py:func:`~torch.nn.functional.logsigmoid`
     - :py:func:`~megengine.functional.nn.logsigmoid`
     -
   * - :py:func:`~torch.nn.functional.hardshrink`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.tanhshrink`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.softsign`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.softplus`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.softmin`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.softmax`
     - :py:func:`~megengine.functional.nn.softmax`
     -
   * - :py:func:`~torch.nn.functional.softshrink`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.gumbel_softmax`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.log_softmax`
     - :py:func:`~megengine.functional.nn.logsoftmax`
     -
   * - :py:func:`~torch.nn.functional.sigmoid`
     - :py:func:`~megengine.functional.nn.sigmoid`
     -
   * - :py:func:`~torch.nn.functional.hardsigmoid`
     - :py:func:`~megengine.functional.nn.hsigmoid`
     -
   * - :py:func:`~torch.nn.functional.silu`
     - :py:func:`~megengine.functional.nn.silu`
     -

Normalization functions
~~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.batch_norm`
     - :py:func:`~megengine.functional.nn.batch_norm`
     -
   * - :py:func:`~torch.nn.functional.instance_norm`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.layer_norm`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.local_response_norm`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.normalize`
     - :py:func:`~megengine.functional.normalize`
     -

Linear functions
~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.linear`
     - :py:func:`~megengine.functional.nn.linear`
     -
   * - :py:func:`~torch.nn.functional.bilinear`
     - :ref:`not-implemented`
     -

Dropout functions
~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.dropout`
     - :py:func:`~megengine.functional.nn.dropout`
     -
   * - :py:func:`~torch.nn.functional.alpha_dropout`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.feature_alpha_dropout`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.dropout2d`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.dropout3d`
     - :ref:`not-implemented`
     -

Sparse functions
~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.embedding`
     - :py:func:`~megengine.functional.nn.embedding`
     -
   * - :py:func:`~torch.nn.functional.embedding_bag`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.one_hot`
     - :py:func:`~megengine.functional.nn.one_hot`
     -

Metric functions
~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.pairwise_distance`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.cosine_similarity`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.pdist`
     - :ref:`not-implemented`
     -

Loss functions
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.binary_cross_entropy_with_logits`
     - :py:func:`~megengine.functional.nn.binary_cross_entropy`
     -
   * - :py:func:`~torch.nn.functional.poisson_nll_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.cosine_embedding_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.cross_entropy`
     - :py:func:`~megengine.functional.nn.cross_entropy`
     -
   * - :py:func:`~torch.nn.functional.ctc_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.hinge_embedding_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.kl_div`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.l1_loss`
     - :py:func:`~megengine.functional.nn.l1_loss`
     -
   * - :py:func:`~torch.nn.functional.mse_loss`
     - :py:func:`~megengine.functional.nn.square_loss`
     -
   * - :py:func:`~torch.nn.functional.margin_ranking_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.multilabel_margin_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.multilabel_soft_margin_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.multi_margin_loss`
     - :py:func:`~megengine.functional.nn.hinge_loss`
     -
   * - :py:func:`~torch.nn.functional.nll_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.smooth_l1_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.soft_margin_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.triplet_margin_loss`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.triplet_margin_with_distance_loss`
     - :ref:`not-implemented`
     -

NN Module
---------
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.parameter.Parameter`
     - :py:class:`~megengine.Parameter`
     -

Containers
~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Module`
     - :py:class:`~megengine.module.Module`
     -
   * - :py:class:`~torch.nn.Sequential`
     - :py:class:`~megengine.module.Sequential`
     -
   * - :py:class:`~torch.nn.ModuleList`
     - MegEngine 原生支持
     -
   * - :py:class:`~torch.nn.ModuleDict`
     - MegEngine 原生支持
     -
   * - :py:class:`~torch.nn.ParameterList`
     - MegEngine 原生支持
     -
   * - :py:class:`~torch.nn.ParameterDict`
     - MegEngine 原生支持
     -

Initialization
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.init.calculate_gain`
     - :py:class:`~megengine.module.init.calculate_gain`
     -
   * - _calculate_fan_in_and_fan_out
     - :py:class:`~megengine.module.init.calculate_fan_in_and_fan_out`
     -
   * - _calculate_correct_fan
     - :py:class:`~megengine.module.init.calculate_correct_fan`
     -
   * - :py:func:`~torch.nn.init.uniform_`
     - :py:class:`~megengine.module.init.uniform_`
     -
   * - :py:func:`~torch.nn.init.normal_`
     - :py:class:`~megengine.module.init.normal_`
     -
   * - :py:func:`~torch.nn.init.constant_`
     - :py:class:`~megengine.module.init.fill_`
     -
   * - :py:func:`~torch.nn.init.ones_`
     - :py:class:`~megengine.module.init.ones_`
     -
   * - :py:func:`~torch.nn.init.zeros_`
     - :py:class:`~megengine.module.init.zeros_`
     -
   * - :py:func:`~torch.nn.init.eye_`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.init.dirac_`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.init.xavier_uniform_`
     - :py:class:`~megengine.module.init.xavier_uniform_`
     -
   * - :py:func:`~torch.nn.init.xavier_normal_`
     - :py:class:`~megengine.module.init.xavier_normal_`
     -
   * - :py:func:`~torch.nn.init.kaiming_uniform_`
     - :py:class:`~megengine.module.init.msra_uniform_`
     -
   * - :py:func:`~torch.nn.init.kaiming_normal_`
     - :py:class:`~megengine.module.init.msra_normal_`
     -
   * - :py:func:`~torch.nn.init.orthogonal_`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.init.sparse_`
     - :ref:`not-implemented`
     -

Convolution Layers
~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Conv1d`
     - :py:class:`~megengine.module.Conv1d`
     -
   * - :py:class:`~torch.nn.Conv2d`
     - :py:class:`~megengine.module.Conv2d`
     -
   * - :py:class:`~torch.nn.Conv3d`
     - :py:class:`~megengine.module.Conv3d`
     -
   * - :py:class:`~torch.nn.ConvTranspose1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.ConvTranspose2d`
     - :py:class:`~megengine.module.ConvTranspose2d`
     -
   * - :py:class:`~torch.nn.ConvTranspose3d`
     - :py:class:`~megengine.module.ConvTranspose3d`
     -
   * - LocalConv2d
     - :py:class:`~megengine.module.LocalConv2d`
     -
   * - DeformableConv2d
     - :py:class:`~megengine.module.DeformableConv2d`
     -
   * - :py:class:`~torch.nn.Conv1d`
     - :py:class:`~megengine.module.Conv1d`
     -
   * - :py:class:`~torch.nn.Unfold`
     - :py:class:`~megengine.module.SlidingWindowTranspose`
     -
   * - :py:class:`~torch.nn.Fold`
     - :py:class:`~megengine.module.SlidingWindow`
     -

Pooling layers
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.MaxPool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MaxPool2d`
     - :py:class:`~megengine.module.MaxPool2d`
     -
   * - :py:class:`~torch.nn.MaxPool3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MaxUnpool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MaxUnpool2d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MaxUnpool3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AvgPool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AvgPool2d`
     - :py:class:`~megengine.module.AvgPool2d`
     -
   * - :py:class:`~torch.nn.AvgPool3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.FractionalMaxPool2d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LPPool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LPPool2d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AdaptiveMaxPool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AdaptiveMaxPool2d`
     - :py:class:`~megengine.module.AdaptiveMaxPool2d`
     -
   * - :py:class:`~torch.nn.AdaptiveMaxPool3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AdaptiveAvgPool1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AdaptiveAvgPool2d`
     - :py:class:`~megengine.module.AdaptiveAvgPool2d`
     -
   * - :py:class:`~torch.nn.AdaptiveAvgPool3d`
     - :ref:`not-implemented`
     -

Padding Layers
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.ReflectionPad1d`
     - :py:class:`~megengine.module.Pad`
     - mode = REFLECT
   * - :py:class:`~torch.nn.ReflectionPad2d`
     - :py:class:`~megengine.module.Pad`
     - mode = REFLECT
   * - :py:class:`~torch.nn.ReflectionPad3d`
     - :py:class:`~megengine.module.Pad`
     - mode = REFLECT
   * - :py:class:`~torch.nn.ReplicationPad1d`
     - :py:class:`~megengine.module.Pad`
     - mode = EDGE
   * - :py:class:`~torch.nn.ReplicationPad2d`
     - :py:class:`~megengine.module.Pad`
     - mode = EDGE
   * - :py:class:`~torch.nn.ReplicationPad3d`
     - :py:class:`~megengine.module.Pad`
     - mode = EDGE
   * - :py:class:`~torch.nn.ZeroPad2d`
     - :py:class:`~megengine.module.Pad`
     - mode = CONSTANT
   * - :py:class:`~torch.nn.ConstantPad1d`
     - :py:class:`~megengine.module.Pad`
     - mode = CONSTANT
   * - :py:class:`~torch.nn.ConstantPad2d`
     - :py:class:`~megengine.module.Pad`
     - mode = CONSTANT
   * - :py:class:`~torch.nn.ConstantPad3d`
     - :py:class:`~megengine.module.Pad`
     - mode = CONSTANT

Non-linear Activations
~~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.ELU`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Hardshrink`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Hardsigmoid`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Hardtanh`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Hardswish`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LeakyReLU`
     - :py:class:`~megengine.module.LeakyReLU`
     -
   * - :py:class:`~torch.nn.LogSigmoid`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MultiheadAttention`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.PReLU`
     - :py:class:`~megengine.module.PReLU`
     -
   * - :py:class:`~torch.nn.ReLU`
     - :py:class:`~megengine.module.ReLU`
     -
   * - :py:class:`~torch.nn.ReLU6`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.RReLU`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.SELU`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.CELU`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.GELU`
     - :py:class:`~megengine.module.GELU`
     -
   * - :py:class:`~torch.nn.Sigmoid`
     - :py:class:`~megengine.module.Sigmoid`
     -
   * - :py:class:`~torch.nn.SiLU`
     - :py:class:`~megengine.module.SiLU`
     -
   * - :py:class:`~torch.nn.Softplus`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Softshrink`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Softsign`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Tanh`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Tanhshrink`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Threshold`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Softmin`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Softmax`
     - :py:class:`~megengine.module.Softmax`
     -
   * - :py:class:`~torch.nn.Softmax2d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LogSoftmax`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AdaptiveLogSoftmaxWithLoss`
     - :ref:`not-implemented`
     -

Normalization Layers
~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.BatchNorm1d`
     - :py:class:`~megengine.module.BatchNorm1d`
     -
   * - :py:class:`~torch.nn.BatchNorm2d`
     - :py:class:`~megengine.module.BatchNorm2d`
     -
   * - :py:class:`~torch.nn.BatchNorm3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.GroupNorm`
     - :py:class:`~megengine.module.GroupNorm`
     -
   * - :py:class:`~torch.nn.SyncBatchNorm`
     - :py:class:`~megengine.module.SyncBatchNorm`
     -
   * - :py:class:`~torch.nn.InstanceNorm1d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.InstanceNorm2d`
     - :py:class:`~megengine.module.InstanceNorm`
     -
   * - :py:class:`~torch.nn.InstanceNorm3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LayerNorm`
     - :py:class:`~megengine.module.LayerNorm`
     -
   * - :py:class:`~torch.nn.LocalResponseNorm`
     - :ref:`not-implemented`
     -

Recurrent Layers
~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.RNNBase`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.RNN`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LSTM`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.GRU`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.RNNCell`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.LSTMCell`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.GRUCell`
     - :ref:`not-implemented`
     -

Transformer Layers
~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Transformer`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TransformerEncoder`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TransformerDecoder`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TransformerEncoderLayer`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TransformerDecoderLayer`
     - :ref:`not-implemented`
     -

Linear Layers
~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Identity`
     - :py:class:`~megengine.module.Identity`
     -
   * - :py:class:`~torch.nn.Linear`
     - :py:class:`~megengine.module.Linear`
     -
   * - :py:class:`~torch.nn.Bilinear`
     - :ref:`not-implemented`
     -

Dropout Layers
~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Dropout`
     - :py:class:`~megengine.module.Dropout`
     -
   * - :py:class:`~torch.nn.Dropout2d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.Dropout3d`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.AlphaDropout`
     - :ref:`not-implemented`
     -

Sparse Layers
~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.Embedding`
     - :py:class:`~megengine.module.Embedding`
     -
   * - :py:class:`~torch.nn.EmbeddingBag`
     - :ref:`not-implemented`
     -

Distance Functions
~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.CosineSimilarity`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.PairwiseDistance`
     - :ref:`not-implemented`
     -

Loss Functions
~~~~~~~~~~~~~~

.. seealso::

   请参考 loss function 的 functional 实现。

.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:class:`~torch.nn.L1Loss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MSELoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.CrossEntropyLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.CTCLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.NLLLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.PoissonNLLLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.KLDivLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.BCELoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.BCEWithLogitsLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MarginRankingLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.HingeEmbeddingLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MultiLabelMarginLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.SmoothL1Loss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.SoftMarginLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MultiLabelSoftMarginLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.CosineEmbeddingLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.MultiMarginLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TripletMarginLoss`
     - :ref:`not-implemented`
     -
   * - :py:class:`~torch.nn.TripletMarginWithDistanceLoss`
     - :ref:`not-implemented`
     -

Vision functions
----------------
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - :py:func:`~torch.nn.functional.pixel_shuffle`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.pad`
     - :ref:`not-implemented`
     -
   * - :py:func:`~torch.nn.functional.interpolate`
     - :py:func:`~megengine.functional.nn.interpolate`
     -
   * - :py:func:`~torch.nn.functional.upsample`
     - :py:func:`~megengine.functional.nn.interpolate`
     -
   * - :py:func:`~torch.nn.functional.upsample_nearest`
     - :py:func:`~megengine.functional.nn.interpolate`
     -
   * - :py:func:`~torch.nn.functional.upsample_bilinear`
     - :py:func:`~megengine.functional.nn.interpolate`
     -
   * - :py:func:`~torch.nn.functional.grid_sample`
     - :py:func:`~megengine.functional.nn.remap`
     -
   * - :py:func:`~torch.nn.functional.affine_grid`
     - :py:func:`~megengine.functional.nn.warp_affine`
     -
   * - :py:func:`~torchnn.ops.nms`
     - :py:func:`~megengine.functional.nn.nms`
     -
   * - :py:func:`~torchnn.ops.roi_align`
     - :py:func:`~megengine.functional.nn.roi_align`
     -
   * - :py:func:`~torchnn.ops.roi_pool`
     - :py:func:`~megengine.functional.nn.roi_pooling`
     -

OpenCV Python Package
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment

   * - cvtColor
     - :py:func:`~megengine.functional.nn.cvt_color`
     -
   * - resize
     - :py:func:`~megengine.functional.nn.interpolate`
     -
   * - remap
     - :py:func:`~megengine.functional.nn.remap`
     -
   * - warpAffine
     - :py:func:`~megengine.functional.nn.warp_affine`
     -
   * - warpPerspective
     - :py:func:`~megengine.functional.nn.warp_perspective`
     -

NVIDIA
~~~~~~
.. list-table::
   :header-rows: 1

   * - Pytorch
     - MegEngine
     - Comment


   * - correlation
     - :py:func:`~megengine.functional.nn.correlation`
     -
   * - nvof
     - :py:func:`~megengine.functional.nn.nvof`
     -

.. _not-implemented:

Not Implemeted
--------------

.. note::

   一些 API 在 MegEngine 中可能还没有实现，但所有的 API 并不是一开始就被设计出来的。
   我们可以像搭积木一样，利用已经存在的基础 API 来组合出 MegEngine 中尚未提供的接口。

   比如 “如何实现 :py:func:`~torch.roll` ” 这个问题，可以使用 :py:func:`~.functional.split` 和 :py:func:`~.functional.concat` 拼接出来：

   .. code-block:: python

      import megengine.functional as F

      def roll(x, shifts, axis):
          shp = x.shape
          dim = len(shp)
          if isinstance(shifts, int):
              assert isinstance(axis, int)
              shifts = [shifts]
              axis = [axis]
          assert len(shifts) == len(axis)
          y = x
          for i in range(len(shifts)):
              axis_ = axis[i]
              shift_ = shifts[i]
              axis_t_ = axis_ + dim if axis_ < 0 else axis_
              assert (
                  dim > axis_t_ >= 0
              ), "axis out of range (expected to be in range of [{}, {}], but got {})".format(
                  -dim, dim - 1, axis_
              )
              if shift_ == 0:
                  continue
                  size = shp[axis_t_]
              if shift_ > 0:
                  a, b = F.split(y, [size - shift_,], axis=axis_t_)
              else:
                  a, b = F.split(y, [-shift_,], axis=axis_t_)
              y = F.concat((b, a), axis=axis_t_)
            return y

   除此之外，你可以尝试在 GitHub Issues 或论坛中针对 API 问题发起求助。

   我们也欢迎你将自己实现的 API 以 Pull Request 的形式提交到 MegEngine 代码库中来～

.. note::

   对于缺失的 Loss Funtions 算子，大都可自行设计实现。

