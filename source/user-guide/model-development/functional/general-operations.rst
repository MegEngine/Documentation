.. _general-operations:

==========================
如何用 Tenosr 进行科学运算
==========================

可以通过 ``megengine.functional.xxx`` 形式调用的 API 被认为是通用 Tensor 运算，
负责提供常见的科学运算接口，该部分的 API 设计尽可能地向 NumPy API 靠拢。
所有的 API 都可以在 :ref:`general-tensor-operations` 中找到。

根据对 Tensor 形状的要求和影响，我们又可以把这些运算分为以下几大类：

* :ref:`element-wise-operations`
* :ref:`reduction-operations`

.. seealso::

   * 并不是所有的 NumPy 中的计算接口都有提供对应的 MegEngine 实现，但在处理数据时，
     你可以选择先调用 NumPy 实现获得 ndarray, 然后 :ref:`create-tensor-from-ndarray` ；
   * 如果你不理解一些 API 的使用方法，可以查询 NumPy 中关于对应 API 用法的介绍。

.. _element-wise-operations:

元素级别运算（Element-wise）
---------------------------

元素级别运算是 Tensor 运算中最常见的一大类，根据操作数的区别，
它既可以指对 Tensor 中每个位置的元素进行相同的运算（即一元运算），
也可以指在不同的 Tensor 之间的对应元素逐个进行相互运算（即二元或更多元运算），
这些运算自身又可以根据运算性质简略地区分为：

* 算术运算 （加减乘除等，参考 :ref:`arithmetic-operations` ）
* 三角函数与反三角函数（参考 :ref:`trigonometric-functions` 与 :ref:`hyperbolic-functions` ）
* 位运算（参考 :ref:`bit-operations` ）
* 逻辑运算（参考 :ref:`logic-functions` ）

在神经网络运算中，也有许多运算是元素级别的，比如激活函数 :py:func:`~.relu` 等。

元素级别的含义
~~~~~~~~~~~~~~

如果两个元素在各自的 Tensor 内占据着相同的位置，那么我们可以称这两个元素是对应的，
其中元素的位置由用于定位每个元素的 :ref:`索引 <access-tensor-element>` 确定。
我们用下面两个 Tensor ``a`` 和 ``b`` 作为例子：

>>> a = megengine.Tensor([[1., 2.], [3., 4.]])
>>> b = megengine.Tensor([[9., 8.], [7., 6.]])

我们使用相同的索引 ``[0][0]`` 去获取分别获取两个 Tensor 中的元素：

>>> a[0][0]
Tensor(1.0, device=xpux:0)

>>> b[0][0]
Tensor(9.0, device=xpux:0)

可以发现，``a`` 中值为 1 的元素对应着 ``b`` 中值为 9 的元素。其它 3 个位置的元素也分别对应。

.. note::

   对应关系由相同的索引定义，它表明了 Tensor 之间必须具有相同的形状才能进行元素间操作。

以加法为例子，我们可以当作是两个矩阵之间进行了矩阵加法：

.. math::

   \begin{bmatrix}
	1 & 2 \\
    3 & 4
   \end{bmatrix} + 
   \begin{bmatrix}
	9 & 8 \\
    7 & 6
   \end{bmatrix} = 
   \begin{bmatrix}
	10 & 10 \\
    10 & 10
   \end{bmatrix} 

>>> a + b
Tensor([[10.0 10.0]
 [10.0 100.]], dtype=int32, device=xpux:0)

.. warning::

   并不是形状完全相同的两个 Tensor 之间才能够进行元素级别的运算，
   如果两个 Tensor 的形状相互 “兼容”，则可以 :ref:`广播（Broadcast） <tensor-broadcasting>` 至相同的形状进行运算。
   这个机制让 Tensor 计算变得非常灵活。

.. seealso::

   人们也用 Component-wise / Point-wise 等术语来指代元素级别的运算。

与矩阵运算的对比
~~~~~~~~~~~~~~~~

.. warning::


与 ``+`` 类似，使用 ``*`` 可以用来计算矩阵的对应元素相乘，也叫哈达玛积（Hadamard product）：

.. math::

   \begin{bmatrix}
	1 & 2 \\
    3 & 4
   \end{bmatrix} \odot 
   \begin{bmatrix}
	9 & 8 \\
    7 & 6
   \end{bmatrix} = 
   \begin{bmatrix}
	9 & 16 \\
    21 & 24
   \end{bmatrix} 

>>> a = megengine.Tensor([[1., 2.], [3., 4.]])
>>> b = megengine.Tensor([[9., 8.], [7., 6.]])
>>> a * b
Tensor([[ 9. 16.]
 [21. 24.]], device=xpux:0)

.. warning::

  
   **不同的框架和库对于一些运算符的定义不同。** 在 Matlab 中使用 ``.*`` 和 ``.^`` 表示元素级别的乘法和乘方，
   使用 ``*`` 和 ``^`` 表示矩阵乘法和乘方，参考官网解释：
   `Array vs. Matrix Operations <https://www.mathworks.com/help/matlab/matlab_prog/array-vs-matrix-operations.html>`_

   一些人会将 ``*`` 误认为是矩阵乘法 :py:func:`~.matmul` , 实际上在 MegEngine 中矩阵乘法对应的运算符是 ``@`` .

   >>> a @ b
   Tensor([[23. 20.]
    [55. 48.]], device=xpux:0)

   它对应于 :py:mod:`functional` 模块中提供的 :py:func:`~.matmul` 接口：

   >>> megengine.functional.matmul(a, b)
   Tensor([[23. 20.]
    [55. 48.]], device=xpux:0)

.. seealso::

   更多与线性代数有关的运算，请参考 :ref:`linear-algebra-functions` . 

.. _reduction-operations:

归约运算（Reductioin）
----------------------

.. note::

   归约运算能够减少一个 Tensor 中元素的数量。

   我们可以理解成统计意义上的降维（Dimensionality reduction）。

一个最简单的例子是对 Tensor 中的元素求和，使用 :py:func:`~.sum` 接口：

>>> a = megengine.Tensor([[1, 2, 3], [4, 5, 6]])
>>> b = megengine.megengine.functional.sum(a)
Tensor(21, dtype=int32, device=xpux:0)

>>> print(a.shape, b.shape)
(2, 3) ()

可以看到，我们对一个形状为 ``(2, 3)`` 的 Tensor 求和后得到了一个 0 维 Tensor.

.. warning::

   * 归约运算并不总是将输入 Tensor 归约为具有单个元素的 0 维 Tensor.
     在传入 ``axis`` 参数且不为 `None` 时，则可以要求沿着轴进行规约，参考 :ref:`axis-argument` ；
   * 我们也可以通过设置参数 ``keepdims=True`` 来保持归约运算前后的维度不变。

.. seealso::

   * 常见的 Tensor 归约运算还有：:py:func:`~.prod` / :py:func:`~.mean` 等，
     可在 :ref:`statistical-functions` 中找到相关 API 和例程。
   * 想要了解更多关于规约的知识，可以参考维基百科中对 
     `Reduction operator <https://en.wikipedia.org/wiki/Reduction_operator>`_ 的解释。

