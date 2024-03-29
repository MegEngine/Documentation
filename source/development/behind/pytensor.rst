.. _pytensor:

===============================
MegEngine Python 层 Tensor 行为
===============================

从逻辑上来讲，各层之间的引用关系如下图所示：

.. mermaid::

    flowchart TD
    var["Python 解释器里的变量，例如 a"]
    var --> |reference | pytensor
    pytensor["Python Tensor(pyobj)"]
    ctensor["C++ Tensor"]
    storage["Storage"]
    pytensor --> |reference| ctensor;
    ctensor --> |reference| storage


三者均通过 refcount 进行资源管理，在引用归零时就释放资源，其中：

* Python Tensor 只包含对 C++ Tensor 的引用；用户可通过 ``id(a)`` 是否一直来验证是否发生了变化
* C++ Tensor 包含：shape / stride / 对 Storage 的引用指针
* Storage 包含：一段显存，即 ptr + length

在各种情况中，各层变量之间的指向关系变化如下表所示：

+---------------------------------+---------------------+--------------+--------------------------+
| 解释器变量行为                  | Python Tensor       | C++ Tensor   | Storage                  |
+=================================+=====================+==============+==========================+
| ``a += 1``                      | 不变                | 新建         | 新建，老的 ref - 1       |
+---------------------------------+---------------------+--------------+--------------------------+
| ``a = a + 1``                   | 新建，老的 ref -1   | 新建         | 新建，老的 ref - 1       |
+---------------------------------+---------------------+--------------+--------------------------+
| ``b = a[0] （<=v1.8)``          | 新建                | 新建         | 新建并拷贝必须的部分     |
+---------------------------------+---------------------+--------------+--------------------------+
| ``b = a[0] (>=v1.9)``           | 新建                | 新建         | 复用老的，即 ref + 1     |
+---------------------------------+---------------------+--------------+--------------------------+
| ``b = a.reshape(...) (>=v1.9)`` | 新建                | 新建         | 复用老的                 |
+---------------------------------+---------------------+--------------+--------------------------+
| ``b = F.transpose(a) (>=v1.9)`` | 新建                | 新建         | 复用老的                 |
+---------------------------------+---------------------+--------------+--------------------------+
| ``a[0] = 1``                    | 不变                | 新建         | 新建并拷贝，老 ref - 1   |
+---------------------------------+---------------------+--------------+--------------------------+
| ``b = a``                       | 不变                | 不变         | 不变                     |
+---------------------------------+---------------------+--------------+--------------------------+

习题：基于以上的内容，就比较容易推导出以下的一些组合的行为了：

* ``a = mge.tensor([1, 2]); b = a; b += 1; print(a)``
* ``a = mge.tensor([1, 2]); b = a; b = b + 1; print(a)``
* ``a = mge.tensor([1, 2]); b = a[0]; b += 1; print(a)``
* ``a = mge.tensor([1, 2]);  b = a; a[0] += 1; print(b)``

正确答案

.. code-block:: python

     [2, 3]
     [1, 2]
     [1, 2]
     [2, 2]

如果你觉得这个行为很奇怪，你可能会希望了解 Python 可变/不可变变量的行为区别。
