.. _tensor-examples:

Tensor 具象化举例
-----------------

.. image:: ../../../_static/images/cube.svg
   :align: center
   :height: 128px

我们可以借助上面这张魔方（ `图片来源 <https://commons.wikimedia.org/wiki/File:Rubiks_cube.jpg>`_ ）来直观地理解 Tensor:

* 首先，我们假设这个魔方是“实心同质”的，是一个存在于现实世界中的 Tensor;
* 这个 Tensor 里面的每个元素的类型（:attr:`~.Tensor.dtype` ）都是一致的（方方正正的形状、外加一样的做工）；
* 而且这是一个维度（:attr:`~.Tensor.ndim` ）为 :math:`3` 的结构，形状（:attr:`~.Tensor.shape` ）为 :math:`(3, 3, 3)` ; 
* 对应地，该 Tensor 的总元素个数（:attr:`~.Tensor.size` ）是 :math:`3*3*3=27`.

如果你将每种颜色代表一个值，而每个魔方块的值可以用其具有的颜色值之和来表示（此时中间块为零），
那么不同的魔方块就具有了各自的取值，就好像 Tensor 中的每个元素可以有自己的取值一样。
事实上，除了魔方以外，还有很多东西可以抽象成 Tensor 数据结构，意味着可以利用 MegEngine 进行相关的计算。

