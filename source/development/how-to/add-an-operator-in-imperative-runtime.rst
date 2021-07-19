.. _add-an-operator-in-imperative-runtime:

====================================
如何在 Imperative Runtime 中添加算子
====================================

我们以添加 ``add`` 算子为例进行说明。

.. _imperative-op-trait:

注册 Op Trait
-------------

1. 在 :src:`imperative/src/impl/ops/` 目录下对应的文件中（例如 ``misc.cpp`` ）：
   
   * 定义好 ``apply_on_var_node`` 方法；
   * 调用宏 ``OP_TRAIT_REG``.

   .. code-block:: cpp

      namespace { namespace add {
      auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
        auto&& op = static_cast<const Add&>(def);
        mgb_assert(inputs.size() == 2);
        return opr::Add::make(inputs[0], inputs[1], op.param());
      }

      OP_TRAIT_REG(Add, Add)
                .apply_on_var_node(apply_on_var_node)
                .fallback();
      }}  // add

.. note::

   可以参考 :src:`dnn/src/common` 中的文件组织，确定对应的文件。

2. 在 :src:`src/core/include/megbrain/ir/ops.td` 中添加你新写的算子。

   .. code-block:: python

      def Add: MgbHashableOp<"Add", [AddParam]>;

.. _imperative-op:

添加算子
--------

.. warning::

  下面的流程中使用 ``add_example.py`` 文件作为举例，然而在实际开发时，
  新增算子（以及测试）最好按照对应的分类放在指定文件（如 ``elemwise.py`` ）中。
  可以参考已有的分类逻辑进行判断，如果依旧不清楚如何分类，可以向核心开发人员进行询问。

.. note::
 
  文档字符串写法请参考 :ref:`python-docstring-style-guide` 。

.. _imperative-op-functional:

Functional 实现
~~~~~~~~~~~~~~~

1. 向 :src:`imperative/python/megengine/functional` 目录添加文件 ``add_example.py``:

   .. code-block:: python

      from typing import Sequence

      from ..core._imperative_rt.core2 import apply
      from ..core.ops import builtin
      from ..tensor import Tensor
       
       
      def add_example(a: Tensor, b: Tensor, m: int=0) -> Tensor:
          """
          add example
          """
          op = builtin.Add(m)
          return apply(op, a, b)

2. 在 :src:`imperative/python/megengine/functional/__init__.py` 文件中进行导入：

   .. code-block:: python

      from .add_example import add_example

3. 在 :src:`imperative/python/test/unit/functional/` 对应文件中添加测试：

   .. code-block:: python

      def test_add_example():
         a = np.random.randn(2, 2, 2).astype(np.float32)
         b = np.random.randn(2, 2, 2).astype(np.float32)
         res = a + b + 3
         c = F.add_example(tensor(a), tensor(b), m=3)[0]
         np.testing.assert_allclose(c.numpy(), res, atol=1e-5)

.. note::

   如果有需要，还应该提供对应的 Module 实现。

.. _imperative-op-module:

Module 实现
~~~~~~~~~~~

1. 向 :src:`imperative/python/megengine/module` 目录添加文件 ``add_example.py``:

   .. code-block:: python

      from abc import abstractmethod
      from typing import Tuple, Union
       
      from ..functional import (
          add_example,
      )
      from .module import Module
       
      class AddExample(Module):
          def __init__(
              self, m: int,
          ):
              super().__init__()
              self.m = m
       
          def forward(self, a, b):
              return add_example(a, b, self.m)

2. 在 :src:`imperative/python/megengine/module/__init__.py` 文件中进行导入：

   .. code-block:: python

      from .add_example import AddExample

3. 在 :src:`imperative/python/test/unit/module/` 对应文件中添加测试。

.. _imperative-op-test:

编译和测试
----------

假设我们需要对 :src:`imperative/python/test/unit/functional/test_functional.py` 中的 ``test_add_example`` 进行测试。

1. 确保执行了 ``make develop`` 命令：

   .. code-block:: shell

      mkdir -p build
      cd build
      cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=ON -DMGE_BUILD_IMPERATIVE_RT=ON
      make -j$(nproc)
      make develop

   .. note::

      设置 ``DMGE_WITH_CUDA=ON`` 将开启 CUDA 来进行测试。

2. 设置 ``PYTHONPATH``:

   .. code-block:: shell

      export PYTHONPATH="$(git rev-parse --show-toplevel)/imperative/python"

3. 使用 ``pytest`` 进行测试：

   .. code-block:: shell

      pytest unit/functional/test_functional.py -k "test_add_example"

.. note::

   * 需要按照 :src:`imperative/python` 下的各个 ``requires`` 文件中的要求安装所需要的对应版本软件；
   * 编辑完 ``.py`` 文件之后请使用 :src:`imperative/python/scripts/format.sh` 进行格式化;
   * 想要执行 CUDA 测试，请确保你已经配置好了 CUDA 环境。

