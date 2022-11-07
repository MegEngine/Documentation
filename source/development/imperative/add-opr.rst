.. _imperative:

==================================================
Imperative in Action —— 如何在 Imperative 添加 op
==================================================



imperative op 与 megdnn / megbrain op 的关系
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``megdnn`` 是底层的算子库，无论是静态图还是动态图最终都会调用 ``megdnn``
的 ``op`` / ``kernel`` 完成计算。

``megbrain`` 的 ``op`` 对应的是静态图实现，其会调用 ``megdnn`` 的
``kernel`` 进行计算。

``imperative`` 的 ``op``
对应的是动态图实现，在某些情况下（\ ``trace``\ ）会调用 ``megbrain`` 的
``op``\ ，也可能直接调用 ``kernel`` 进行计算。

在 MegDNN 添加一个 op
~~~~~~~~~~~~~~~~~~~~~~~~~

MegDNN 设计原理
^^^^^^^^^^^^^^^^^^^^^

``MegDNN``
是一个跨平台的计算库，提供了包括卷积、池化、矩阵乘法、点乘等在内的基本算术运算和神经网络原语。\ ``MegDNN``
在底层针对 ``x86``\ 、\ ``CUDA``\ 、\ ``ARM``
等不同平台做了各类算子的针对性实现，并为上层（主要是 ``MegBrain`` 和
``Imperative``\ ）提供统一的 ``API`` 接口。

.. figure:: ../../_static/images/megdnn-api.png


``MegDNN`` 向上提供的 ``API`` 写在了 ``/dnn/include/megdnn/oprs/``
下的头文件中，常见的 ``API`` 有以下几种：

-  ``deduce_layout``

   根据输入 ``Tensor`` 的 ``layout``\ ，推断输出 ``Tensor`` 的
   ``layout``\ （如果有输出张量的话）。

-  ``get_workspace_in_bytes``

   根据输入和输出 ``Tensor`` 的 ``layout``\ ，获取需要的 ``workspace``
   的大小，返回值为 ``size_t``\ ，通常声明为
   ``virtual``\ ，对不同后端做不同实现。

-  ``check_exec``

   对 ``kernel`` 的执行做一些检查，通常会结合 ``deduce_layout`` 和
   ``get_workspace_in_bytes`` 检查输入输出 ``Tensor``
   的合法性（\ ``shape``\ 、\ ``dtype``\ ），\ ``workspace``
   的大小是否满足要求等。

-  ``exec``

   ``kernel`` 的具体实现函数，通常声明为
   ``virtual``\ ，对不同后端做不同实现。

注意，这些 ``API`` 的命名并非是固定的，比如当一个算子有反向的 ``kernel``
时，通常反向算子的 ``API`` 会叫 ``deduce_layout_bwd``\ （前向的算子的
``API``\ 可能叫
``deduce_layout_fwd``\ ）等，以和前向算子的方法区分（虽然本质上都是类似的计算，\ ``dnn``
并不知道前向和反向的概念）。

``MegDNN`` 实现的（部分）组织架构如下图所示：

.. figure:: ../../_static/images/megdnn-architecture.png


-  ``Inferface Class``

   定义 ``opr`` 的接口。

-  ``common``

   定义了各个平台的公共代码，比如上面介绍的各个 ``operator`` 的
   ``deduce_layout`` 等方法，还有 ``OperatorBase`` （所有 ``Opr``
   的基类，所有 ``Opr`` 都直接或间接继承自
   ``OperatorBase``\ ）的方法定义、\ ``Handle`` （\ ``MegDNN``
   的抽象计算设备，\ ``MegDNN`` 的所有 ``operator`` 都是由 ``handle``
   创建的）的方法定义。

-  ``naive``

   顾名思义，\ ``naive`` 下的 ``kernel`` 实现是 ``opr``
   的简单粗暴（无优化）版实现，不考虑性能、内存效率等，一般用于对其他平台下的
   ``opr`` 实现进行正确性验证。

   当 ``opr`` 没有其他后端的特定实现时，也会调用 ``naive``
   下的实现作为默认实现。

-  ``fallback``

   通用的代码实现，适用于采用 ``SSE`` 或 ``NEON``
   等技术并没有加速效果的简单 ``opr``\ 。

-  ``arm_common``

   针对 ``ARMv7`` 和 ``AArch64`` 的通用代码优化技术。

-  ``armv7/aarch64/x86/CUDA/OpenCL``

   针对特定平台的代码优化，可能包含底层汇编代码等。

``MegDNN`` 中对于同一 ``opr`` 可能包含不同的算法实现。比如卷积，在
``MegDNN``
中包含直接卷积、\ ``im2col-gemm``\ 、\ ``winograd``\ 、\ ``FFT``
等。因此在 ``MegDNN`` 中，有不同的选择 ``kernel``
的算法，可以采用指定、启发式方法等来选择算法（\ ``kernel``\ ）：

.. figure:: ../../_static/images/choose-algorithm.png


添加 MegDNN opr 实践
^^^^^^^^^^^^^^^^^^^^^^^^^^

根据\ `MegDNN
文件结构 <https://wiki.megvii-inc.com/display/brainuser/MegDNN#MegDNN-%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84>`__\ ，我们要在
``MegDNN`` 下实现一个 ``opr``\ ，通常需要：

1. 在 ``/dnn/include/megdnn/oprs/`` 下声明这个 ``opr``
   的类（作为对上层的接口）。
2. 在 ``/dnn/src/`` 下添加这个 ``opr`` 在不同后端的实现代码。
3. 在 ``/dnn/test/`` 下添加这个 ``opr`` 的测试，实现了多个后端的
   ``kernel`` 也需要添加不同后端下的测试。

接下来，我们尝试添加一个 ``Add opr`` ，这个 ``opr`` 接收两个
``Tensor``\ ：\ ``A`` 和 ``B``\ ，并且有三个 ``int``
参数：\ ``x``\ 、\ ``y`` 和 ``z``\ 。输出 ``Tensor``\ 为：
``C = x * A + y * B + z``.

以这个 ``opr`` 为例，我们将介绍如何定义一个
``opr``\ ，如何实现其前向和反向的接口，不同后端下的 ``opr``
实现（\ ``naive`` 和 ``CUDA``\ ）以及如何为 ``MegDNN opr`` 添加测试。

实现前向算子 AddForward
'''''''''''''''''''''''''''''''

1. 在 ``/dnn/scripts/opr_param_defs.py`` 中为我们的 ``opr`` 添加定义。

   .. code:: python

      (pdef('Add').
       add_fields('int32', Doc('x', 'The multiplication coefficient of the first Tensor'), '1').
       add_fields('int32', Doc('y', 'The multiplication coefficient of the second Tensor'), '1').
       add_fields('int32', Doc('z', 'The offset of matrix addition'), '0')
       )

   这里定义了我们的 ``opr`` 的名字为 ``Add``\ ，有三个 ``int32``
   类型的参数 ``x``\ 、\ ``y``\ 、\ ``z``\ ，分别为第一个 ``Tensor``
   的乘法系数、第二个 ``Tensor`` 的乘法系数以及最后加上的偏移量。

2. 在 ``dnn/include/megdnn/oprs/`` 下选择合适的文件添加 ``Add``
   类的声明。

   根据 `MegDNN
   文件结构 <https://wiki.megvii-inc.com/display/brainuser/MegDNN#MegDNN-%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84>`__\ ，我们添加的
   ``opr`` 是一个 ``Tensor`` 计算操作，因此我们选择在 ``general.h``
   中添加 ``Add`` 类的声明。

   .. code:: cpp

      class AddBase : public OperatorBase {
          DEF_OPR_IMPL(AddBase, OperatorBase, 2, 1);

      protected:
          void deduce_layout_fwd(
                  const TensorLayout& data1, const TensorLayout& data2,
                  TensorLayout& dst);
          void check_layout_fwd(
                  const TensorLayout& data1, const TensorLayout& data2,
                  const TensorLayout& dst);
      };

      class AddForward : public AddBase {
          DEF_OPR_PARAM(Add);
          DEF_OPR_IMPL(AddForward, AddBase, 2, 1);

      public:
          virtual void exec(_megdnn_tensor_in data1, _megdnn_tensor_in data2,
                            _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
          void deduce_layout(const TensorLayout& data1, const TensorLayout& data2,
                             TensorLayout& dst);
          virtual size_t get_workspace_in_bytes(
                             const TensorLayout& data1,
                             const TensorLayout& data2,
                             const TensorLayout& dst) = 0;

      protected:
          void check_exec(const TensorLayout& data1, const TensorLayout& data2,
                          const TensorLayout& dst, size_t workspace_in_bytes);
      };
      using Add = AddForward;

   这里我们创建一个直接继承自 ``OperatorBase`` 的基类 ``AddBase``
   ，前向算子 ``AddForward`` 继承自 ``AddBase``
   类。这样做是出于可扩展性考虑，因为我们之后还要添加反向算子。

   ``AddBase`` 类定义了前向的 ``deduce_layout`` 方法和 ``check_layout``
   方法（都是用于辅助 ``AddForward`` 类的方法的）。

   在 ``AddForward`` 类里我们用 ``DEF_OPR_PARAM(Add);``
   表明这个类中需要用到我们刚刚在 ``opr_param_defs.py`` 中定义的
   ``param Add``\ 。

   ``DEF_OPR_IMPL(AddForward, AddBase, 2, 1);`` 这句话的含义是
   ``AddForward`` 算子接收两个输入并且有一个输出。

   注意到推断 ``layout`` 的方法 ``deduce_layout`` 和检查合法性方法
   ``check_exec``
   不是虚函数，因为这些检查与具体后端无关，而实际的执行函数 ``exec``
   和获取 ``get_workspace_in_bytes``
   则是纯虚函数，因为它们的实现与具体后端有关。

   另外值得注意的是，我们定义在 ``opr_param_defs.py`` 中的参数
   ``x``\ 、\ ``y``\ 、\ ``z`` 是不作为 ``opr`` 的参数传到类的方法（如
   ``exec``\ ）中的，类方法接受的参数为输出和输出 ``Tensor`` 以及
   ``workspace``\ （\ ``workspace``
   是一个字节数组，表明计算中需要使用的临时空间的大小）。

3. 在 ``dnn/src/common``
   下添加所有后端平台的公共类，实现公共方法（也就是与具体后端无关的方法）。

   创建 ``add.cpp`` 文件，实现 ``dnn/include/megdnn/oprs/general.h``
   中的非虚函数：

   .. code:: cpp

      #include "megdnn/oprs.h"
      #include "src/common/utils.h"

      namespace megdnn {

      void AddBase::deduce_layout_fwd(
              const TensorLayout& data1,  const TensorLayout& data2,
              TensorLayout& dst) {
          megdnn_assert(data1.is_physical_contiguous());
          megdnn_assert(data2.is_physical_contiguous());
          megdnn_assert(dst.is_physical_contiguous());
          auto errmsg = [&]() {
              return megdnn_layout_msg(data1) + ", " + megdnn_layout_msg(data2) + ", " +
                     megdnn_layout_msg(dst);
          };
          megdnn_assert_eq_layout(data1, data2);
          auto data1_dtype = data1.dtype, data2_dtype = data2.dtype;
          megdnn_assert(
                  data1_dtype == data2_dtype &&
                  (data1_dtype.category() == DTypeCategory::INT ||
                   data1_dtype.category() == DTypeCategory::FLOAT));
          dst = TensorLayout{data1};
      }

      void AddBase::check_layout_fwd(
                  const TensorLayout& data1, const TensorLayout& data2,
                  const TensorLayout& dst) {
          TensorLayout dst_expected;
          megdnn_assert_eq_shape(data1, data2);
          megdnn_assert_eq_dtype(data1, data2);
          megdnn_assert_eq_shape(data1, dst);
          megdnn_assert_eq_dtype(data1, dst);
          deduce_layout_fwd(data1, data2, dst_expected);
          megdnn_assert_eq_shape(dst_expected, dst);
      }

      void AddForward::deduce_layout(
                  const TensorLayout& data1, const TensorLayout& data2,
                  TensorLayout& dst) {
          check_layout_fwd(data1, data2, dst);
      }

      void AddForward::check_exec(
                  const TensorLayout& data1, const TensorLayout& data2,
                  const TensorLayout& dst, size_t workspace_in_bytes) {
          check_layout_fwd(data1, data2, dst);
          auto required_workspace_in_bytes = get_workspace_in_bytes(data1, data2, dst);
          megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
      }

      }   // namespace megdnn

   ``AddBase::deduce_layout_fwd`` 方法主要是根据输出对输出的
   ``dtype``\ 、\ ``layout`` 等属性做推断，在
   ``AddForward::deduce_layout`` 中直接调用这个方法就好了。

   ``AddBase::check_layout_fwd``
   方法是对输入和输出的属性进行检查，同样地，在
   ``AddForward::check_exec`` 调用该方法检查执行过程的合法性。

4. 在 ``dnn/src/common/handle_impl.h`` 的
   ``\#define MEGDNN_FOREACH_OPR_CLASS(cb)`` 中添加
   ``cb(AddForward)``\ 。

   各个平台都会 ``include`` 这个文件，这里 ``megdnn`` 会为这些 ``opr``
   创建 ``operator``\ ，详见 ``dnn/src/common/handle.cpp``\ 。

5. 定义各个平台的虚函数。

   以 ``naive`` 版本为例，在 ``dnn/src/naive/`` 下创建文件夹 ``add``\ 。

   创建 ``opr`` 的头文件 ``opr_impl.h``\ ：

   .. code:: cpp

      #pragma once
      #include "megdnn/oprs.h"

      namespace megdnn {
      namespace naive {

      class AddForwardImpl : public AddForward {
      public:
          using AddForward::AddForward;
          void exec(
                  _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
                  _megdnn_workspace workspace) override;
          size_t get_workspace_in_bytes(
                  const TensorLayout& data1, const TensorLayout& data2,
                  const TensorLayout& dst) override {
              return 0;
          }

      private:
          template <typename T>
          void exec_internal(
                  int x, const T* __restrict data1, int y, const T* __restrict data2, int z,
                  T* __restrict dst, size_t n);
      };

      }  // namespace naive
      }  // namespace megdnn

   这里函数 ``exec`` 和 ``get_workspace_in_bytes`` 就是我们在
   ``class AddForward`` 里声明的虚函数，函数 ``exec_internal``
   是在我们执行 ``exec`` 时，根据每种 ``type`` 都生成一个执行
   ``exec_internal``\ ，再通过宏 ``MEGDNN_DISPATCH_CPU_KERN_OPR`` 将
   ``opr`` 的执行 ``kernel`` 放到 ``handle`` 上执行。

   具体实现，创建 ``opr_impl.cpp``\ ：

   .. code:: cpp

      #include "src/naive/add/opr_impl.h"

      #include "src/common/utils.h"
      #include "src/naive/handle.h"

      namespace megdnn {
      namespace naive {

      template <typename T>
      void AddForwardImpl::exec_internal(
              int x, const T* __restrict data1, int y, const T* __restrict data2, int z,
              T* __restrict dst, size_t n) {
          rep(i, n) { dst[i] = x * data1[i] + y * data2[i] + z; }
      }

      void AddForwardImpl::exec(
              _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) {
          check_exec(data1.layout, data2.layout, dst.layout, workspace.size);
          auto n = data1.layout.total_nr_elems();
      #define cb(DType)                                                             \
          if (data1.layout.dtype == DType()) {                                      \
              using ctype = typename DTypeTrait<DType>::ctype;                      \
              MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                    \
                      param().x, data1.ptr<ctype>(), param().y, data2.ptr<ctype>(), \
                      param().z, dst.ptr<ctype>(), n));                             \
              return;                                                               \
          }
          MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
      #undef cb
          megdnn_assert_internal(0);
      }

      }  // namespace naive
      }  // namespace megdnn

   可以发现 ``opr`` 的实际实现逻辑是写在 ``exec_internal`` 里的，我们在
   ``opr_param_def.py`` 中定义的参数可以通过 ``param().xxx``
   的方式获得，并作为参数传递给 ``exec_internal``\ 。

6. 在 ``dnn/src/naive/handle.cpp`` 加上 ``add`` 的头文件
   ``opr_impl.h``\ 。

   .. code:: cpp

      #include "src/naive/add/opr_impl.h"

7. 添加测试。

   在 ``dnn/test/common/`` 下创建 ``add.h``\ 。在这里构造测试需要的
   ``args``\ 。

   .. code:: cpp

      #pragma once
      #include "megdnn/basic_types.h"
      #include "megdnn/opr_param_defs.h"

      namespace megdnn {
      namespace test {

      namespace add {

      struct TestArg {
          TensorShape data1, data2;
          TestArg(TensorShape data1, TensorShape data2) : data1(data1), data2(data2) {}
      };

      inline std::vector<TestArg> get_args() {
          std::vector<TestArg> args;

          args.emplace_back(TensorShape{2, 2}, TensorShape{2, 2});
          return args;
      }

      }  // namespace add
      }  // namespace test
      }  // namespace megdnn

   这里声明了输入 ``Tensor`` 的 ``shape``\ 。

   注意，这里需要的 ``args`` 只是输入 ``Tensor``\ ，在
   ``opr_param_defs.py`` 中定义的参数不需要在这里构造。

   在 ``dnn/src/common/opr_trait.h`` 添加这个 ``opr`` 的 ``traits``\ ：

   .. code:: cpp

      DEF(AddForward, 3, true, true);

   这里， ``3`` 表示测试有 ``3`` 个
   ``Tensor``\ （两个输入、一个输出）。第三个参数表示是否需要
   ``workspace``\ 。第四个参数表示是否可以 ``deduce_layout``\ 。

   接下来，在 ``dnn/test/naive/`` 下创建
   ``add.cpp``\ ，添加详细的测试代码：

   .. code:: cpp

      #include "test/common/add.h"
      #include "megdnn/dtype.h"
      #include "megdnn/oprs.h"
      #include "test/common/checker.h"
      #include "test/naive/fixture.h"

      namespace megdnn {
      namespace test {

      TEST_F(NAIVE, ADD) {
          Checker<Add> checker(handle(), false);
          Add::Param param;
          param.x = 2;
          param.y = 3;
          param.z = 4;

          checker.set_param(param).exect(
                  Testcase{
                          TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4}),
                          TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8}),
                          {}},
                  Testcase{
                          {},
                          {},
                          TensorValue({1, 2, 2}, dtype::Float32(), {21, 26, 31, 36})});
      }

      }  // namespace test
      }  // namespace megdnn

   这里创建了一个名叫 ``ADD`` 的测试，它属于 ``Add`` 这个 ``opr``\ 。

   第一个 ``TestCase`` 下为执行 ``opr`` 前的三个 ``Tensor``\ ，第二个
   ``TestCase`` 为执行 ``opr`` 之后的三个 ``Tensor``\ 。

8. 跑测试。

   编译代码，在 ``build`` 下执行这个命令来跑 ``ADD`` 测试：

   .. code:: bash

      ./dnn/test/megdnn_test --gtest_filter="NAIVE.ADD:NAIVE.ADD"

   如果得到如下结果，说明测试通过：

   .. figure:: ../../_static/images/naive-add.png


至此，我们成功地添加了一个 ``kernel`` 的 ``naive`` 实现。

实现反向算子 AddBackward
''''''''''''''''''''''''''''''''

接下来，我们实现 ``Add`` 算子的反向 ``DNN`` 算子 ``AddBackwardData1`` 和
``AddBackwardData2``\ 。这两个算子分别是根据前向的输出 ``diff``
和一个输入张量计算得到另一个输入张量。

前面提到过，在 ``MegDNN``
层实际上没有“反向”的概念，可以把反向也认为是一个前向的操作。因此，实现反向
``kernel`` 的过程和实现前向 ``kernel`` 的过程其实是一样的。

1. 首先，我们在 ``dnn/scripts/opr_param_defs.py`` 里加入这两个 ``Opr``
   的定义：

   .. code:: python

      (pdef('AddBackwardData1').
       add_fields('int32', Doc('x', 'The multiplication coefficient of the first Tensor'), '1').
       add_fields('int32', Doc('y', 'The multiplication coefficient of the second Tensor'), '1').
       add_fields('int32', Doc('z', 'The offset of matrix addition'), '0')
       )

      (pdef('AddBackwardData2').
       add_fields('int32', Doc('x', 'The multiplication coefficient of the first Tensor'), '1').
       add_fields('int32', Doc('y', 'The multiplication coefficient of the second Tensor'), '1').
       add_fields('int32', Doc('z', 'The offset of matrix addition'), '0')
       )

   可以发现除了名字以外，这里的定义和前向 ``Add``
   算子是一样的，并无不同。

2. 在 ``dnn/include/megdnn/oprs/general.h`` 中定义 ``Opr``\ 。

   在基类 ``AddBase`` 类中增加反向的方法：

   .. code:: cpp

      class AddBase : public OperatorBase {
          DEF_OPR_IMPL(AddBase, OperatorBase, 2, 1);

      protected:
          ...
          void deduce_layout_bwd(
                  const TensorLayout& diff, TensorLayout& data);
          void check_layout_bwd(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2);
      };

   定义两个反向 ``Opr`` 的类：

   .. code:: cpp

      class AddBackwardData1 : public AddBase {
          DEF_OPR_PARAM(AddBackwardData1);
          DEF_OPR_IMPL(AddBackwardData1, AddBase, 2, 1);

      public:
          virtual void exec(
                  _megdnn_tensor_in diff, _megdnn_tensor_in data2,
                  _megdnn_tensor_out data1,
                  _megdnn_workspace workspace) = 0;
          void deduce_layout(
                  const TensorLayout& diff, TensorLayout& data1, TensorLayout& data2);
          virtual size_t get_workspace_in_bytes(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2) = 0;

      protected:
          void check_exec(
                  const TensorLayout& diff, const TensorLayout& data2,
                  const TensorLayout& data1, size_t workspace_in_bytes);
      };

      class AddBackwardData2 : public AddBase {
          DEF_OPR_PARAM(AddBackwardData2);
          DEF_OPR_IMPL(AddBackwardData2, AddBase, 2, 1);

      public:
          virtual void exec(
                  _megdnn_tensor_in diff, _megdnn_tensor_in data1,
                  _megdnn_tensor_out data2,
                  _megdnn_workspace workspace) = 0;
          void deduce_layout(
                  const TensorLayout& diff, TensorLayout& data1, TensorLayout& data2);
          virtual size_t get_workspace_in_bytes(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2) = 0;

      protected:
          void check_exec(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2, size_t workspace_in_bytes);
      };

   反向算子和前向算子的区别在于：反向算子以前向算子的输出 ``dst``
   作为反向算子的输入之一 ``diff``\ ，并与其他的输入 ``tensor``
   一起计算某一个输入张量。

   比如 ``AddBackwardData1`` 是用 ``diff`` 和
   ``data2``\ （输入的第二个张量） 计算
   ``data1``\ （输入的第一个张量）。\ ``AddBackwardData2`` 与之类似。

   因此类中的函数签名有些许不同，在实现 ``kernel``
   时需要根据实际情况填写函数参数。

3. 在 ``dnn/src/common/add.cpp`` 中实现第二步中新增的方法。

   .. code:: cpp

      void AddBase::deduce_layout_bwd(
              const TensorLayout& diff, TensorLayout& data) {
          megdnn_assert(diff.is_physical_contiguous());
          megdnn_assert(data.is_physical_contiguous());
          auto errmsg = [&]() {
              return megdnn_layout_msg(diff) + ", " + megdnn_layout_msg(data);
          };
          auto diff_dtype = diff.dtype, data_dtype = data.dtype;
          megdnn_assert(
                  diff_dtype.category() == DTypeCategory::INT ||
                  diff_dtype.category() == DTypeCategory::FLOAT);
          data = TensorLayout{diff};
      }

      void AddBase::check_layout_bwd(
              const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2) {
          TensorLayout data1_expected, data2_expected;
          megdnn_assert_eq_shape(diff, data1);
          megdnn_assert_eq_dtype(diff, data1);
          megdnn_assert_eq_shape(diff, data2);
          megdnn_assert_eq_dtype(diff, data2);
          megdnn_assert_eq_shape(data1, data2);
          megdnn_assert_eq_dtype(data1, data2);
          deduce_layout_bwd(diff, data1_expected);
          deduce_layout_bwd(diff, data2_expected);
          megdnn_assert_eq_shape(data1_expected, data1);
          megdnn_assert_eq_shape(data2_expected, data2);
      }
      ...
      void AddBackwardData1::deduce_layout(
              const TensorLayout& diff, TensorLayout& data1, TensorLayout& data2) {
          check_layout_bwd(diff, data1, data2);
      }

      void AddBackwardData1::check_exec(
              const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2, 
              size_t workspace_in_bytes) {
          check_layout_bwd(diff, data1, data2);
          auto required_workspce_in_bytes = get_workspace_in_bytes(diff, data1, data2);
          megdnn_assert(workspace_in_bytes >= required_workspce_in_bytes);
      }

      void AddBackwardData2::deduce_layout(
              const TensorLayout& diff, TensorLayout& data1, TensorLayout& data2) {
          check_layout_bwd(diff, data1, data2);
      }

      void AddBackwardData2::check_exec(
              const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2, 
              size_t workspace_in_bytes) {
          check_layout_bwd(diff, data1, data2);
          auto required_workspce_in_bytes = get_workspace_in_bytes(diff, data1, data2);
          megdnn_assert(workspace_in_bytes >= required_workspce_in_bytes);
      }

4. 在 ``dnn/src/common/handle_impl.h`` 中 ``cb`` 新增的 ``opr``\ ：

   .. code:: cpp

      cb(AddForward)      \
      cb(AddBackwardData1) \
      cb(AddBackwardData2)

5. 在 ``dnn/src/naive/add/opr_impl.h`` 下添加反向 ``opr``
   的类定义，添加在 ``general.h`` 中的纯虚函数的签名。

   .. code:: cpp

      class AddBackwardData1Impl : public AddBackwardData1 {
      public:
          using AddBackwardData1::AddBackwardData1;
          void exec(
                  _megdnn_tensor_in diff, _megdnn_tensor_in data2, _megdnn_tensor_out data1,
                  _megdnn_workspace workspace) override;
          size_t get_workspace_in_bytes(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2) override {
              return 0;
          }

      private:
          template <typename T>
          void exec_internal(
                  int x, T* __restrict data1, int y, const T* __restrict data2, int z,
                  const T* __restrict diff, size_t n);
      };

      class AddBackwardData2Impl : public AddBackwardData2 {
      public:
          using AddBackwardData2::AddBackwardData2;
          void exec(
                  _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_out data2,
                  _megdnn_workspace workspace) override;
          size_t get_workspace_in_bytes(
                  const TensorLayout& diff, const TensorLayout& data1,
                  const TensorLayout& data2) override {
              return 0;
          }

      private:
          template <typename T>
          void exec_internal(
                  int x, const T* __restrict data1, int y, T* __restrict data2, int z,
                  const T* __restrict diff, size_t n);
      };

6. 在 ``dnn/src/naive/add/opr_impl.cpp`` 中添加反向 ``opr`` 的 ``exec``
   实现。

   .. code:: cpp

      template <typename T>
      void AddBackwardData1Impl::exec_internal(
              int x, T* __restrict data1, int y, const T* __restrict data2, int z,
              const T* __restrict diff, size_t n) {
          rep(i, n) { data1[i] = (diff[i] - y * data2[i] - z) / x; }
      }

      void AddBackwardData1Impl::exec(
              _megdnn_tensor_in diff, _megdnn_tensor_in data2, _megdnn_tensor_out data1,
              _megdnn_workspace workspace) {
          check_exec(diff.layout, data1.layout, data2.layout, workspace.size);
          auto n = diff.layout.total_nr_elems();
      #define cb(DType)                                                             \
          if (diff.layout.dtype == DType()) {                                       \
              using ctype = typename DTypeTrait<DType>::ctype;                      \
              MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                    \
                      param().x, data1.ptr<ctype>(), param().y, data2.ptr<ctype>(), \
                      param().z, diff.ptr<ctype>(), n));                            \
              return;                                                               \
          }
          MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
      #undef cb
          megdnn_assert_internal(0);
      }

      template <typename T>
      void AddBackwardData2Impl::exec_internal(
              int x, const T* __restrict data1, int y, T* __restrict data2, int z,
              const T* __restrict diff, size_t n) {
          rep(i, n) { data2[i] = (diff[i] - x * data1[i] - z) / y; }
      }

      void AddBackwardData2Impl::exec(
              _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_out data2,
              _megdnn_workspace workspace) {
          check_exec(diff.layout, data1.layout, data2.layout, workspace.size);
          auto n = diff.layout.total_nr_elems();
      #define cb(DType)                                                             \
          if (diff.layout.dtype == DType()) {                                       \
              using ctype = typename DTypeTrait<DType>::ctype;                      \
              MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                    \
                      param().x, data1.ptr<ctype>(), param().y, data2.ptr<ctype>(), \
                      param().z, diff.ptr<ctype>(), n));                            \
              return;                                                               \
          }
          MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
      #undef cb
          megdnn_assert_internal(0);
      }

   这里与前向的区别主要在于 ``exec_internal`` 方法，因为是根据 ``diff``
   和其中一个张量推导另一个张量。需要注意带上 ``const``
   限定符的是输入，要推导的张量不带 ``const`` 限定符。

7. 添加测试。

   在 ``dnn/test/common/add.h`` 中为两个反向 ``opr`` 需要的测试 ``args``
   添加构造：

   .. code:: cpp

      namespace add_backward_data1 {
      struct TestArg {
          TensorShape diff, data2;
          TestArg(TensorShape diff, TensorShape data2) : diff(diff), data2(data2) {}
      };

      inline std::vector<TestArg> get_args() {
          std::vector<TestArg> args;

          args.emplace_back(TensorShape{2, 2}, TensorShape{2, 2});
          return args;
      }
      }  // namespace add_backward_data1

      namespace add_backward_data2 {
      struct TestArg {
          TensorShape diff, data1;
          TestArg(TensorShape diff, TensorShape data1) : diff(diff), data1(data1) {}
      };

      inline std::vector<TestArg> get_args() {
          std::vector<TestArg> args;

          args.emplace_back(TensorShape{2, 2}, TensorShape{2, 2});
          return args;
      }
      }  // namespace add_backward_data2

   在 ``dnn/test/naive/add.cpp`` 中增加两个新的测试用例，验证反向
   ``kernel`` 计算结果是否正确。

   需要注意的是，这里的 ``TestCase`` 的参数顺序需要和
   ``dnn/test/common/add.h`` 中声明的顺序保持一致：

   .. code:: cpp

      TEST_F(NAIVE, ADDBACKWARDDATA1) {
          Checker<AddBackwardData1> checker(handle(), false);
          AddBackwardData1::Param param;
          param.x = 2;
          param.y = 3;
          param.z = 4;

          checker.set_param(param).exect(
                  Testcase{
                          TensorValue({1, 2, 2}, dtype::Float32(), {21, 26, 31, 36}),
                          TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8}),
                          {}},
                  Testcase{{}, {}, TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4})});
      }

      TEST_F(NAIVE, ADDBACKWARDDATA2) {
          Checker<AddBackwardData2> checker(handle(), false);
          AddBackwardData2::Param param;
          param.x = 2;
          param.y = 3;
          param.z = 4;

          checker.set_param(param).exect(
                  Testcase{
                          TensorValue({1, 2, 2}, dtype::Float32(), {21, 26, 31, 36}),
                          TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4}),
                          {}},
                  Testcase{{}, {}, TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8})});
      }

8. 跑测试。

   编译代码，在 ``build`` 下执行这个命令来跑 ``ADD`` 测试：

   .. code:: bash

      ./dnn/test/megdnn_test --gtest_filter="NAIVE.ADD:NAIVE.ADD*"

   ``ADD*`` 表示运行所有名称以 ``ADD``
   开头的测试。也可以修改为明确跑某一个测试。

   如果得到如下结果，说明测试通过（这里还测试了其他的名称以 ``ADD``
   开头的测试）：

   .. figure:: ../../_static/images/naive-add-backward.png


这样，我们就实现了 ``naive`` 的 ``Add`` 反向算子。

不同后端的实现
''''''''''''''''''''''

许多 ``kernel`` 都需要在不同后端分别实现，这一节我们讲一下在 ``CUDA``
后端实现 ``Add`` 算子。

1. 在 ``dnn/src/cuda/`` 下创建 ``add`` 文件夹，创建 ``common.cuh``
   文件，作为前向算子和反向算子的公共头文件，\ ``cuh`` 是
   ``cuda header`` 的缩写。

   以前向为例，添加如下代码：

   .. code:: cpp

      #pragma once
      #include <cuda_runtime_api.h>
      #include <stdint.h>

      namespace megdnn {
      namespace cuda {
      namespace add {

      template <typename T>
      void forward_proxy(
              int x, const T* __restrict data1, int y, const T* __restrict data2, int z,
              T* __restrict dst, size_t n, cudaStream_t stream);

      }  // namespace add
      }  // namespace cuda
      }  // namespace megdnn

2. 在 ``dnn/src/cuda/add/`` 下创建 ``opr_impl.h``
   文件，\ ``AddForwardImpl`` 类，继承自 ``general.h`` 中的
   ``class AddForward``\ ，并声明对应的虚函数。

   .. code:: cpp

      #pragma once
      #include "megdnn/oprs.h"

      namespace megdnn {
      namespace cuda {

      class AddForwardImpl : public AddForward {
      public:
          using AddForward::AddForward;
          void exec(
                  _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
                  _megdnn_workspace workspace) override;
          size_t get_workspace_in_bytes(
                  const TensorLayout& data1, const TensorLayout& data2,
                  const TensorLayout& dst) override {
              return 0;
          }
      };

      }  // namespace cuda
      }  // namespace megdnn

3. 在 ``dnn/src/cuda/add/`` 下创建 ``add_forward.cu`` 文件，实现 ``Add``
   算子前向在 ``CUDA`` 端的逻辑。

   .. code:: cpp

      #include "megdnn/dtype.h"
      #include "src/cuda/add/common.cuh"
      #include "src/cuda/utils.cuh"

      namespace {

      template <typename T>
      __global__ void forward_kernel(
              int x, const T* __restrict data1, int y, const T* __restrict data2, int z,
              T* __restrict dst, size_t n) {
          size_t i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {
              dst[i] = x * data1[i] + y * data2[i] + z;
          }
      }

      }  // anonymous namespace

      namespace megdnn {
      namespace cuda {
      namespace add {

      template <typename T>
      void forward_proxy(
              int x, const T* __restrict data1, int y, const T* __restrict data2, int z,
              T* __restrict dst, size_t n, cudaStream_t stream) {
          forward_kernel<T><<<DIVUP(n, NR_THREADS), NR_THREADS, 0, stream>>>(
                  x, data1, y, data2, z, dst, n);
          after_kernel_launch();
      }

      #define INST(T)                                                                     \
          template void forward_proxy<T>(                                                 \
                  int, const T* __restrict, int, const T* __restrict, int, T* __restrict, \
                  size_t, cudaStream_t);
      #define cb(DType) INST(typename DTypeTrait<DType>::ctype)
      MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

      }  // namespace add
      }  // namespace cuda
      }  // namespace megdnn

4. 在 ``dnn/src/cuda/add/`` 下创建 ``add_forward.cpp``\ ，实现
   ``AddForwardImpl::exec`` 方法，\ ``exec`` 方法做的唯一的事就是通过
   ``forward_proxy`` 调用 ``add_forward.cu`` 中的
   ``forward_kernel``\ （在 ``CUDA`` 端的具体计算逻辑）。

   .. code:: cpp

      #include "src/cuda/add/common.cuh"
      #include "src/cuda/add/opr_impl.h"

      #include "src/cuda/utils.h"

      namespace megdnn {
      namespace cuda {

      void AddForwardImpl::exec(
              _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) {
          check_exec(data1.layout, data2.layout, dst.layout, workspace.size);
          auto stream = cuda_stream(this->handle());
          auto n = data1.layout.total_nr_elems();
      #define cb(DType)                                                             \
          if (data1.layout.dtype == DType()) {                                      \
              using ctype = typename DTypeTrait<DType>::ctype;                      \
              add::forward_proxy<ctype>(                                            \
                      param().x, data1.ptr<ctype>(), param().y, data2.ptr<ctype>(), \
                      param().z, dst.ptr<ctype>(), n, stream);                      \
          }
          MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
      #undef cb
      }

      }  // namespace cuda
      }  // namespace megdnn

5. 在 ``dnn/src/cuda/handle_create.cpp`` 中注册我们刚刚添加的 ``cuda``
   方法。

   .. code:: cpp

      ...
      #include "src/cuda/add/opr_impl.h"
      ...
      MEGDNN_SPECIALIZE_CREATE_OPERATOR(AddForward);

6. 添加 ``CUDA kernel`` 的测试。

   在 ``dnn/test/cuda/`` 下创建 ``add.cpp``\ ，添加测试用例：

   .. code:: cpp

      #include "test/common/add.h"
      #include "megdnn/dtype.h"
      #include "megdnn/oprs.h"
      #include "test/common/checker.h"
      #include "test/cuda/fixture.h"

      namespace megdnn {
      namespace test {

      TEST_F(CUDA, ADD) {
          Checker<Add> checker(handle_cuda());
          Add::Param param;
          param.x = 2;
          param.y = 3;
          param.z = 4;

          checker.set_param(param).exect(
                  Testcase{
                          TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4}),
                          TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8}),
                          {}},
                  Testcase{
                          {},
                          {},
                          TensorValue({1, 2, 2}, dtype::Float32(), {21, 26, 31, 36})});
      }

      }  // namespace test
      }  // namespace megdnn

7. 编译，运行测试。

   在 ``build`` 目录下添加如下编译参数：

   .. code:: bash

      cmake .. -DMGE_WITH_TEST=ON -DMGE_WITH_CUDA=ON

   编译：

   .. code:: bash

      rlaunch --cpu=40 --memory=50000 -- make -j40
      make develop

   因为运行 ``CUDA`` 测试需要 ``gpu``\ ，如果 ``workspace`` 上没有
   ``gpu`` 的话，可以用 ``rlaunch`` 临时申请一张：

   .. code:: bash

      rlaunch --gpu=1 --cpu=1 --memory=50000 -- bash

   在 ``build`` 目录下运行测试：

   .. code:: bash

      ./dnn/test/megdnn_test --gtest_filter="CUDA.ADD:CUDA.ADD"

   得到如下结果，说明测试通过：

   .. figure:: ../../_static/images/cuda-add.png


反向算子与之类似，在这里不再详细介绍。

在 MegBrain 添加一个 op
~~~~~~~~~~~~~~~~~~~~~~~~~~~

有了 ``kernel`` 之后，我们需要在 ``MegBrain`` 层添加算子的静态图实现。

这一节我们介绍如何在 ``MegBrain`` 层添加一个 ``opr``\ 。

1. 在 ``/src/opr/include/megbrain/opr`` 文件夹下，添加 ``opr`` 的声明。

   这个文件夹下是按照 ``opr`` 的功能分类的，比如 ``basic_arith.h``
   存放着\ ``elemwise``\ ，\ ``typecvt`` 等基础的算术运算的
   ``opr``\ ，\ ``imgproc.h`` 存放着 ``warp_affine``, ``resize``
   等图像处理 ``opr``\ ， 这里我们将 ``Add`` 放到 ``misc.h`` 里面。

   .. code:: cpp

      MGB_DEFINE_OPR_CLASS(
              AddForward, intl::MegDNNOprWrapperFwd<megdnn::AddForward>) // {
          void scn_do_execute() override;
          void init_output_static_infer_desc() override;
          void add_input_layout_constraint() override;

      public:
          MGE_WIN_DECLSPEC_FUC AddForward(
                  VarNode* data1, VarNode* data2, const Param& param,
                  const OperatorNodeConfig& config);

          MGE_WIN_DECLSPEC_FUC static SymbolVar make(
                  SymbolVar data1, SymbolVar data2, const Param& param,
                  const OperatorNodeConfig& config = {});
      };
      using Add = AddForward;

   每个 ``MegBrain`` 的 ``Operator`` 都继承自
   ``mgb::cg::OperatorNodeBase``\ ，对于是否有\ ``megdnn opr``\ ，是否可以根据
   ``input`` 推导 ``output shape``\ 等，\ ``MegBrain``
   定义了很多\ ``Mixin``, 你可以让你的 ``Operator`` 直接继承自
   ``MegBrain`` 提供的一些基类，来简化你的代码。例如：这里是继承自
   ``intl::MegDNNOprWrapperFwd``\ 。基类选择的一个依据是看其接口实现是否符合需要，可以参考与你要实现的
   ``opr`` 类似的 ``opr`` 的基类选择。

   这里的 ``scn_do_execute`` 方法是调用 ``MegDNN`` 层的 ``kernel``
   完成计算；\ ``init_output_static_infer_desc``
   的作用是初始化输出的一些信息，比如是否可以推导
   ``shape``\ 、是否可以推导 ``workspace``
   等；\ ``add_input_layout_constraint`` 是获取输入的 ``layout``
   的限制。

2. 在 ``src/opr/impl/misc.cpp`` 文件中添加这个 ``opr`` 的实现。

   .. code:: cpp

      /* ================= Add =================  */

      MGB_DYN_TYPE_OBJ_FINAL_IMPL(AddForward);

      AddForward::AddForward(
              VarNode* data1, VarNode* data2, const Param& param,
              const OperatorNodeConfig& config)
              : Super(data1->owner_graph(), config, "add", {data1, data2}) {
          init_megdnn_opr(*this, param);
          mgb_assert(data1->shape().eq_shape(data2->shape()));
          add_input({data1, data2});
          output(0)->dtype(data1->dtype());
      }

      SymbolVar AddForward::make(
              SymbolVar data1, SymbolVar data2, const Param& param,
              const OperatorNodeConfig& config) {
          return data1.insert_single_output_opr<AddForward>(
                  data1.node(), data2.node(), param, config);
      }

      void AddForward::init_output_static_infer_desc() {
          using namespace cg::static_infer;
          auto infer_shape = [](TensorShape& dest, const InpVal& iv) {
              auto ishp = iv.val[0].shape();
              dest = ishp;
              return true;
          };
          owner_graph()->static_infer_manager().register_shape_infer(
                  output(0), {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});

          auto infer_workspace = [this](TensorShape& dest, const InpVal& iv) {
              auto data_dtype = input(0)->dtype();
              auto data_shape = iv.val[0].shape();
              TensorLayout data_layout(data_shape, data_dtype);
              dest.ndim = 1;
              dest[0] = megdnn_opr()->get_workspace_in_bytes(
                      data_layout, data_layout, data_layout);
              return true;
          };
          owner_graph()->static_infer_manager().register_shape_infer(
                  output(1), {SourceType::DEP,
                              {{input(0), DepType::SHAPE}, {input(1), DepType::SHAPE}},
                              infer_workspace});
      }

      void AddForward::add_input_layout_constraint() {
          mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
      }

      void AddForward::scn_do_execute() {
          megdnn_opr()->exec(
                  input(0)->dev_tensor().as_megdnn(), input(1)->dev_tensor().as_megdnn(),
                  output(0)->dev_tensor().as_megdnn(),
                  intl::get_megdnn_workspace_from_var(output().back()));
      }

   ``AddForward`` 是构造函数，这里对输入的 ``shape``
   做了判断，\ ``add_input({data1, data2});`` 表明静态图中有 ``data1``
   和 ``data2`` 这两个 ``tensor``
   作为输入，\ ``output(0)->dtype(data1->dtype());`` 表明输出和
   ``data1`` 的数据类型相同，这里取 ``output(0)`` 是因为 ``output``
   可能有多个，它们是用一个链表存储的，这里只有一个输出，所以是
   ``output(0)``\ 。

   从 ``AddForward::make`` 的返回值可以看出来，它的功能是构造一个
   ``SymbolVar``\ ，也就是构造这个算子的输出。

   函数 ``init_output_static_infer_desc`` 这里推导了输出的 ``shape`` 和
   ``workspace``\ 。

   函数 ``scn_do_execute`` 通过 ``megdnn_opr()->exec()`` 将输入传到
   ``MegDNN`` 层的 ``kernel`` 执行。

3. 在 ``src/opr/impl/misc.sereg.h`` 中添加序列化 ``Opr`` 的代码。

   .. code:: cpp

      MGB_SEREG_OPR(Add, 2);

   这里 ``2`` 表示这个 ``opr`` 接收两个输入；\ ``nullptr`` 表示这个
   ``opr`` 无需转换成其他 ``opr``\ （有时为了模型兼容性部分 ``opr``
   会转换成其他 ``opr``\ ）。

4. 在 ``src/serialization/impl/schema.fbs`` 中添加对应的
   ``opr param``\ ，新加算子应该放在 ``union``
   的最后，并且数字是递增的。

   .. code:: cpp

      param.Add = 93,

5. 添加测试。

   在 ``src/opr/test/misc.cpp`` 中添加测试用例：

   .. code:: cpp

      TEST(TestOprMisc, Add) {
          auto graph = ComputingGraph::make();
          HostTensorGenerator<> gen_data{0, 1000};
          auto host_data1 = gen_data({2, 2, 2}), host_data2 = gen_data({2, 2, 2});
          opr::AddForward::Param param{2, 3, 4};
          auto data1 = opr::Host2DeviceCopy::make(*graph, host_data1),
               data2 = opr::Host2DeviceCopy::make(*graph, host_data2),
               dst = opr::Add::make(data1, data2, param);

          HostTensorND host_dst;
          auto func = graph->compile({make_callback_copy(dst, host_dst)});
          func->execute();

          auto pdata1 = host_data1->ptr<float>();
          auto pdata2 = host_data2->ptr<float>();
          auto pdst = host_dst.ptr<float>();
          for (size_t i = 0; i < host_data1->layout().total_nr_elems(); ++i) {
              ASSERT_EQ(param.x * pdata1[i] + param.y * pdata2[i] + param.z, pdst[i]);
          }
      }

   这个 ``TestCase`` 生成了两个输入 ``host_data1`` 和
   ``host_data2``\ ，它们的 ``shape`` 都是 ``{2, 2, 2}``\ 。

   ``opr::AddForward::Param param{2, 3, 4};`` 设置了 ``AddForward``
   的三个 ``param`` （分别是 ``x``\ 、\ ``y``\ 、\ ``z``\ ）。

   ``opr::Host2DeviceCopy::make`` 负责把 ``host`` 上的输出复制到
   ``device`` 上（计算大多在 ``device`` 上进行）。

   最后根据
   ``ASSERT_EQ(param.x * pdata1[i] + param.y * pdata2[i] + param.z, pdst[i]);``
   判断输出是否与期望结果相符。

6. 编译。

   在 ``build`` 目录下执行：

   .. code:: bash

      cmake .. -DMGE_WITH_TEST=ON
      rlaunch --cpu=40 --memory=50000 -- make -j40
      make develop

7. 运行测试。

   在 ``build`` 目录下执行：

   .. code:: bash

      ./test/megbrain_test --gtest_filter="TestOprMisc.Add*"

   得到如下结果，说明测试通过：

   .. figure:: ../../_static/images/megbrain-add.png


用同样的方式可以添加反向算子以及对应的测试。

在 Imperative 添加一个 op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

接下来我们在 ``imperative`` 层添加算子的动态图实现。

1. 首先在 ``src/core/include/megbrain/ir/ops.td`` 添加算子的声明：

   .. code:: cpp

      def Add: MgbHashableOp<"Add", [AddParam]>;

   这里我们声明了一个 ``imperative`` 算子
   ``Add``\ ，并且这个算子是有参数的（\ ``AddParam``\ ）。如果我们要声明一个没有参数的算子，则在该位置填上
   ``EmptyParam``\ 。

2. 在 ``imperative/src/impl/ops`` 下新建文件 ``add.cpp``\ ，添加算子的
   ``imperative`` 实现。

   这里可以根据需要实现
   ``apply_on_var_node``\ 、\ ``apply_on_physical_tensor`` 等方法，其中
   ``apply_on_var_node`` 是必须要实现的。

   .. code:: cpp

      #include "../dnn_op_helper.h"
      #include "megbrain/imperative/ops/autogen.h"

      #include "../op_trait.h"

      #include "megbrain/opr/misc.h"
      #include "megdnn/oprs/general.h"

      namespace mgb {
      namespace imperative {

      namespace {
      namespace add {

      std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
              const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
          auto&& op = def.cast_final_safe<Add>();
          mgb_assert(input_descs.size() == 2, "Add expects two inputs");
          auto comp_node = input_descs[0].comp_node;
          TensorLayout data1 = input_descs[0].layout, data2 = input_descs[1].layout;

          if (!data1.ndim) {
              return {{{data1, comp_node}}, false};
          }
          if (!data2.ndim) {
              return {{{data2, comp_node}}, false};
          }

          mgb_assert(
                  data1.dtype == dtype::Float32() || data1.dtype == dtype::Int32(),
                  "data1 dtype must be float32 or int32");
          mgb_assert(
                  data2.dtype == dtype::Float32() || data2.dtype == dtype::Int32(),
                  "data2 dtype must be float32 or int32");

          mgb_assert(data1.is_contiguous(), "data1 should be contiguous");
          mgb_assert(data2.is_contiguous(), "data2 should be contiguous");

          mgb_assert(data1.eq_layout(data2), "data1 layout doesn't match data2");

          TensorLayout dst = data1;
          dst.init_contiguous_stride();
          return {{{dst, comp_node}}, true};
      }

      auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
          auto&& op = static_cast<const Add&>(def);
          mgb_assert(inputs.size() == 2);
          OperatorNodeConfig config{op.make_name()};
          return opr::Add::make(inputs[0], inputs[1], op.param(), config);
      }

      OP_TRAIT_REG(Add, Add)
              .infer_output_attrs_fallible(infer_output_attrs_fallible)
              .apply_on_var_node(apply_on_var_node)
              .fallback();
      }  // namespace add

      }  // anonymous namespace
      }  // namespace imperative
      }  // namespace mgb

3. 在 ``imperative/python/megengine/functional`` 的对应位置添加算子的
   ``python`` 实现。

   这里我们选择新建文件 ``add.py``\ ，通过 ``builtin`` 在 ``functional``
   调用 ``imperative`` 算子。

   .. code:: python

      # pylint: disable=redefined-builtin
      from typing import Sequence

      from ..core._imperative_rt.core2 import apply
      from ..core.ops import builtin
      from ..tensor import Tensor


      def add_example(data1: Tensor, data2: Tensor, x: int=1, y: int=1, z: int=0) -> Tensor:
          add = builtin.Add(x, y, z)
          return apply(add, data1, data2)[0]

   ``x``\ 、\ ``y``\ 、\ ``z`` 是 ``imperative`` 算子的参数，在
   ``builtin`` 的时候填入。返回的结果是一个链表，因此我们通过下标
   ``[0]`` 取第一个结果。

4. 在 ``imperative/python/megengine/functional/__init__.py`` 下
   ``import`` 算子的 ``python`` 定义。

   .. code:: python

      from .add import *

5. （可选）有时候会需要将算子封装为一个 ``module``
   的形式进行调用，可以在 ``imperative/python/megengine/module``
   下添加对应的 ``module`` 声明。

   这里我们新建一个文件 ``add.py``\ ：

   .. code:: python

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
              self.x = x
              self.y = y
              self.z = z

          def forward(self, data1, data2):
              return add(data1, data2, self.x, self.y, self.z)

   这个 ``module`` 的前向逻辑只包含了一个 ``add`` 算子的调用。

   注意，还需要在 ``imperative/python/megegnine/module/__init__.py`` 下
   ``import`` 这个 ``module`` 才可以使用：

   .. code:: python

      from .add import AddExample

6. 添加测试。

   在 ``imperative/python/test/unit/functional/test_functional.py``
   下添加 ``imperative`` 层的测试。

   .. code:: python

      def test_add():
          data1 = tensor(np.array([[1, 2], [0, 4]], dtype=np.float32))
          data2 = tensor(np.array([[4, 6], [7, 3]], dtype=np.float32))

          expected_result = np.array([[6, 2], [-5, 17]], dtype=np.float32)

          actual_result = F.add_example(data1, data2, x=2, y=-3, z=16)[0]
          np.testing.assert_equal(
              expected_result, expected_result
          )

7. 验证测试结果。

   在 ``build`` 目录下，编译之后运行测试：

   .. code:: bash

      cmake .. -DMGE_WITH_TEST=ON
      rlaunch --cpu=40 --memory=50000 -- make -j40
      make develop
      python3 -m pytest -sv ../imperative/python/test/unit/functional/test_functional.py::test_add

   ``pytest`` 后面通过相对路径找到对应的测试文件的测试用例。

   得到如下结果，说明测试通过：

   .. figure:: ../../_static/images/imperative-add.png


通过同样的方式，可以为反向算子添加 ``imperative`` 实现。
