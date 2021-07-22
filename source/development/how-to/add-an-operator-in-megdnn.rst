.. _add-an-operator-in-megdnn:

========================
如何在 MegDNN 中添加算子
========================

MegDNN 是算子（Operator）级别的 DNN 库，存放于 :src:`dnn` 下，
类似于 MKL-DNN、OpenBlas 等库，提供了包括卷积、池化、矩阵乘法、点乘等在内的基本算术运算和神经网络原语。

.. panels::
   :container: +full-width text-center
   :card:

   对使用者可见的统一接口
   ^^^^^^^^^^^^^^^^^^^^^^
   .. mermaid::

      graph TB
         DNN[MegDNN API] --> x86 & ARMv7 & AArch64 & CUDA & ...
         style DNN fill:#f9f,stroke:#333,stroke-width:4px

   MegDNN 为上层提供了对使用者可见的统一接口，
   而在底层针对 CUDA、ARM、x86 等不同平台做了针对性实现，是一个跨平台的计算库。
   ---
   MegDNN API 内部实现
   ^^^^^^^^^^^^^^^^^^^
   .. mermaid::

      classDiagram
          class OperatorBase {
          +OperatorBase(Handle* handle) : m_handle(handle)
          +Handle* handle() const
          +bool is_thread_safe() const
          +void set_error_tracker(void*)
          -Handle* m_handle
          }

          class OperatorA {
          +virtual void exec(...)
          +void deduce_layout(...)
          +virtual size_t get_workspace_in_bytes(...)
          #void check_exec(...)
          }

          class OperatorB {
          +virtual void exec(...)
          +void deduce_layout(...)
          +virtual size_t get_workspace_in_bytes(...)
          #void check_exec(...)
          }

          class OperatorX {
          ...
          }

          OperatorBase <|-- OperatorA
          OperatorBase <|-- OperatorB
          OperatorBase <|-- OperatorX

MegDNN 组织架构
---------------
.. panels::
   :footer: text-center
   :card:

   .. mermaid::

      graph TB
         IC[Interface Class] --> common
         common --> naive & cuda & ...
         naive --> fallback
         fallback --> arm_common & x86
         arm_common --> armv7 & aarch64

         style IC fill:#f9f,stroke:#333,stroke-width:4px
         style common fill:#888
         style naive fill:#888
         style fallback fill:#888
         style arm_common fill:#888

    +++++++++++++++++++++++++++++++++++
    以上文件位于 :src:`dnn/src` 目录内。
    ---

    Interface Class
      定义算子的接口

    common
      定义各个平台的公共代码，比如各个算子的 

      * ``deduce_layout`` 方法
      * ``OperatorBase`` 方法
      * ``Handle`` 方法

    naive
      简单粗暴的算子实现，不考虑性能、内存效率等，仅仅用于正确性验证。

    fallback
      通用的代码实现，适合那些使用 SSE 或 NEON 等技术后并没有加速效果的算子。

    arm_common
      针对 ARMv7  和 AArch64 的通用代码优化。

    armv7/aarch64/x86/CUDA/...
      针对特定平台的代码优化，可能包含底层汇编代码等。

   +++++++++++++++++++++++++++++++++++++++++++++++
   更详细的说明请参考 :ref:`megdnn-organize` 。

.. note::

   MegDNN 中对于同一算子可能包含不同的算法实现。
   例如卷积实现，在 MegDNN 中包含直接卷积、im2col-gemm、winograd、FFT 等。
   因此在 MegDNN 中，可以采用指定或者启发式方法选择算法。

如何添加一个算子23
----------------

以添加一个 ``C = A + B + m`` 加法算子 ``Add`` 的 naive 的实现为例。

其中 A、B 为输入 tensor, C 为输出 tensor, 而 m 为输入参数。

添加参数定义
~~~~~~~~~~~~

在 :src:`dnn/scripts/opr_param_defs.py` 中添加相关参数的定义：

.. code-block:: python

   (pdef('Add').
   add_fields('int32', Doc('m', 'param of Add opr'), '3')
   )

这段代码的作用是：添加了名为 ``Add`` 的 算子，输入参数为 ``m``, 默认值为 3.

.. note::

   ``add_fields`` 目前支持的数据类型可参考 :src:`dnn/scripts/gen_param_defs.py` .

添加算子定义
~~~~~~~~~~~~

在 :src:`dnn/include/megdnn/oprs/` 的对应文件中定义这个算子（假设定义在 ``general.h`` 中）：

.. code-block:: cpp

   class AddForward : public OperatorBase {
         DEF_OPR_PARAM(Add);
         DEF_OPR_IMPL(AddForward, OperatorBase, 2, 1);

   public:
       virtual void exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                         _megdnn_tensor_out C, _megdnn_workspace workspace) = 0;
       void deduce_layout(const TensorLayout& A, const TensorLayout& B,
                          TensorLayout& C);
       virtual size_t get_workspace_in_bytes(const TensorLayout& A,
                                             const TensorLayout& B,
                                             const TensorLayout& C) = 0;
    
   protected:
       void check_exec(const TensorLayout& A, const TensorLayout& B,
                       const TensorLayout& C, size_t workspace_in_bytes);
   };
   using Add = AddForward;

.. note::

   此处 ``exec`` 为包含计算逻辑的接口，接受输入 ``A``, ``B`` 输出 ``C`` 以及 ``workspace``.
   其中 ``workspace`` 表明计算中需要使用的临时空间大小，是一个字节数组。

添加 common 定义
~~~~~~~~~~~~~~~~

在 :src:`dnn/src/common` 中添加所有平台的共有类，在该目录下创建 ``add.cpp`` 文件。

接着实现刚才在 ``include/megdnn/oprs/`` 定义的虚函数，
比如 ``deduce_layout_fwd`` 推断前向结果时的 layout.

.. code-block:: cpp
   :caption: dnn/src/common/add.cpp

   #include "megdnn/oprs.h"
   #include "src/common/utils.h"
    
   namespace megdnn {
    
   void Add::deduce_layout(const TensorLayout& A, const TensorLayout& B,
                           TensorLayout& C) {
       megdnn_assert(A.ndim == 3);
       size_t in = A.shape[0];
       size_t ih = A.shape[1];
       size_t iw = A.shape[2];
       megdnn_assert_eq_layout(A, B);
       megdnn_assert_eq_dtype(A, B);
       C = TensorLayout(TensorShape({in, ih, iw}), A.dtype);
   }
   void Add::check_exec(const TensorLayout& A, const TensorLayout& B,
                        const TensorLayout& C, size_t workspace_in_bytes) {
       TensorLayout C_expected;
       megdnn_assert_eq_dtype(A, C);
       megdnn_assert_eq_dtype(B, C);
       deduce_layout(A, B, C_expected);
       megdnn_assert_eq_layout(C_expected, C);
    
       auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
       megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
   }
    
   }  // namespace megdnn

添加 common callback
~~~~~~~~~~~~~~~~~~~~

在 :src:`dnn/src/common/handle_impl.h` 文件的 ``#define MEGDNN_FOREACH_OPR_CLASS(cb)`` 中添加 ``cb(Add)``.
这样 common 下面的各个平台中可以 include 这个文件，定义算子需要使用的通用功能。

.. seealso::

   参考 :src:`dnn/src/common/handle.cpp` 

添加平台实现
~~~~~~~~~~~~

接下来定义各个平台的函数，以 naive 版本为例：

在 :src:`dnn/src/naive/` 中创建文件夹 ``add``, 在其中实现以下文件：

.. code-block:: cpp
   :caption: opr_impl.h

   namespace megdnn {
   namespace naive {
 
   class AddImpl: public Add {
       public:
           using Add::Add;
           void exec(_megdnn_tensor_in A,
                   _megdnn_tensor_in B,
                   _megdnn_tensor_out C,
                   _megdnn_workspace workspace) override;
           size_t get_workspace_in_bytes(const TensorLayout &,
                   const TensorLayout &,
                   const TensorLayout &) override {
               return 0;
           }
   };
 
   } // namespace naive
   } // namespace megdnn

.. code-block:: cpp
   :caption: opr_impl.cpp

   #include "src/naive/add/opr_impl.h"

   #include "src/common/utils.h"
   #include "src/naive/handle.h"

   namespace {
   template <typename T>
   void exec_internal(const T * __restrict A,
           const T * __restrict B,
           T * __restrict C,
           int m,
           size_t n) MEGDNN_NOEXCEPT
   {
       rep(i, n) {
           C[i] = A[i] + B[i] + m;
       }
   }
    
   } // anonymous namespace
   namespace megdnn {
   namespace naive {
    
   void AddImpl::exec(_megdnn_tensor_in A,
           _megdnn_tensor_in B,
           _megdnn_tensor_out C,
           _megdnn_workspace workspace)
   {
       check_exec(A.layout, B.layout, C.layout, workspace.size);
       auto n = A.layout.total_nr_elems();
   #define cb(DType) \
       if (A.layout.dtype == DType()) { \
           using T = typename DTypeTrait<DType>::ctype; \
           MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<T>(A.ptr<T>(), \
                   B.ptr<T>(), \
                   C.ptr<T>(), param().m, n)); \
           return; \
       }
       MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
   #undef cb
   }
   } // namespace megdnn


功能很简单：在 ``exec`` 中写实际的执行，每种 type 都生成一个执行，
然后通过 MEGDNN_DISPATCH_CPU_KERN_OPR 把将算子的执行 kernel 放到 handle 上执行。

添加 Handle
~~~~~~~~~~~

最后，在 :src:`dnn/src/naive/handle.cpp` 头部添加 ``#include "src/naive/add/opr_impl.h"``,
里面调用了之前 ``handle_impl.cpp`` 中定义的宏。

.. note::

   如果是 CUDA 平台，则是在 :src:`dnn/src/cuda/handle_create.cpp` 中添加。

为 ``HandleImpl`` 添加函数 ``create_opr<>``, 通过这个函数，创建平台的算子。

实际操作为：在 ``handle_impl.h`` 里添加 ``cb(Add)``.

至此， ``Add`` 算子创建完成。

添加测试
--------

下面为上面的 ``Add`` 算子添加测试。

.. note::

   * 一般 naive 实现不必测试，此处只是展现下。
   * 在提供其他平台如 arm/x86/cuda 的实现时，测试结果应当和 naive 版本保持一致。

在 :src:`dnn/test/common/` 下创建 ``add.h``, 其中构造所需要测试的参数：

.. code-block:: cpp

   #pragma once
   #include "megdnn/opr_param_defs.h"
   #include "megdnn/basic_types.h"
   #include <iostream>
    
   namespace megdnn {
   namespace test {
    
   namespace add {
   struct TestArg {
       param::Add param;
       TensorShape src;
       TestArg(param::Add param, TensorShape src):
           param(param), src(src)
       {}
   };
    
   inline std::vector<TestArg> get_args() {
       std::vector<TestArg> args;
    
       param::Add cur_param;
       cur_param.m = 10;
       args.emplace_back(cur_param, TensorShape{1, 8, 8});
       return args;
   }
    
   } // namespace add
   } // namespace test
   } // namespace megdnn

修改 :src:`dnn/src/common/opr_trait.h`, 添加该算子的 traits:

.. code-block:: cpp

   DEF(Add, 3, true, true);

其中 3 表示其有 3 个参数（假设分别是 A, B, C ），
第三个参数表示是否需要有 workspace, 第四个参数表示是否可以 deduce_layout. 
其描述这个算子，为 ``exec_proxy.h`` 和 ``deduce_layout_proxy.h`` 导向正确的实现。

在 :src:`dnn/test/naive` 下创建 ``add.cpp`` 文件，添加详细的测试代码。

* 其中 ``ADD`` 这个测试用例本质上没有额外的意义，因为目前只有 naive 实现，
  如果有其它后端的优化实现，可以基于 ``Checker`` 类来做正确性验证。
* ``ADD2`` 这个测试用例基于用户指定的输入输出来验证结果，
  即用户跑指定输入，然后得到的输出与用户给定的输出对比，一般用来检测 naive 正确性。

.. code-block:: cpp

   #include "test/common/add.h"
   #include "megdnn/dtype.h"
   #include "megdnn/oprs.h"
   #include "test/common/checker.h"
   #include "test/naive/fixture.h"
    
   namespace megdnn {
   namespace test {
    
   TEST_F(NAIVE, ADD) {
       std::vector<add::TestArg> args = add::get_args();
       Checker<Add> checker(handle());
       for (auto&& arg : args) {
           checker.set_param(arg.param)
                   .set_dtype(0, dtype::Float32())
                   .set_dtype(1, dtype::Float32())
                   .set_dtype(2, dtype::Float32())
                   .execs({arg.src, arg.src, {}});
       }
   }
    
   TEST_F(NAIVE, ADD2) {
       Checker<Add> checker(handle(), false);
       Add::Param param;
       param.m = 3;
       checker.set_param(param).exect(
               Testcase{TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4}),
                        TensorValue({1, 2, 2}, dtype::Float32(), {2, 1, 3, 5}),
                        {}},
               Testcase{{},
                        {},
                        TensorValue({1, 2, 2}, dtype::Float32(), {6, 6, 9, 12})});
   }
    
   }  // namespace test
   }  // namespace megdnn

.. note::

   其它 device 的测试（比如 CUDA）可以直接用 ``execs`` （而非 ``exect`` ）
   测试指定 shape 输入的结果，会自动和 naive 的结果做对比。

编译和测试
----------

.. warning::

   测试带 CUDA 后端的算子时，需要注意是否编译了对应显卡的代码，
   显卡型号对应编译选项可以通过脚本 :src:`third_party/getcudacap.sh` 来获取 CUDA 信息。

.. note::

   编译方法请参考 :ref:`install` 。

我们需要在跑 CMake 命令时设置 ``MGE_WITH_TEST=ON`` 以支持测试。

执行测试
~~~~~~~~

.. code-block:: shell

   $ ../../dnn/test/megdnn_test --gtest_filter="NAIVE.ADD:NAIVE.ADD2"
   Note: Google Test filter = NAIVE.ADD:NAIVE.ADD2
   [==========] Running 2 tests from 1 test case.
   [----------] Global test environment set-up.
   [----------] 2 tests from NAIVE
   [ RUN      ] NAIVE.ADD
   [       OK ] NAIVE.ADD (0 ms)
   [ RUN      ] NAIVE.ADD2
   [       OK ] NAIVE.ADD2 (0 ms)
   [----------] 2 tests from NAIVE (0 ms total)
    
   [----------] Global test environment tear-down
   [==========] 2 tests from 1 test case ran. (1 ms total)
   [  PASSED  ] 2 tests.

常见问题
--------

.. dropdown:: 精度对不齐怎么办？

   这里的精度指要测试的 backend(cuda, arm, OpenCL) 等和 naive 结果精度对不上，目前默认的误差是 1e-4.
   常见原因：

   * cuda 的 float 精度计算与 naive 有误差，多次累加容易导致误差逐步放大
   * int8 round 的问题：比如 armv7 上的除法是牛顿除法，1/2 的结果不是 0.5, 而是 0.49998 等；
     cuda 上也常见这些问题，导致在 round 时 naive 向上 round, arm 或 cuda 向下 round, 结果相差 1.
     这种误差暂时没找到很好的解决办法，只能调大 epsilon.
     对于量化类型，我们需要保证计算结果是无偏的即可，可以通过 ``checker.set_max_avg_biased_error`` 来调节。

   调试方法：

   * 选择一个最小复现，根据 test 的 index 来选择性打印一些中间结果来看。

.. dropdown:: 出现错误 ``assertion 'm_nr_call && "cpu dispatch must be called"' failed`` 怎么办？

  由于对于 CPU 后端，MegDNN 直接通过 ``handle`` 的 ``dispatch`` 接口将具体描述 kernel 语义的 ``function`` 对象发给 Graph,
  也就是上面的 ``exec_internal<float>`` 函数指针；有时候可能忘记在调用的地方包宏 ``MEGDNN_DISPATCH_CPU_KERN_OPR``,
  导致这个 kernel 未发给 Graph, 这种错误无法在 MegDNN 测出来，因为 MegDNN 默认的 ``dispatcher`` 是 inplace 的，
  而且这种错误是某种 race, 可能导致随机错误，所以 MegDNN 定义 ``CPUDispatchChecker`` 来检测这个行为。

.. dropdown:: 出现错误 ``mgb::MegDNNError cuda error invalid device function(98) occurred;`` 怎么办；

  这种错误是对应设备的 device 函数不存在，一般是你没有编译测试用的 GPU 对应的代码导致的。
  可以参考编译与测试的注意事项，编译对应 GPU 的代码，重跑 megdnn_test.
