.. _add-an-operator-in-graph-runtime:

===============================
如何在 Graph Runtime 中添加算子
===============================
Graph Runtime 算子与 MegDNN 算子的区别之处在于它维护了内存资源，
是 Graph 的一个节点，而 MegDNN 算子只是描述计算语义；
对于有的 Graph Runtime 算子而言，它将 MegDNN 的算子包装了一层，
同时隐去后端硬件信息，继承通用的 ``MegDNNOprWrapperFwd`` 的接口类。

如何添加一个算子
----------------

我们以添加一个 ``Add`` 算子为例。

添加算子定义
~~~~~~~~~~~~

在 :src:`src/opr/include/megbrain/opr` 文件夹下，添加对应算子声明。

.. note::

   里面有一些头文件，按照分类将一些算子组织到一起：

   * 比如 ``basic_arith.h`` 存放着 ``elemwise``, ``typecvt`` 等基础算术运算算子
   * 又比如 ``imgproc.h`` 存放着 ``warp_affine``, ``resize`` 等图像处理算子
   * 作为举例，这里我们将 ``Add`` 放到 ``misc.h`` 里面

.. code-block:: cpp
   :caption: src/opr/include/megbrain/opr/misc.h

   MGB_DEFINE_OPR_CLASS(AddForward,
                        intl::MegDNNOprWrapperFwd<megdnn::AddForward>) // {
   public:
       AddForward(VarNode* a, VarNode* b, const Param& param,
                  const OperatorNodeConfig& config);
    
       static SymbolVar make(SymbolVar a, SymbolVar b,
                             const Param& param = {},
                             const OperatorNodeConfig& config = {});
   };
   using Add = AddForward;

* 每个 Graph Runtime 的 ``Operator`` 都继承自 ``mgb::cg::OperatorNodeBase``.
* 对于 “是否有 MegDNN 算子”，“是否可以根据 input 推导 output shape” 等情况，Graph Runtime 定义了很多 ``Mixin``.
  你可以让你的 Operator 直接继承自 Graph Runtime 提供的一些基类，来简化你的代码。
  例如上面的代码继承自 ``intl::MegDNNOprWrapperFwd``. 
* 你可以仔细阅读 :src:`src/opr/include/megbrain/opr/internal` 中的代码，选择一个合适的基类，
  然后覆盖基类的虚函数，实现你的 Operator 完整的功能。

添加算子实现
~~~~~~~~~~~~

在 :src:`src/opr/impl/misc.cpp` 文件下（与上一步的 ``misc.h`` 对应），添加这个算子的实现：

.. code-block:: cpp

   /* ================= Add =================  */
   MGB_DYN_TYPE_OBJ_FINAL_IMPL(AddForward);
   AddForward::AddForward(VarNode* a, VarNode* b,
                          const Param& param,
                          const OperatorNodeConfig& config)
           : Super{a->owner_graph(), config, "add", {a, b}} {
       init_megdnn_opr(*this, param);
       mgb_assert(a->shape().eq_shape(b->shape()));
       add_input({a, b});
       output(0)->dtype(a->dtype())
   }
    
   SymbolVar AddForward::make(SymbolVar a, SymbolVar b,
                              const Param& param,
                              const OperatorNodeConfig& config) {
       return a.insert_single_output_opr<AddForward>(a.node(), b.node(), param,
                                                     config);
   }

.. note::

   ``MGB_DYN_TYPE_OBJ_FINAL_IMPL`` 用来实现 RTTI 的功能，用以记录算子的类型信息。
   因为这里的 ``Add`` 继承自 ``intl::MegDNNOprWrapperFwd``, 大部分接口都在这个基类中定义了。

按需支持求导
~~~~~~~~~~~~

如果你的算子需要支持求导，那么需要添加这个算子的导数，还需要为它添加求导。

* 对于 ``Add`` 算子来讲，它的导数很简单，就是 ``grad``, 其中 ``wrt_idx`` 表示第几个输入的导数。
* 如果对于比较复杂的算子如 ``Conv``, 它需要定义 ``convforward``, ``convbackwarddata``, ``convbackwardfilter`` 算子，
  需要分别在 MegDNN 和 Graph Runtime 中添加。

对于 ``Add`` 算子而言，在 :src:`src/opr/impl/misc.cpp` 文件下，添加导数：

.. code-block:: cpp

   #if MGB_ENABLE_GRAD
   MGB_IMPL_OPR_GRAD(Add) {
       MGB_MARK_USED_VAR(opr);
       MGB_MARK_USED_VAR(wrt_idx);
       SymbolVar og{out_grad.at(0)};
       return og.node();
   }
   #endif

.. seealso::

   关于宏 ``MGB_IMPL_OPR_GRAD`` 可以阅读 :src:`src/core/include/megbrain/graph/grad_impl.h#L167` 。

添加序列化功能
~~~~~~~~~~~~~~

在 :src:`src/opr/impl/misc.sereg.h` 文件中添加算子序列化代码。

* 对于 ``Add`` 算子而言，只需要序列化 Param, 所以可以直接用一行代码；
* 如果需要序列化额外信息，需要特化 ``OprLoadDumpImpl`` 或者 ``OprMaker`` .

.. code-block:: cpp

   MGB_SEREG_OPR(Add, 2);

同时需要在 :src:`src/serialization/impl/schema.fbs` 中添加对应算子的 Param:

.. code-block::

   param.Add = 75， 

.. note::

   这个值是递增的，新加算子需要加到 union ``OperatorParam`` 最底部

添加测试
--------

Graph Runtime 提供了 ``AutoOprChecker`` 来实现算子的测试。

.. seealso::

   参考 :src:`test/src/autocheck.cpp` 中的实现。

我们可以参照已有测试的实现（如 :src:`src/opr/test/dnn/roi_align.cpp` ），
在 :src:`src/opr/test/misc.cpp` 文件夹下，为 ``Add`` 添加测试：

* 第一个测试是单纯测试 ``Add`` 算子的正确性；
* 第二个测试会测求导，组一个子图加 loss 来求 delta y / delta x,
  即将数学定义作为正确结果，将 MegDNN 的 backward 算子的结果作为被比较对象，具体见 ``AutoOprChecker`` 中的代码

.. code-block:: cpp

   TEST(TestOprMisc, Add) {
       auto graph = ComputingGraph::make();
       HostTensorGenerator<> gen{0, 1000};
       opr::AddForward::Param param{3};
       auto host_a = gen({2, 2, 2}), host_b = gen({2, 2, 2});
       auto a = opr::Host2DeviceCopy::make(*graph, host_a),
            b = opr::Host2DeviceCopy::make(*graph, host_b),
            c = opr::Add::make(a, b, param);
       HostTensorND host_c;
       auto func = graph->compile({make_callback_copy(c, host_c)});
       func->execute();
    
       auto pa = host_a->ptr<float>();
       auto pb = host_b->ptr<float>();
       auto pc = host_c.ptr<float>();
       for (size_t i = 0; i < host_a->layout().total_nr_elems(); ++i) {
           ASSERT_EQ(pa[i] + pb[i] + param.m, pc[i]);
       }
   }
    
   TEST(TestOprMisc, Add2) {
       using Checker = AutoOprChecker<2, 1>;
       opr::AddForward::Param param{3};
       auto make_graph =
               [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
           mgb_assert(inputs.size() == 2);
           return {opr::Add::make(inputs[0], inputs[1], param)};
       };
       auto fwd = [param](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
           TensorShape oshp = inp[0]->shape();
           auto pa = inp[0]->ptr<float>();
           auto pb = inp[1]->ptr<float>();
           auto pc = dest[0].resize(oshp).ptr<float>();
    
           for (size_t i = 0; i < oshp.total_nr_elems(); ++i) {
               pc[i] = pa[i] + pb[i] + param.m;
           }
       };
       Checker checker{make_graph, fwd};
       checker.run({TensorShape{2, 2, 2}, TensorShape{2, 2, 2}})
               .run({TensorShape{2, 2, 2}, TensorShape{2, 2, 2}})
               .run({TensorShape{2, 2, 2}, TensorShape{2, 2, 2}});
   }

编译和测试
----------

跑 CMake 命令时设置 ``MGE_WITH_TEST=ON`` 以支持 Graph Runtime 与 MegDNN 测试。

.. note::

   编译方法请参考 :ref:`install` 。


.. code-block:: shell

    $ ../brain/megbrain/megbrain_test --gtest_filter="TestOprMisc.Add*"
    Note: Google Test filter = TestOprMisc.Add*
    [==========] Running 2 tests from 1 test case.
    [----------] Global test environment set-up.
    [----------] 2 tests from TestOprMisc
    [ RUN      ] TestOprMisc.Add
    ...
    [       OK ] TestOprMisc.Add2 (20 ms)
    [----------] 2 tests from TestOprMisc (27 ms total)
     
    [----------] Global test environment tear-down
    [==========] 2 tests from 1 test case ran. (27 ms total)
    [  PASSED  ] 2 tests.

常见问题
--------

.. dropdown:: gold: undefined reference to vtable for <NameOfOpr>

   忘记添加 ``MGB_DYN_TYPE_OBJ_FINAL_IMPL``

.. dropdown:: error: undefined reference to <NameOfOpr>

   可以看一下是不是 MegDNN 添加算子的时候出现了遗漏。

