.. _imperative:

============================================
Imperative 包含哪些模块？为什么这么设计？
============================================


Python 层
----------------------------------------

Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

神经网络模型是由对输入数据执行操作的各种层（\ ``Layer``\ ），或者说模块（\ ``Module``\ ）组成。

``Module`` 用来定义网络模型结构，用户实现算法时要用组合模块
``Module (megengine/module)``
的方式搭建模型，定义神经网络时有些结构经常在模型中反复使用，将这样的结构封装为一个
``Module``\ ，既可以减少重复代码也降低了复杂模型编码的难度。

一个 ``module`` 类主要有两类函数：

-  ``__init__``\ ：构造函数，定义了模型各个层的大小。用户自定义的
   ``Module`` 都源自基类
   ``class Module``\ （见\ ``imperative/python/megengine/module/module.py``\ ），所以在构造函数中一定要先调用
   ``super().__init__()``\ ，设置 ``Module``
   的一些基本属性。模型要使用的所有层 / 模块都需要在构造函数中声明。

   .. code:: python

      class Module(metaclass=ABCMeta):
          r"""Base Module class.

          Args:
              name: module's name, can be initialized by the ``kwargs`` parameter
                  of child class.
          """

          def __init__(self, name=None):
              self._modules = []

              if name is not None:
                  assert (
                      isinstance(name, str) and name.strip()
                  ), "Module's name must be a non-empty string"

              self.name = name

              # runtime attributes
              self.training = True
              self.quantize_disabled = False

              # hooks
              self._forward_pre_hooks = OrderedDict()
              self._forward_hooks = OrderedDict()

              # used for profiler and automatic naming
              self._name = None
              self._short_name = None

          # 抽象方法，由继承的 Module 自己实现
          @abstractmethod
          def forward(self, inputs):
              pass

          # 其他方法
          ...

-  ``forward``\ ：定义模型结构，实现前向传播，也就是将数据输入模型到输出的过程。这里会调用
   ``Functional (megengine/functional)``
   中的函数进行前向计算，\ ``forward`` 表示的是模型实现的逻辑。

来看一个例子：

.. code:: python

   class Simple(Module):
       def __init__(self):
           super().__init__()
           self.a = Parameter([1.23], dtype=np.float32)

       def forward(self, x):
           x = x * self.a
           return x

对于上面这个简单的例子（见
``imperative/python/examples/simple/a_mul_b.py``\ ），\ ``__init__``
表明模型中需要一个参数 ``a``\ ，它的初值是固定的，\ ``forward``
中实现了具体的计算逻辑，也就是对传入的参数与 ``a``
进行乘法运算。对于一些更复杂的计算操作（如卷积、池化等）就需要借助
``Functional`` 中提供的方法来完成。

除了 ``__init__`` 和 ``forward``\ ，基类 ``class Module``
提供了很多属性和方法，常用的有：

-  ``def buffers(self, recursive: bool = True, **kwargs) -> Iterable[Tensor]``\ ：返回一个可迭代对象，遍历当前模块的所有
   ``buffers``
-  ``def parameters(self, recursive: bool = True, **kwargs) -> Iterable[Parameter]``\ ：返回一个可迭代对象，遍历当前模块所有的
   ``parameters``
-  ``def tensors(self, recursive: bool = True, **kwargs) -> Iterable[Parameter]``\ ：返回一个此
   ``module`` 的 ``Tensor`` 的可迭代对象
-  ``def children(self, **kwargs) -> "Iterable[Module]"``\ ：返回一个可迭代对象，该对象包括属于当前模块的直接属性的子模块
-  ``def named_buffers(self, prefix: Optional[str] = None, recursive: bool = True, **kwargs) -> Iterable[Tuple[str, Tensor]]``\ ：返回当前模块中
   ``key`` 与 ``buffer`` 的键值对的可迭代对象，这里 ``key`` 是从该模块至
   ``buffer`` 的点路径（\ ``dotted path``\ ）
-  ``def named_parameters(self, prefix: Optional[str] = None, recursive: bool = True, **kwargs) -> Iterable[Tuple[str, Parameter]]``\ ：返回当前模块中
   ``key`` 与 ``parameter`` 的键值对的可迭代对象，这里 ``key``
   是从该模块至 ``buffer`` 的点路径（\ ``dotted path``\ ）
-  ``def named_tensors(self, prefix: Optional[str] = None, recursive: bool = True, **kwargs) -> Iterable[Tuple[str, Tensor]]``\ ：返回当前模块中
   ``key`` 与 ``Tensor``\ （\ ``buffer + parameter``\ ）
   的键值对的可迭代对象，这里 ``key`` 是从该模块至 ``Tensor``
   的点路径（\ ``dotted path``\ ）
-  ``def named_modules(self, prefix: Optional[str] = None, **kwargs) -> "Iterable[Tuple[str, Module]]"``\ ：返回一个可迭代对象，该对象包括当前模块自身在内的其内部所有模块组成的
   ``key-module`` 键-模块对，这里 ``key``
   是从该模块至各子模块的点路径（\ ``dotted path``\ ）
-  ``def named_children(self, **kwargs) -> "Iterable[Tuple[str, Module]]"``\ ：返回一个可迭代对象，该对象包括当前模块的所有子模块（\ ``submodule``\ ）与键（\ ``key``\ ）组成的
   ``key-submodule`` 对，这里 ``key`` 是子模块对应的属性名
-  ``def state_dict(self, rst=None, prefix="", keep_var=False)``\ ：返回模块的状态字典，状态字典是一个保存当前模块所有可学习的
   ``Tensor``
   （\ ``buffer + parameter``\ ）的字典。出于兼容性考虑，字典中的
   ``value`` 的数据结构类型为 ``numpy.ndarray`` （而不是
   ``Tensor``\ ），并且不可修改，是只读的
-  ``def load_state_dict(self, state_dict: Union[dict, Callable[[str, Tensor], Optional[np.ndarray]]], strict=True, )``\ ：加载一个模块的状态字典，这个方法常用于模型训练过程的保存与加载

值得一提的是，\ ``Parameters`` 和 ``Buffer`` 都是与 ``Module`` 相关的
``Tensor``\ ，它们的区别可以理解为：

-  ``Parameter``
   是模型的参数，在训练过程中会通过反向传播进行更新，因此值是可能改变的，常见的有
   ``weight``\ 、\ ``bias`` 等
-  ``Buffer`` 是模型用到的统计量，不会在反向传播过程中更新，常见的有
   ``mean``\ 、\ ``var`` 等

在 ``imperative/python/megengine/module`` 目录下可以看到可以看到 ``megengine/module`` 下已经实现了许多常用的
``module``\ ，定义模型时可以复用其中的模块。


关于 ``Module`` 的更多方法，可以参考
``imperative/python/megengine/module/module.py``.

Functional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们在 ``module``
中定义网络结构时经常需要包含一些计算操作，这些计算操作就定义在
``functional`` 中。

``functional`` 中实现了各类计算函数，包含对很多 ``op``
的封装，供实现模型时调用。

``functional`` 中有些 ``op`` 完全是由 ``Python``
代码实现，有些则需要调用 ``C++`` 接口完成计算（没错，这里的计算就需要
``MegDNN kernel``\ ）。对于后者，需要有机制确保我们的实现能够转发到底层正确执行，所以你在
``functional`` 的许多 ``op`` 实现中会看到 ``builtin`` 和 ``apply``\ ：

-  ``builtin``

   ``builtin`` 封装了所有的 ``op``\ ，我们在 ``functional`` 中通过
   ``builtin.SomeOp(param)`` 的方式获得一个算子
   ``SomeOp``\ ，\ ``param`` 表示获取 ``SomeOp`` 需要的参数。

-  ``apply``

   通过 ``builtin`` 获取到 ``op`` 后，需要调用 ``apply`` 接口来调用
   ``op`` 的底层实现进行计算。\ ``apply`` 是在 ``Python`` 层调用底层
   ``op`` 的接口，\ ``apply`` 的第一个参数是 ``op``\ （通过 ``builtin``
   获得），后面的参数是执行这个 ``op`` 需要的参数，都为 ``Tensor``\ 。在
   ``imperative`` 中 ``op`` 计算通过 ``apply(op, inputs)``
   的方式向下传递，只不过在不同的层对于 ``op`` 和 ``inputs``
   会有不一样的封装方式。

``Functional`` 中的许多 ``op`` 都需要通过 ``builtin`` 和 ``apply``
调用底层 ``MegDNN`` 的 ``op``
来进行计算操作。然而在实际的计算发生前，很多时候需要在 ``Python``
层做一些预处理。

来看下面这个例子（见
``imperative/python/megengine/functional/tensor.py``\ ）：

.. code:: python

   def concat(inps: Iterable[Tensor], axis: int = 0, device=None) -> Tensor:
       ...
       if len(inps) == 1:
           return inps[0]

       if device is None:
           device = get_device(inps)
       device = as_device(device)
       (result,) = apply(builtin.Concat(axis=axis, comp_node=device.to_c()), *inps)
       return result

这里 ``concat`` 方法先对输入数量、\ ``device``
做了一些预处理，然后才调用 ``builtin`` 和 ``apply`` 向下转发。

而对于 ``diag`` 这个 ``op``\ ，无需预处理直接向下传递即可（见
``imperative/python/megengine/functional/tensor.py``\ ）:

.. code:: python

   def diag(inp, k=0) -> Tensor:
       ...
       op = builtin.Diag(k=k)
       (result,) = apply(op, inp)
       return result

因此，对于实现了对应 ``kernel`` 的 ``op``\ ，其在 ``imperative``
层的实现通常非常的短。

上面这个 ``concat`` 的 ``apply`` 调用会进入
``imperative/python/src/tensor.cpp`` 中的 ``py_apply`` 函数，并通过解析
``Python`` 中的参数，将它们转换成 ``C++`` 中的对应类型，然后调用
``imperative::apply``\ ，进入 ``dispatch`` 层。

部分 ``functional`` 的 ``op`` 不直接调用 ``py_apply`` 而是有对应的
``cpp`` 实现，比如 ``squeeze``\ ：

.. code:: python

   def squeeze(inp: Tensor, axis: Optional[Union[int, Sequence[int]]] = None) -> Tensor:
       return squeeze_cpp(inp, axis)

这样的实现往往是需要在调用 ``py_apply`` 之前做一些预处理（在 ``C++``
层面做预处理），直接调用 ``py_apply`` 的方法则将预处理部分放在了
``python`` 层。

比如 ``_squeeze_cpp`` 在 ``C++`` 里做了预处理，再调用 ``py_apply`` ：

.. code:: cpp

   py::object _squeeze_cpp(py::handle inp_hdl, py::handle axis_hdl) {
       /*
           ...
           some preprocess code in C++
           ...
       */
       std::shared_ptr<OpDef> op = RemoveAxis::make(axis = axis);
       py::object Op = py::cast(op);
       PyObject* p[2] = {Op.ptr(), inp_hdl.ptr()};
       // py_apply
       py::tuple ret = py::reinterpret_steal<py::object>(py_apply(NULL, p, 2));
       return ret[0];
   }

对于预处理较耗时的 ``op``\ ，我们可以把预处理部分定义在 ``C++`` 层。

在 ``imperative/python/src/tensor.cpp`` 中可以看到有对应 ``cpp`` 实现的
``op``\ ：

.. code:: cpp

   #define WRAP_FUNC_PY35(FUNC)                                \
       PyObject* py35_##FUNC(PyObject* self, PyObject* args) { \
           auto* arr = &PyTuple_GET_ITEM(args, 0);             \
           auto size = PyTuple_GET_SIZE(args);                 \
           return FUNC(self, arr, size);                       \
       }
   WRAP_FUNC_PY35(py_apply);
   WRAP_FUNC_PY35(dtype_promotion);
   WRAP_FUNC_PY35(get_device);
   WRAP_FUNC_PY35(make_shape_tuple);
   WRAP_FUNC_PY35(getitem_cpp);
   WRAP_FUNC_PY35(setitem_cpp);
   WRAP_FUNC_PY35(split_cpp);
   WRAP_FUNC_PY35(expand_dims_cpp);
   WRAP_FUNC_PY35(squeeze_cpp);
   WRAP_FUNC_PY35(transpose_cpp);
   WRAP_FUNC_PY35(broadcast_cpp);
   WRAP_FUNC_PY35(reshape_cpp);
   WRAP_FUNC_PY35(adaptive_pool2d_cpp);
   WRAP_FUNC_PY35(Const);
   WRAP_FUNC_PY35(astype_cpp);
   WRAP_FUNC_PY35(matmul_cpp);
   WRAP_FUNC_PY35(batched_matmul_cpp);
   WRAP_FUNC_PY35(convert_single_value_cpp);
   WRAP_FUNC_PY35(convert_inputs_cpp);
   WRAP_FUNC_PY35(astensor1d_cpp);
   WRAP_FUNC_PY35(pixel_shuffle_cpp);

对应的函数在 ``imperative/python/src/tensor_utils.cpp`` 中实现。

Optimizer
^^^^^^^^^^^^^^^

``MegEngine`` 中的 ``optimizer``
模块实现了基于各种常见优化策略的优化器。

.. code:: bash

   linrongjian@hhb-ssd:/data/MegBrain/imperative/python/megengine/optimizer$ tree -L 1
   .
   ├── adadelta.py
   ├── adagrad.py
   ├── adam.py
   ├── adamw.py
   ├── clip_grad.py
   ├── __init__.py
   ├── lamb.py
   ├── lr_scheduler.py
   ├── multi_step_lr.py
   ├── optimizer.py
   ├── __pycache__
   └── sgd.py

``optimizer.py`` 中的 ``class Optimizer``
是所有优化器的抽象基类，规定了必须提供的接口，同时为用户提供了包括
``SGD``\ 、\ ``ADAM``
在内的常见优化器实现。这些优化器能够基于参数的梯度信息，按照算法所定义的策略执行更新。

以 ``SGD`` 优化器为例，优化神经网络模型参数的基本流程如下：

.. code:: python

   from megengine.autodiff import GradManager
   import megengine.optimizer as optim

   model = MyModel()
   gm = GradManager().attach(model.parameters())
   optimizer = optim.SGD(model.parameters(), lr=0.01)  # lr may vary with different model

   for data, label in dataset:
       with gm:
           pred = model(data)
           loss = loss_fn(pred, label)
           gm.backward()
           optimizer.step().clear_grad()

-  这里我们构造了一个优化器 ``optimizer``\ ，传入参数是 ``model``
   需要被优化的 ``Parameter``\ ，和 ``learning rate``

-  优化器通过执行 ``step()`` 方法进行一次优化

-  优化器通过执行 ``clear_grad()`` 方法清空参数梯度

   -  为何要手动清空梯度？

      梯度管理器执行 ``backward()`` 方法时，
      会将当前计算所得到的梯度以累加的形式积累到原有梯度上，而不是直接做替换。
      因此对于新一轮的梯度计算，通常需要将上一轮计算得到的梯度信息清空。
      何时进行梯度清空是由人为控制的，这样可允许灵活进行梯度的累积。

关于 ``optimizer`` 的更多资料，可以参考 `MegEngine 官方文档 - 使用
Optimizer
优化参数 <https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/optimizer/index.html?highlight=optimizer>`__.

Quantization
^^^^^^^^^^^^^^^^^^

量化是一种对深度学习模型参数进行压缩以降低计算量的技术。它基于这样一种思想：神经网络是一个近似计算模型，不需要其中每个计算过程的绝对的精确。因此在某些情况下可以把需要较多比特存储的模型参数转为使用较少比特存储，而不影响模型的精度。

量化通过舍弃数值表示上的精度来追求极致的推理速度。直觉上用低精度/比特类型的模型参数会带来较大的模型掉点，但在经过一系列精妙的量化处理之后，其掉点可以变得微乎其微。

如下图所示，量化通常是将浮点模型（常见神经网络的 ``Tensor``
数据类型一般是 ``float32``\ ）处理为一个量化模型（\ ``Tensor``
数据类型为 ``int8`` 等）。

.. mermaid::
   :align: center
   :caption: 通常以浮点模型为起点，经过中间的量化处理后最终变成量化模型

   flowchart LR
         FM[Float Model] -- processing --> QM[Quantized Model]

量化基本流程
''''''''''''''''''''

``MegEngine``
中支持工业界的两类主流量化技术，分别是训练后量化（\ ``PTQ``\ ）和量化感知训练（\ ``QAT``\ ）。

1. 训练后量化（\ ``Post-Training Quantization``, ``PTQ``\ ）

   训练后量化，顾名思义就是将训练后的 ``Float``
   模型转换成低精度/比特模型。

   比较常见的做法是对模型的权重（\ ``weight``\ ）和激活值（\ ``activation``\ ）进行处理，把它们转换成精度更低的类型。虽然是在训练后再进行精度转换，但为了获取到模型转换需要的一些统计信息（比如缩放因子
   ``scale``\ ），仍然需要在模型进行前向计算时插入观察者（\ ``Observer``\ ）。

   使用训练后量化技术通常会导致模型掉点，某些情况下甚至会导致模型不可用。可以使用小批量数据在量化之前对
   ``Observer`` 进行校准（\ ``Calibration``\ ），这种方案叫做
   ``Calibration`` 后量化。也可以使用 ``QAT`` 方案。

2. 量化感知训练（\ ``Quantization-Aware Training``, ``QAT``\ ）

   ``QAT`` 会向 ``Float``
   模型中插入一些伪量化（\ ``FakeQuantize``\ ）算子，在前向计算过程中伪量化算子根据
   ``Observer``
   观察到的信息进行量化模拟，模拟数值截断的情况下的数值转换，再将转换后的值还原为原类型。让被量化对象在训练时“提前适应”量化操作，减少训练后量化的掉点影响。

   而增加这些伪量化算子模拟量化过程又会增加训练开销，因此模型量化通常的思路是：

   -  按照平时训练模型的流程，设计好 ``Float``
      模型并进行训练，得到一个预训练模型；
   -  插入 ``Observer`` 和 ``FakeQuantize`` 算子，得到
      ``Quantized-Float`` 模型（\ ``QFloat`` 模型）进行量化感知训练；
   -  训练后量化，得到真正的 ``Quantized`` 模型（\ ``Q``
      模型），也就是最终用来进行推理的低比特模型。

   过程如下图所示（实际使用时，量化流程也可能会有变化）：

   .. mermaid::
      :align: center

      flowchart LR
         FM[Float Model] --> |train| PFM[Pre-trained Float Model]
         PFM --> |Observer| PQFM[Pre-trained QFloat Model]
         PFM --> |FakeQuantize| PQFM
         PQFM --> |QAT| FQFM[Fine-tuned QFloat Model]
         FQFM --> |PTQ| QM[Q Model]

   注意这里的量化感知训练 ``QAT`` 是在预训练好的 ``QFloat``
   模型上微调（\ ``Fine-tune``\ ）的（而不是在原来的 ``Float``
   模型上），这样减小了训练的开销，得到的微调后的模型再做训练后量化
   ``PTQ``\ （“真量化”），\ ``QModel`` 就是最终部署的模型。

模型（\ ``Model``\ ）与模块（\ ``Module``\ ）
'''''''''''''''''''''''''''''''''''''''''''''''''''''

量化是一个对模型（\ ``Model``\ ）的转换操作，但其本质其实是对模型中的模块（
``Module``\ ） 进行替换。

在 ``MegEngine`` 中，对应与 ``Float Model`` 、\ ``QFloat Model`` 和
``Q Model`` 的 ``Module`` 分别为：

1. 进行正常 ``float`` 运算的默认 ``Module``
2. 带有 ``Observer`` 和 ``FakeQuantize`` 算子的 ``qat.QATModule``
3. 无法训练、专门用于部署的 ``quantized.QuantizedModule``

以 ``Conv`` 算子为例，这些 ``Module`` 对应的实现分别在：

-  ``Float Module``：``imperative/python/megengine/module/conv.py``
-  ``qat.QATModule``：``imperative/python/megengine/module/qat/conv.py``
-  ``quantized.QuantizedModule``：``imperative/python/megengine/module/quantized/conv.py``

量化配置 QConfig
''''''''''''''''''''''''

量化配置包括 ``Observer`` 和 ``FakeQuantize``
两部分，要设置它们，用户可以使用 ``MegEngine``
预设配置也可以自定义配置。

1. 使用 ``MegEngine`` 预设配置

   ``MegEngine``
   提供了多种\ `量化预设配置 <https://www.megengine.org.cn/doc/stable/zh/reference/quantization.html#qconfig-list>`__\ 。

   以 ``ema_fakequant_qconfig``
   为例，用户可以通过如下代码使用该预设配置：

   .. code:: python

      import megengine.quantization as Q
      Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

2. 用户自定义量化配置

   用户还可以自己选择 ``Observer`` 和 ``FakeQuantize``\ ，灵活配置
   ``QConfig`` （可参考
   ``imperative/python/megengine/quantization/qconfig.py`` 灵活选择
   ``weight_observer``\ 、\ ``act_observer``\ 、\ ``weight_fake_quant``
   和 ``act_fake_quant``\ ）。

   可选的 ``Observer`` 和 ``FakeQuantize`` 可参考\ `量化 API
   参考 <https://www.megengine.org.cn/doc/stable/zh/reference/quantization.html#qconfig-obsever>`__\ 页面。

``QConfig``
提供了一系列用于对模型做量化的接口，要使用这些接口，需要网络的
``Module`` 能够在 ``forward`` 时给权重、激活值加上 ``Observer`` 以及进行
``FakeQuantize``\ 。

模型转换的作用是：将普通的 ``Float Module`` 替换为支持这些操作的
``QATModule``\ （可以训练），再替换为
``QuantizeModule``\ （无法训练、专用于部署）。

以 ``Conv2d`` 为例，模型转换的过程如图：

.. mermaid::
   :align: center

   flowchart LR
       M[module.Conv2d] -- quantize_qat --> QATM[module.qat.Conv2d] -- quantize --> QM[module.quantized.Conv2d]


在量化时常常会用到算子融合（\ ``Fusion``\ ）。比如一个 ``Conv2d``
算子加上一个 ``BatchNorm2d`` 算子，可以用一个 ``ConvBn2d``
算子来等价替代，这里 ``ConvBn2d`` 算子就是 ``Conv2d`` 和 ``BatchNorm2d``
的融合算子。

``MegEngine`` 中提供了一些预先融合好的 ``Module``\ ，比如
``ConvRelu2d``\ 、\ ``ConvBn2d`` 和 ``ConvBnRelu2d``
等。使用融合算子会使用底层实现好的融合算子（\ ``kernel``\ ），而不会分别调用子模块在底层的
``kernel``\ ，因此能够加快模型的速度，而且框架还无需根据网络结构进行自动匹配和融合优化，同时存在融合和不需融合的算子也可以让用户能更好的控制网络转换的过程。

实现预先融合的 ``Module``
也有缺点，那就是用户需要在代码中修改原先的网络结构（把可以融合的多个
``Module`` 改为融合后的 ``Module``\ ）。

模型转换的原理是，将父 ``Module`` 中的 ``Quantable`` （可被量化的）子
``Module`` 替换为新 ``Module``\ 。而这些 ``Quantable submodule``
中可能又包含 ``Quantable submodule``\ ，这些 ``submodule``
不会再进一步转换，因为其父 ``Module`` 被替换后的 ``forward``
计算过程已经改变了，不再依赖于这些子 ``Module``\ 。

有时候用户不希望对模型的部分 ``Module`` 进行转换，而是保留其 ``Float``
状态（比如转换会导致模型掉点），则可以使用 ``disable_quantize``
方法关闭量化。

比如下面这行代码关闭了 ``fc`` 层的量化处理：

.. code:: python

   model.fc.disable_quantize()

由于模型转换过程修改了原网络结构，因此模型保存与加载无法直接适用于转换后的网络，读取新网络保存的参数时，需要先调用转换接口得到转换后的网络，才能用
``load_state_dict`` 将参数进行加载。

量化代码
''''''''''''''''

要从一个 ``Float``
模型得到一个可用于部署的量化模型，大致需要经历三个步骤：

1. 修改网络结构。将 ``Float`` 模型中的普通 ``Module`` 替换为已经融合好的
   ``Module``\ ，比如 ``ConvBn2d``\ 、\ ``ConvBnRelu2d`` 等（可以看看
   ``imperative/python/megengine/module/quantized``
   目录下提供了哪些已融合的模块）。然后在正常模式下预训练模型，并且在每轮迭代保存网络检查点。

   以 ``ResNet18`` 的 ``BasicBlock`` 为例，模块修改前的代码为：

   .. code:: python

      class BasicBlock(M.Module):
            def __init__(self, in_channels, channels):
               super().__init__()
               self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=dilation, bias=False)
               self.bn1 = M.BatchNorm2d
               self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
               self.bn2 = M.BatchNorm2d
               self.downsample = (
                  M.Identity()
                  if in_channels == channels and stride == 1
                  else M.Sequential(
                  M.Conv2d(in_channels, channels, 1, stride, bias=False)
                  M.BatchNorm2d
               )

            def forward(self, x):
               identity = x
               x = F.relu(self.bn1(self.conv1(x)))
               x = self.bn2(self.conv2(x))
               identity = self.downsample(identity)
               x = F.relu(x + identity)
               return x

   注意到现在的前向中使用的都是普通 ``Module``
   拼接在一起，而实际上许多模块是可以融合的。

   用可以融合的模块替换掉原先的 ``Module``\ ：

   .. code:: python

      class BasicBlock(M.Module):
            def __init__(self, in_channels, channels):
               super().__init__()
               self.conv_bn_relu1 = M.ConvBnRelu2d(in_channels, channels, 3, 1, padding=dilation, bias=False)
               self.conv_bn2 = M.ConvBn2d(channels, channels, 3, 1, padding=1, bias=False)
               self.downsample = (
                  M.Identity()
                  if in_channels == channels and stride == 1
                  else M.ConvBn2d(in_channels, channels, 1, 1, bias=False)
               )
               self.add_relu = M.Elemwise("FUSE_ADD_RELU")

            def forward(self, x):
               identity = x
               x = self.conv_bn_relu1(x)
               x = self.conv_bn2(x)
               identity = self.downsample(identity)
               x = self.add_relu(x, identity)
               return x

   注意到此时前向中已经有许多模块使用的是融合后的 ``Module``\ 。

   再对该模型进行若干论迭代训练，并保存检查点：

   .. code:: python

      for step in range(0, total_steps):
          # Linear learning rate decay
          epoch = step // steps_per_epoch
          learning_rate = adjust_learning_rate(step, epoch)

          image, label = next(train_queue)
          image = tensor(image.astype("float32"))
          label = tensor(label.astype("int32"))

          n = image.shape[0]

          loss, acc1, acc5 = train_func(image, label, net, gm)  # traced
          optimizer.step().clear_grad()

          # Save checkpoints

   完整代码见：

   -  `修改前的模型结构 <https://github.com/MegEngine/Models/blob/master/official/vision/classification/resnet/model.py>`__
   -  `修改后的模型结构 <https://github.com/MegEngine/Models/blob/master/official/quantization/models/resnet.py>`__

2. 调用 ``quantize_qat`` 方法（见
   ``imperative/python/megengine/quantization/quantize.py``\ ） 将
   ``Float`` 模型转换为 ``QFloat``
   模型，并进行微调（量化感知训练或校准，取决于 ``QConfig``\ ）。

   使用 ``quantize_qat`` 方法将 ``Float`` 模型转换为 ``QFloat``
   模型的代码大致为：

   .. code:: python

      from megengine.quantization import ema_fakequant_qconfig, quantize_qat

      model = ResNet18()

      # QAT
      quantize_qat(model, ema_fakequant_qconfig)

      # Or Calibration:
      # quantize_qat(model, calibration_qconfig)

   将 ``Float`` 模型转换为 ``QFloat`` 模型后，加载预训练 ``Float``
   模型保存的检查点进行微调 / 校准：

   .. code:: python

      if args.checkpoint:
          logger.info("Load pretrained weights from %s", args.checkpoint)
          ckpt = mge.load(args.checkpoint)
          ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
          model.load_state_dict(ckpt, strict=False)

      # Fine-tune / Calibrate with new traced train_func
      # Save checkpoints

   完整代码见：

   -  `Finetune <https://github.com/MegEngine/Models/blob/master/official/quantization/finetune.py>`__
   -  `Calibration <https://github.com/MegEngine/Models/blob/master/official/quantization/calibration.py>`__

3. 调用 ``quantize`` 方法（见
   ``imperative/python/megengine/quantization/quantized.py``\ ）将
   ``QFloat`` 模型转换为 ``Q`` 模型，也就是可用于模型部署的量化模型。

   需要在推理的方法中设置 ``trace`` 的
   ``capture_as_const=True``\ ，以进行模型导出：

   .. code:: python

      from megengine.quantization import quantize

      @jit.trace(capture_as_const=True)
      def infer_func(processed_img):
          model.eval()
          logits = model(processed_img)
          probs = F.softmax(logits)
          return probs

      quantize(model)

      processed_img = transform.apply(image)[np.newaxis, :]
      processed_img = processed_img.astype("int8")
      probs = infer_func(processed_img)

      infer_func.dump(output_file, arg_names=["data"])

   调用了 ``quantize`` 后，\ ``model`` 就从 ``QFloat`` 模型转换为了
   ``Q`` 模型，之后便使用这个 ``Quantized`` 模型进行推理。

   调用 ``dump`` 方法将模型导出，便得到了一个可用于部署的量化模型。

   完整代码见：

   -  `Inference and
      dump <https://github.com/MegEngine/Models/blob/master/official/quantization/inference.py>`__

关于 ``quantization`` 的更多资料，可以参考 `MegEngine 官方文档 -
量化（Quantization） <https://www.megengine.org.cn/doc/stable/zh/user-guide/model-development/quantization/index.html#quantization-guide>`__.

dispatch / transformation层
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

什么是 Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 ``Python`` 层，我们通过 ``builtin`` 和 ``apply`` 封装了很多
``op``\ ，一个 ``op`` 会向下层发射指令对 ``tensor`` 进行操作并返回
``tensor``\ ，发射的 ``op`` 指令到 ``dispatch`` 层，\ ``dispatcher``
需要做一些处理，这些处理叫做 **Transformation**.

一些重要的 Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Transformation`` 主要包含这几类：

-  **DimExpansionTransformation**\ （\ ``imperative/src/impl/transformations/dim_expansion.cpp``\ ）：某些
   ``op`` 对输入类型的 ``shape`` 有要求，在这里做处理。

-  **DtypePromoteTransformation**\ （\ ``imperative/src/impl/transformations/grad.cpp``\ ）：某些
   ``op`` 会将所有的输入的类型提升为同一类型之后再进行计算。比如 ``int``
   类型 ``tensor`` 和 ``float`` 类型 ``tensor`` 进行计算，需要把 ``int``
   类型的 ``tensor`` 转换为 ``float`` 类型 ``tensor``\ 。

-  **InterpreterTransformation**\ （\ ``imperative/src/impl/transformations/eval.cpp``\ ）：顾名思义，这类
   ``Transformation`` 将指令转发到 ``Interpreter`` 层（\ ``Interpreter``
   可以认为是 ``Imperative``
   中所有计算操作的入口）进行计算，并获取指令的计算结果。\ ``Transformation``
   通常是叠加的，\ ``InterpreterTransformation``
   是最后一层，其后不再跟其他的 ``Transformation`` 处理。

-  **FormatTransformation**\ （\ ``imperative/src/impl/transformations/format.cpp``\ ）：由于在不同情况下对不同
   ``format`` 的 ``Tensor`` 的计算速度不同，因此需要对 ``NHWC`` 和
   ``NCHW`` 的 ``Tensor``
   进行转换，为了不让用户感知到这样的转换，这部分的工作由
   ``FormatTransformation`` 完成。

-  **GradTransformation**\ （\ ``imperative/src/impl/transformations/grad.cpp``\ ）：训练模型时需要通过反向传播更新模型参数，反向传播需要支持
   ``op`` 的自动微分。要实现求导，就需要在前向执行 ``op``
   的时候记录某些信息，以便之后进行反向求导。\ ``Autodiff``
   算法会根据输入的前向图生成一个完整的前向反向图，所谓的前传反传训练过程对
   ``Autodiff`` 来说就是一个图的前向过程，\ ``grad``
   的数值是在“前向”的过程中就已经拿到的。

-  **TracingTransformation**\ （\ ``imperative/src/impl/transforamtions/trace.cpp``\ ）：

   在介绍 ``Trace``
   之前，我们需要先明确一下计算图的概念。计算图可以认为是对输入的数据（\ ``tensor``\ ）、\ ``op``
   以及 ``op``
   执行的顺序的表示。计算图分为动态图和静态图。动态图是在前向过程中创建、反向过程销毁的。前向逻辑本身是可变的，所以执行流程也是可变的（因此叫动态图），而静态图的执行流程是固定的。也就是说，动态图在底层是没有严格的图的概念的（或者说这个图本身一直随执行流程变化）。对于动态图来说，\ ``graph``
   的 ``node`` 对应的概念是 ``function`` / 算子，而 ``edge``
   对应的概念是 ``tensor``\ ，所以在图中需要记录的是 ``graph`` 中
   ``node`` 和 ``edge`` 之间的连接关系，以及 ``tensor`` 是 ``function``
   的第几个输入参数。

   ``Trace``
   的作用就是将动态图执行转换为静态图执行，这样做的好处就是执行速度更快了，并且占用的显存更少了。因为静态图需要先构建再运行，可以在运行前对图结构进行优化（融合算子、常数折叠等），而且只需要构建一次（除非图结构发生变化）。而动态图是在运行时构建的，既不好优化还会占用较多显存。

   ``Trace`` 中所有的东西都会进行静态优化（加速）。

   加了 ``Trace`` 之后，模型在训练时第一个 ``iter``
   是动态图执行，\ ``Trace`` 会记录下 ``tensor``\ 、\ ``op`` 以及 ``op``
   的执行顺序这些信息（构建静态图）并进行计算，在第二个 ``iter``
   就跑的是构建好的静态图。

-  **LazyEvalTransformation**\ （\ ``imperative/src/impl/transformations/lazy.cpp``\ ）：类似
   ``TracingTransformation``\ ，也会记录 ``tensor``\ 、\ ``op``
   等信息构建静态图，不同的是 ``LazyEvalTransformation`` 在第一个
   ``iter`` 不会跑动态图，但会在第二个 ``iter`` 开始跑静态图。

-  **ScalarTransformation**\ ：用于判断指令的输出是否为
   ``scalar``\ 。因为 ``dispatch`` 的 ``Tensor`` 要发到 ``Interpreter``
   层，而 ``Interpreter`` 层不接受 ``ndim == 0`` 的 ``Tensor``\ （在
   ``Interpreter`` 中 ``ndim`` 为 ``0`` 表示 ``Tensor`` 的 ``shape``
   未知），也就是一个 ``scalar``\ ，因此 ``ScalarTransformation`` 会将
   ``ndim`` 为 ``0`` 的 ``Tensor`` 表示为 ``ndim`` 不为 ``0`` 的
   ``Tensor`` （具体是多少与具体 ``op`` 有关）发往 ``Interpreter``\ 。

不同的 ``Transformation`` 之间拥有固定的执行顺序：比如
``InterpreterTransformation`` 是执行实际计算并获取计算结果的（需要进入
``Interpreter``\ ），所以它是在最后一个执行的。\ ``TracingTransformation``
/ ``LazyEvalTransformation`` / ``CompiledTransformation`` 等属于
``Trace`` 相关的操作，因为 ``Trace`` 需要记录所有指令，所以这些
``Transformation`` 是在倒数第二层执行的。如 ``ScalarTransformation``
这样只对 ``Scalar`` 做处理的 ``Transformation`` 往往在较上层。

举个例子：

.. code:: python

   import megengine as mge
    
   w = mge.Parameter(8, dtype="float32")
   x = mge.tensor(4, dtype="float32")
   gm = mge.autodiff.GradManager()
   gm.attach(w)
   optimizer = mge.optimizer.SGD([w], lr=0.01)
    
   @mge.jit.trace
   def func(x):
       with gm:
           y = w + x
           gm.backward(y)
           optimizer.step().clear_grad()
       return y
    
   y = func(x)
   print(y.numpy())

上面代码包含求导（\ ``GradTransformation``\ ）、对标量做操作（一个实数转为
``Tensor``\ ，\ ``ScalarTransformation``\ ）、\ ``Trace``\ （\ ``TracingTransformation``\ ），还有就是实际的运算（\ ``InterpreterTransformation``\ ）。

**因为不同的 Transformation
有逻辑上的先后关系，所以开发者往往需要手动规划它们之间的顺序。**

不同类型的 ``Transformation`` 之间是解耦的，这样便于开发与维护。

如何发射 op 指令
^^^^^^^^^^^^^^^^^^^^^^

上面提到，\ ``functional`` 中的 ``op`` 通过 ``imperative::apply`` 进入
``dispatch``\ ，会调用 ``dispatch.cpp``
（\ ``imperative/src/impl/dispatch.cpp``\ ） 中的 ``apply`` 方法。

在 ``dispatch.cpp`` 中有三个 ``apply`` 方法的签名：

-  ``ValueRefList apply(const Operator& op, Span<ValueRef> inputs)``

   对应一个 ``operator`` 的 ``apply``\ ，这里的 ``op`` 是更为“广义”的
   ``op``\ （包含一些不对用户暴露的、方便开发者开发与调试 ``imperative``
   的语法糖），比如 ``CreateTensor``\ 、\ ``GetName`` 等，而不是
   ``ops.td`` 中定义的 ``op``\ 。简单说就是， ``dispatch`` 层会包含一些
   ``OpDef`` 所没有的 ``operator``\ （见
   ``imperative/src/include/megbrain/imperative/basic_operators.h``\ ），比如：

   -  ``ApplyOp``\ ：一般意义上的 ``op``
   -  ``GetAttr``\ ：获取 ``tensor`` 的属性，比如
      ``DType``\ 、\ ``Device``\ 、\ ``Shape``\ 、\ ``Value``\ 、\ ``Data``
   -  ``CreateTensor``\ ：从 ``host`` / ``device`` 值构造 ``tensor``

   这个 ``apply`` 方法是每个进入 ``dispatch`` 层的 ``transformation``
   都会执行的，后两个 ``apply`` 方法最终也都会调用这个 ``apply``\ 。

-  ``ValueRefList apply(const OpDef& def, Span<ValueRef> inputs)``

   对应一个“狭义”的 ``op``\ （\ ``ops.td`` 中定义的 ``op``\ ） 的
   ``apply``\ ，会调用
   ``apply(const Operator& op, Span<ValueRef> inputs)``\ 。

-  ``ValueRefList apply(const Subgraph& graph, Span<ValueRef> inputs)``

   通常在求导的时候会调用到这个 ``apply``\ 。这里的 ``graph``
   是一个包含多个 ``transformation`` 的数据结构，这个 ``apply`` 会对这些
   ``transformation`` 逐个调用
   ``apply(const Operator& op, Span<ValueRef> inputs)``\ 。

``ValueRefList apply(const Operator& op, Span<ValueRef> inputs)`` 会调用
``dispatch.cpp`` 的 ``apply_release`` 方法 或者 ``apply_debug`` 方法
（取决于编译的 ``MegEngine`` 是 ``release mode`` 还是
``debug mode``\ ，可以在编译 ``MegEngine`` 时用参数指定
``CMAKE_BUILD_TYPE`` 为 ``Release`` 或 ``Debug``\ ，默认为
``Release``\ ，见 ``CMakeLists.txt``\ ）。\ ``Debug`` 版 ``MegEngine``
相比 ``Release`` 版在 ``gdb``
调试时可以看到更多符号，但是少了一些编译器优化，主要在开发者进行调试时使用。

以 ``apply_release`` 方法为例，讲一下 ``transformation`` 的执行流程：

.. code:: cpp

   ValueRefList apply_release(const Operator& op, Span<ValueRef> inputs) {
       auto& context = Transformation::get_context();
       size_t& depth = context.next_transformation;
       mgb_assert(depth < context.transformations.size());
       auto& transformation = *context.transformations[depth++];
       CleanupGuard _{[&] { --depth; }};
       return transformation.apply_transformation(op, inputs);
   }

1. 变量 ``context`` 维护一个 ``transformation``
   的上下文环境，\ ``context`` 里记录了当前所有要执行的
   ``transformation``\ （包括指向 ``transformation``
   的指针、当前要执行的 ``transformation`` 的名字和索引）。
2. 一个 ``size_t`` 类型的变量 ``depth`` 记录当前需要执行的
   ``transformation``\ ，每次执行了一个 ``transformation`` 则 ``depth``
   加一，执行结束后 ``depth`` 减一。
3. 执行当前的 ``transformation``\ ，调用 ``apply_transformation``
   执行对应的 ``transformation``\ ，执行结束后回收资源。

执行 ``InterpreterTransformation`` 时，通过调用
``imperative/src/impl/transformations/eval.cpp`` 的
``apply_transformation`` 方法将指令转发到 ``Interpreter`` 层；而执行其他
``Transformation`` 之后则会继续调用 ``imperative::apply`` 执行后面的
``Transformation``\ 。

InterpreterTransformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``InterpreterTransformation`` 的实现在
``imperative/src/impl/transformations/eval.cpp``\ 。

作为 ``dispatch`` 层最后一类执行的
``transformation``\ ，\ ``InterpreterTransformation`` 是最基本最简单的
``transformation``\ 。它会将指令发到 ``Interpreter``\ 。

来看一下 ``InterpreterTransformation`` 的 ``apply_transformation``
方法：

.. code:: cpp

   ValueRefList InterpreterTransformation::apply_transformation(
           const Operator& op, Span<ValueRef> inputs) {
       if (auto* op_val = op.as<ApplyOp>()) {
           ...
       } else if (auto* get_attr = op.as<GetAttr>()) {
           ...
       } else if (auto* create_tensor = op.as<CreateTensor>()) {
           ...
       } else if (auto* dtr_command = op.as<DTRCommand>()) {
           ...
       } else if (auto* rename_value = op.as<RenameValue>()) {
           ...
       } else if (op.is<GetName>()) {
           ...
       } else if (op.is<DupTensor>()) {
           ...
       } else {
           return op.fallback(inputs);
       }
   }

可以看到 ``InterpreterTransformation`` 包含一些很常见的操作 /
``op``\ ，比如：获取 ``tensor``
的属性（\ ``DType``\ 、\ ``Shape``\ 、\ ``Device``\ 等）、创建一个
``tensor``\ 、拷贝 ``tensor`` 等。

关于 ``InterpreterTransformation`` 的所有方法可以参考
``imperative/src/include/megbrain/imperative/transformations/eval.h``\ 。

interpreter 层
~~~~~~~~~~~~~~~~~~

由于 ``MegBrain``
已经是一个非常成熟的静态图框架，因此在开发动态图（\ ``Imperative Runtime``\ ）深度学习框架
``MegEngine`` 的过程中，复用许多静态图中的组件可以大大降低开发成本。

proxy_graph
^^^^^^^^^^^^^^^^^

``proxy_graph``
顾名思义，就是在动态图（\ ``imperative``\ ）中转发对应的方法（执行
``op``\ 、\ ``shape`` 推导）到静态图上，通过复用 ``MegBrain``
的组件来降低开发成本。因此，本质上可以认为 ``proxy_graph`` 是
``MegBrain`` 中的静态图。

在 ``imperative/src/include/megbrain/imperative/proxy_graph_detail.h``
中可以找到这些函数声明，它们会隐含地操作一个静态图 / ``proxy graph``\ ：

-  ``infer_output_attrs_fallible``

   根据 ``OpDef`` 通过 ``apply_on_var_node`` 构造出一个静态图
   ``op``\ （称为 ``proxy opr``\ ），然后复用 ``MegBrain`` 的
   ``StaticInferManager`` 进行 ``shape`` 推导。

-  ``apply_on_physical_tensor``

   根据 ``infer_output_attrs_fallible`` 推导的 ``shape`` 结果去分配
   ``op`` 输出的显存，并调用 ``proxy opr`` 的 ``execute`` 函数（会转发到
   ``MegDNN`` 的 ``exec`` 函数）执行计算操作。

-  ``make_backward_graph``

   在求导时，\ ``Grad Manager`` 会记录下来一些求导需要的信息（输入
   ``tensor``\ 、\ ``op`` 以及它们执行的顺序、输出
   ``tensor``\ ），\ ``make_backward_graph``
   会根据这些信息造一个反向的计算图，供求导使用。

-  ``get_input_layout_constraint``

   一般用来判断一个输入 ``tensor`` 的 ``layout``
   是否满足一些限制：比如判断 ``tensor`` 是否是连续的。

   如果不满足限制，则会造一个满足限制的 ``tensor``\ ，供
   ``apply_on_physical_tensor`` 使用。

在实现 ``imperative`` 算子时，需要实现对应的 ``apply_on_var_node``
方法，表明我们需要复用 ``MegBrain``
中的组件（通过上述的几个方法）。其他的方法（\ ``apply_on_physical_tensor``
等）如果在 ``imperative`` 中实现了（调用 ``MegDNN``
中的对应方法）则直接使用 ``imperative`` 中的方法，否则通过
``proxy_graph`` 转发到 ``MegBrain`` 中的对应方法。

使用 proxy_graph 的缺点
'''''''''''''''''''''''''''''''

复用 ``proxy_graph`` 导致很多操作都依赖于 ``proxy opr``\ ，因此每次
``apply`` 都需要构造一次 ``proxy opr``\ （通过 ``apply_on_var_node``
方法）。

对于一些 ``host`` 开销比较大（某些特殊 ``op`` 或者 ``op``
的输入比较小）的情况下，单独构造 ``proxy opr``
的开销太大了，这会导致一些小模型性能不佳。

于是便有了 ``mini_graph``\ （见
``imperative/src/impl/proxy_graph/mini_graph.h``\ ）。

mini_graph
''''''''''''''''''

   由于 ``proxy_graph`` 构造 ``proxy opr`` 的开销较大，\ ``mini_graph``
   的主要思想是缓存 ``proxy_graph``\ ，这样在第二次执行该 ``op``
   的时候就不需要完全重新构造，而是直接去 ``cache``
   中读取，这样可以有效降低后续执行的开销。

然而即使有了 ``mini_graph`` 缓存
``proxy opr``\ ，通过静态图来执行某些操作依然是比较耗时的。由于很大一部分算子的
``shape`` 推导只是一些数字的四则运算，因此我们完全不需要通过
``StaticInferManager`` 来推导 ``shape``\ ，可以在 ``python``
层提前计算好输出的 ``shape`` 再作为参数往下传递。

通过 ``OpDef`` 系统我们可以为某些 ``op``
做针对性的操作，也就是说，可以将提前计算好的 ``shape`` 作为 ``op``
的参数，这样就不必再通过 ``StaticInferManager`` 推导 ``op`` 的
``shape``\ 。

比如 ``reshape`` 就将 ``shape`` 作为参数通过 ``OpDef`` 传给
``proxy_graph``\ （见 ``src/core/include/megbrain/ir/ops.td``\ ）：

.. code:: cpp

   def Reshape: MgbHashableOp<"Reshape", [OptionalAxisV1Param]> {
     let extraArguments = (ins
       MgbArrayAttr<MgbI32Attr>:$shape
     );
   }

接下来详细介绍一下 ``proxy_graph_detail.h`` 中的方法。

infer_output_attrs_fallible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``infer_output_attrs_fallible`` 方法对 ``shape``
进行推导，将结果保存在一个叫 ``TensorShape``
（\ ``dnn/include/megdnn/basic_types.h``\ ）的数据结构中，我们可以通过
``get_shape``
方法（\ ``imperative/src/impl/interpreter/interpreter_impl.h``\ ）获取
``shape``\ 。

实际上，\ ``infer_output_attrs_fallible``
做的操作不止这些（\ ``imperative/src/impl/proxy_graph/mini_graph.h``\ ）：

.. code:: cpp

   std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
           const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
       auto& minigraph = get_cached_minigraph(def, inputs);
       auto _ = scoped_attach(&minigraph);
       auto sess = minigraph.infer_session(inputs);
       std::tuple<SmallVector<LogicalTensorDesc>, bool> ret;
       auto& [descs, noerr] = ret;
       for (size_t i = 0; i < minigraph.output_size(); ++i) {
           descs.emplace_back();
           auto& desc = descs.back();
           desc.layout.dtype = minigraph.output_var(i)->dtype();
           desc.layout.format = minigraph.output_var(i)->format();
           desc.comp_node = minigraph.output_var(i)->comp_node();
           if (auto* shape = sess.infer_shape(i, false)) {
               desc.layout.init_contiguous_stride(*shape);
               noerr = true;
           } else {
               noerr = false;
           }
       }
       return ret;
   }

``mini_graph`` 从缓存中获取 ``OpDef`` 的 ``apply_on_var_node`` 构造的
``proxy_graph``\ ，并构造一个 ``std::tuple`` ``ret`` ，\ ``ret``
中的元素是都 ``LogicalTensorDesc``
类型（\ ``imperative/src/include/megbrain/imperative/physical_tensor.h``\ ），一个
``LogicalTensorDesc`` 对象存放了一个 ``tensor`` 的相关属性：

.. code:: c++

   struct LogicalTensorDesc {
       TensorLayout layout;
       CompNode comp_node;
       DeviceTensorND value;  // cpu:default
   };

之后逐个将从缓存中读取到的属性赋值给 ``ret`` 中元素，并且推导输出的
``shape``\ ，如果推导了 ``shape``\ ，则设置 ``noerr`` 为
``true``\ ，否则为 ``false``\ 。

这样，在内存中便有了一份由当前 ``op`` 与输入推导出的输出 ``tensor``
的信息，可以调用 ``struct ChannelImpl``\ （见
``imperative/src/impl/interpreter/interpreter_impl.h`` ）的
``get_value``\ 、\ ``get_shape``\ 、\ ``get_dtype``\ 、\ ``get_device``
等方法获取相应的属性。

apply_on_physical_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``apply_on_physical_tensor`` 的主要作用是调用 ``proxy opr`` 的
``execute`` 函数进行计算。

在执行计算操作前， ``apply_on_physical_tensor`` 需要通过获取
``infer_output_attrs_fallible`` 的结果来进行显存分配。

一般 ``apply_on_physical`` 的函数签名为：

.. code:: c++

   SmallVector<TensorPtr> apply_on_physical_tensor(
           const OpDef& def, const SmallVector<TensorPtr>& inputs,
           SmallVector<LogicalTensorDesc>& output_descs, const bool& validated)

第一个形参 ``def`` 表示 ``op``\ ，第二个形参 ``inputs`` 是输入
``tensor``\ ，第三个形参 ``output_descs`` 表示输出 ``tensor``
的属性（也就是我们通过 ``infer_output_attrs_fallible``
方法得到并保存在内存中的信息），第四个形参 ``validated`` 表示是否能通过
``infer_output_attrs_fallible`` 获取输出的属性。

以卷积为例，其在调用 ``kernel``
进行计算前有这样一段代码（\ ``imperative/src/impl/ops/convolution.cpp``\ ）：

.. code:: cpp

   // 涉及到分配显存，需要先获取输出的大小
   auto out_layout = [&] {
       if (validated) {
           return output_descs[0].layout;
       } else {
           TensorLayout out_layout{inputs[0]->dtype()};
           dnn_opr.op()->deduce_layout(
                   inputs[0]->layout(), inputs[1]->layout(), empty_bias.layout,
                   empty_bias.layout, out_layout);
           return out_layout;
       }
   }();

   // 分配显存并调用 kernel 执行计算操作
   auto out = Tensor::make(out_layout, cn);
       dnn_opr.exec_fastrun(inputs[0], inputs[1], empty_bias, empty_bias, out);
       return {out};

这里表示如果 ``validated`` 为 ``true``\ ，说明我们已经通过
``infer_output_attrs_fallible`` 提前算好了输出的大小，否则就需要调用
``dnn`` 层的 ``deduce_layout`` 方法来推导输出的大小。

分配显存后，再将新构造的 ``Tensor`` 放到 ``kernel``
里执行计算（\ ``exec_fastrun`` 方法）。

op 注册
~~~~~~~~~~~

以 ``Elemwise`` 这个 ``op`` 为例，介绍一下在 ``imperative`` 中 ``op``
注册是如何工作并且能够替换默认版本实现的。

dnn/scripts/opr_param_defs.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

因为计算操作最终都需要调用 ``MegDNN`` 的算子，我们先看一下在 ``MegDNN``
中如何定义算子。在 ``dnn/scripts/opr_param_defs.py`` 中可以找到所有
``MegDNN`` 的 ``op`` 的参数定义。

对于 ``Elemwise``\ ，因为它表示的是一类算子，所以它有一个 ``Mode``
参数来标识具体算子，可选值有 ``RELU``\ 、\ ``ABS``\ 、\ ``ACOS`` 等：

.. code:: python

   pdef('Elemwise').add_enum(
       'Mode',
       Doc('RELU = 0', 'unary: max(x, 0)'),
       Doc('ABS = 1', 'unary: abs(x)'),
       Doc('ACOS = 2', 'unary: acos(x)'),
       ...
       Doc('ISINF = 63', 'unary: isinf(x)'),
   )

编译 ``MegEngine`` 时，会根据这个脚本生成一个对应的头文件
``build/dnn/include/megdnn/opr_param_defs.h`` 和一个给 ``tablegen``
使用的 ``op param`` 的文件
``build/src/core/include/megbrain/ir/param_defs.td``\ 。

opr_param_defs.h
^^^^^^^^^^^^^^^^^^^^^^

``opr_param_defs.h`` 是 ``C++``
版本的参数声明代码（\ ``opr_param_defs.py`` 是 ``Python``
版本的参数声明代码）。它根据我们在 ``opr_param_defs.py``
中填写的参数和说明生成这个 ``op`` 对应的结构体。

对于 ``Elemwise``\ ，生成了如下的一个 ``struct``\ ：

.. code:: cpp

   struct Elemwise {
       static MEGDNN_CONSTEXPR uint32_t TAG = 791173831u;
       enum class Mode: uint32_t {
           //! unary: max(x, 0)
           RELU = 0,
           //! unary: abs(x)
           ABS = 1,
           //! unary: acos(x)
           ACOS = 2,
           ...
           //! unary: isinf(x)
           ISINF = 63
       };
       static MEGDNN_CONSTEXPR uint32_t MODE_NR_MEMBER = 64;
       union { struct {
       Mode mode;
       }; };
       Elemwise(Mode mode_=Mode::RELU) {
           memset(this, 0, sizeof(*this));
           this->mode = mode_;
       }
   };

因为一个 ``op`` 只能是 ``Elemwise`` 的一个 ``mode``\ ，这里用联合体来存
``Mode``\ ，并且将 ``opr_param_defs.py`` 中的 ``Doc``
中的内容作为这个结构体的实现和注释。

tablegen
^^^^^^^^^^^^^^

在 ``imperative/tablegen`` 目录下 ``MegEngine`` 会根据 ``OpDef``
生成一些 ``op`` 的 ``C++`` 实现以及 ``Python`` 与 ``C++``
的类型绑定。\ ``autogen.cpp`` 根据 ``targets``
目录下的代码生成规则生成对应的代码，放在 ``generated`` 目录下。

ops.td
^^^^^^^^^^^^

``src/core/include/megbrain/ir/ops.td`` 中定义了 ``op``
本体，\ ``tablegen`` 会根据 ``ops.td`` 中的内容生成对应 ``op`` 的 文件。

以 ``Elemwise`` 为例，其 ``param`` 为：

.. code:: c++

   def Elemwise : MgbHashableOp<"Elemwise", [ElemwiseParam], [NoSideEffect]> {
     let inputs = (ins Variadic<AnyType>:$input);
     let results = (outs AnyType);
     let nameFunction = [{
       return to_string($_self.mode);
     }];
   }

这里的 ``ElemwiseParam`` 定义在 ``param_defs.td``
中。\ ``param_defs.td`` 是根据 ``MegDNN`` 的 ``op`` 定义文件
``dnn/scripts/opr_param_defs.py`` 生成的，与 ``op`` 的参数有关。

生成的文件：opdef.h.inl, opdef.cpp.inl, opdef.py.inl, opdef.cpy.inl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``mlir`` 的 ``tablegen`` 根据 ``ops.td`` 生成在
``imperative/tablegen/generated/``
目录下生成四个文件：\ ``opdef.h.inl``, ``opdef.cpp.inl``,
``opdef.py.inl`` 和 ``opdef.cpy.inl``\ 。

这四个文件就是对应的 ``op`` 在 ``OpDef`` 的实现。

1. ``opdef.h.inl``\ ：头文件
2. ``opdef.cpp.inl``\ ：\ ``op`` 对应的 ``cpp`` 实现
3. ``opdef.py.inl``\ ：类似于 ``pybind``\ ，用来实现 ``python`` 和
   ``cpp`` 的类型绑定（已废弃，现在用 ``opdef.cpy.inl``\ ）
4. ``opdef.cpy.inl``\ ：用来实现 ``python`` 和 ``cpp`` 的类型绑定

在 ``opdef.h.inl`` 中关于 ``Elemwise`` 的定义如下：

.. code:: cpp

   class Elemwise : public OpDefImplBase<Elemwise> {
       MGB_DYN_TYPE_OBJ_FINAL_DECL;

   public:
       using Mode = ::megdnn::param::Elemwise::Mode;
       Mode mode = ::megdnn::param::Elemwise::Mode::RELU;
       Elemwise() = default;
       Elemwise(Mode mode_, std::string scope_ = {}): mode(mode_) { set_scope(scope_); }
       Elemwise(::megdnn::param::Elemwise packed_param_0): mode(packed_param_0.mode) {}
       ::megdnn::param::Elemwise param() const {
           return {mode};
       }
   };

可以看到 ``Elemwise`` 类继承自 ``OpDefImplBase`` 类。

而 ``OpdefImpleBase`` 类则继承自 ``OpDef``
类（\ ``imperative/src/include/megbrain/imperative/op_def.h``\ ）：

.. code:: cpp

   template <typename T>
   class OpDefImplBase : public OpDef {
   public:
       template <typename... Args>
       static std::shared_ptr<T> make(Args&&... args) {
           return std::make_shared<T>(std::forward<Args>(args)...);
       }
   };

OpDef 类方法
^^^^^^^^^^^^^^^^^^

``OpDef`` 类定义在
``imperative/src/include/megbrain/imperative/op_def.h``\ ：

.. code:: cpp

   class OpDef : public Hashable,
                 public NonCopyableObj,
                 public std::enable_shared_from_this<OpDef> {
       mutable const OpTrait* m_trait = nullptr;
       std::string m_scope;

   public:
       using allocator_t =
               std::function<DeviceTensorStorage::RawStorage(CompNode, size_t)>;
       virtual ~OpDef() = default;

       static std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node);

       static DispatchMode decide_dispatch_mode(
               const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs);

       static SmallVector<TensorPtr> apply_on_physical_tensor(
               const OpDef& def, const SmallVector<TensorPtr>& inputs,
               SmallVector<LogicalTensorDesc>& output_descs, const bool& validated);

       static void apply_on_device_tensornd(
               const OpDef& def, const SmallVector<DeviceTensorND>& inputs,
               SmallVector<DeviceTensorND>* outputs);

       static cg::VarNodeArray apply_on_var_node(
               const OpDef& def, const VarNodeArray& inputs);

       static std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
               const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs);

       static EncodedSubgraph make_backward_graph(
               const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
               const SmallVector<bool>& input_requires_grad,
               const SmallVector<bool>& output_has_grad);

       static std::vector<std::pair<const char*, std::string>> props(const OpDef& def);

       static EncodedSubgraph make_forward_graph(
               const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs);

       static SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
               const OpDef& def, const SmallVector<TensorPtr>& inputs);

       const OpTrait* trait() const;

       std::string to_string() const;

       const std::string scope() const;

       const std::string make_name() const;

       void set_scope(const std::string& scope);

       virtual size_t hash() const;

       virtual bool is_same_st(const Hashable&) const;

       static void set_allocator(allocator_t allocator);
       DeviceTensorStorage::RawStorage allocate(CompNode, size_t) const;

       std::shared_ptr<OpDef> shared_from_this() const {
           return const_cast<OpDef&>(*this)
                   .std::enable_shared_from_this<OpDef>::shared_from_this();
       }
   };

``OpDef`` 类中定义了很多 ``static`` 方法：

-  ``make_from_op_node``

   将一个 ``MegBrain`` 中的 ``op`` 转为一个 ``OpDef`` 中的 ``op``。

-  ``decide_dispatch_mode``

   目前 ``MegEngine`` 支持两种 ``DispatchMode``\ ：

   .. code:: cpp

      enum DispatchMode { DEFAULT_CPU = 0, KERNEL = 1 };

   ``decide_dispatch_mode``
   的作用是判断在哪类设备上进行计算，也就是判断要进行 ``host`` 计算还是
   ``device`` 计算。

   ``host`` 和 ``device`` 的区别是：

   -  ``host`` 是同步、延迟较高的一类设备，如 ``CPU``\ ；
   -  ``device`` 是异步、延迟较低的一类设备，如 ``GPU``\ 。

-  ``apply_on_physical_tensor``

   真实地执行 ``op``\ ，也就是进行 ``op``
   计算。默认实现是在静态图中执行静态 ``op``\ ，也可以直接调用 ``dnn``
   的 ``kernel`` 执行 ``op``\ 。

-  ``apply_on_device_tensornd``

   调用对应的 ``dnn op`` 来计算结果。由于 ``apply_on_physical_tensor``
   也可以实现相同的效果，因此 ``apply_on_device_tensornd`` 使用较少。

-  ``apply_on_var_node``

   构造一个静态图 ``op``\ ，用于后续 ``MegBrain`` 中的组件。

   比如 ``infer_output_attrs_fallible`` 进行 ``shape`` 推导就需要用到
   ``OpDef`` 构造的静态图 ``op``\ 。

-  ``infer_output_attrs_fallible``

   在 ``interpreter`` 线程中推导输出 ``tensor`` 的
   ``shape``\ 、\ ``dtype``\ 、\ ``device`` 等信息，必要时提前报错。

-  ``make_backward_graph``

   根据前向记录的信息，构造一个反向的计算图，用于求导时使用。

-  ``make_forward_graph``

   在 ``Python`` 层的部分 ``op`` 没有对应的 ``cpp`` 实现，而是使用
   ``Python`` 层的其他 ``op`` 拼接而成，对于这类 ``op``\ ，会有一个
   ``make_forward_graph`` 根据组成这类 ``op`` 的 ``op`` 生成对应的图。

-  ``get_input_layout_constraint``

   判断输入 ``tensor`` 的 ``layout`` 是否满足限制。

imperative/src/impl/ops/ 下的 cpp 文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 ``imperative/src/impl/ops/`` 目录下有很多 ``cpp`` 文件：

.. code:: bash

   linrongjian@hhd-2:/data/MegBrain/imperative/src/impl/ops$ tree -L 1
   .
   ├── adaptive_pooling.cpp
   ├── atlas_runtime.cpp
   ├── autogen.cpp
   ├── backward_graph.cpp
   ├── batch_norm.cpp
   ├── broadcast.cpp
   ├── cambricon_runtime.cpp
   ├── collective_comm.cpp
   ├── cond_take.cpp
   ├── convolution.cpp
   ├── custom_opdef.cpp
   ├── deformable_conv2d.cpp
   ├── deformable_psroi_pooling.cpp
   ├── elemwise.cpp
   ├── extern_opr.cpp
   ├── group_norm.cpp
   ├── indexing.cpp
   ├── io_remote.cpp
   ├── lamb.cpp
   ├── layer_norm.cpp
   ├── magicmind_runtime.cpp
   ├── matmul.cpp
   ├── matrix_inverse.cpp
   ├── misc.cpp
   ├── nms.cpp
   ├── opr_attr.cpp
   ├── padding.cpp
   ├── pixel_shuffle.cpp
   ├── pooling.cpp
   ├── reduce.cpp
   ├── resize.cpp
   ├── rng.cpp
   ├── rnn.cpp
   ├── softmax.cpp
   ├── specializations.cpp
   ├── tensor_manip.cpp
   ├── tensorrt_runtime.cpp
   ├── utility.cpp
   ├── vision.cpp
   ├── warp_affine.cpp
   └── warp_perspective.cpp


这些文件一起做了两件事：

1. 为所有的 ``OpDef`` 定义 ``apply_on_var_node`` 方法，这个方法是
   ``OpDef`` 与静态图建立联系的关键，因此每一个 ``OpDef``
   都必须实现此方法；
2. 为某些 ``OpDef`` 重写特定的方法。

比如 ``padding.cpp`` 中实现了
``apply_on_var_node``\ 、\ ``apply_on_physical_tensor`` 还有
``infer_output_attrs_fallible`` 方法：

.. code:: cpp

   namespace padding {
   auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
       // apply_on_var_node implementation code
       ...
   }

   SmallVector<TensorPtr> apply_on_physical_tensor(
           const OpDef& def, const SmallVector<TensorPtr>& inputs,
           SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
       // apply_on_physical_tensor implementation code
       ...
   }

   std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
           const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
       // infer_output_attrs_fallible implementation code
       ...
   }

   // register function
   OP_TRAIT_REG(Padding, Padding, opr::Padding)
           .apply_on_var_node(apply_on_var_node)
           .apply_on_physical_tensor(apply_on_physical_tensor)
           .infer_output_attrs_fallible(infer_output_attrs_fallible)
           .fallback();

   }  // namespace padding

``OP_TRAIT_REG`` 宏用来注册 ``imperative op``
的方法实现，对于注册了的方法，使用的是我们 ``gen`` 出来的代码实现。

如果没有注册，则会使用 ``proxy_graph`` 中的方法。

OP_TRAIT_REG 与 struct OpTrait
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

上面提到，我们在实现 ``imperative`` 的 ``op`` 时有时需要重写
``apply_on_physical_tensor`` 等方法，这里需要用到 ``OP_TRAIT_REG`` 宏。

``OP_TRAIT_REG`` 宏为 ``OpDef`` 注册重写的方法，可供注册的方法写在了
``struct OpTrait``\ ：

.. code:: cpp

   struct OpTrait {
       const char* name;
       OpDefMaker make_from_op_node;
       DecideDispatchMode decide_dispatch_mode;
       ApplyOnPhysicalTensor apply_on_physical_tensor;
       ApplyOnDeviceTensorND apply_on_device_tensornd;
       ApplyOnVarNode apply_on_var_node;
       InferOutputAttrsFallible infer_output_attrs_fallible;
       GetInputLayoutConstraint get_input_layout_constraint;
       GradMaker make_backward_graph;
       Props props;
       HashFunc hash;
       IsSame is_same_st;
       MakeNameFunc make_name;
       GraphMaker make_forward_graph;
       OpTrait(const char* name);
       static OpTrait* find_by_name(const char* name);
       static OpTrait* find_by_typeinfo(Typeinfo* type);
       static void for_each_trait(thin_function<void(OpTrait&)> visitor);
   };

没有使用 ``OP_TRAIR_REG`` 注册的方法，则会默认使用 ``proxy_graph``
中的方法。

异步执行
~~~~~~~~~~~~

``Interpreter`` 代码中包含两个线程：\ ``interpreter`` 线程与 ``worker``
线程。

``Interpreter`` 线程对应 ``Python`` 主线程，\ ``worker``
线程是一个后台线程。

interpreter 线程
^^^^^^^^^^^^^^^^^^^^^^

1. ``interpreter`` 线程会接收 ``Python`` 侧发送的 ``op`` 执行命令。
2. 对于接收到的 ``op`` 指令，\ ``interpreter`` 线程会先做一些处理：

   -  调用 ``OpDef::decide_dispatch_mode`` 方法，根据 ``op`` 和
      ``inputs`` 判断是进行 ``host`` 计算还是 ``device``
      计算（\ ``host`` 和 ``device`` 的区别见 ``2.4.7`` 节）。一般大
      ``shape`` ``tensor`` 在 ``device`` 上计算更快，但是需要发送命令到
      ``device`` 并等待结果，小 ``shape`` ``tensor`` 在 ``host``
      上计算更快。提前对不同情况做判断可以对一些情况进行加速。
   -  调用 ``OpDef::infer_output_attrs_fallible`` 方法检查输出
      ``tensor`` 属性是否符合限制（比如输出的 ``target_shape``
      是否合法），提前报错。

3. 如果是 ``host`` 计算，则 ``interpreter`` 线程直接进行计算，如果需要
   ``worker`` 线程的计算结果，可能需要进行同步操作获取 ``worker``
   线程的计算结果；如果是 ``device`` 计算，\ ``interpreter``
   线程会把包含 ``op`` 和 ``inputs`` 的 ``cmd`` 发送给 ``worker``
   线程，由 ``worker`` 线程去分配显存并进行真正的计算。
4. 由于 ``interpreter`` 线程与 ``worker``
   线程是异步的关系，因此发送完命令之后 ``interpreter``
   线程可以去做其他的事，这样可以使代码执行更快。

worker 线程
^^^^^^^^^^^^^^^^^

1. ``worker`` 线程是后台线程，一直在等 ``interpreter`` 线程的命令。

   具体命令可以参考 ``imperative/src/impl/interpreter/commands.h``\ 。

2. 接收到命令之后，\ ``worker`` 线程会对输出 ``tensor``
   进行显存分配，并做真实的计算（\ ``OpDef::apply_on_physical_tensor``\ ）。

3. 计算完成后，将结果返回给 ``interpreter`` 线程。

在 ``worker`` 线程执行一个 ``op`` 的计算操作时，这时如果 ``interpreter``
发来了新的 ``op`` 指令，新的指令会在一个队列 ``WorkQueue`` 上排队，等
``worker`` 线程完成了一个任务后会从队列取走一个任务。

队列的长度被设置为足够大，如果出现了队列满的情况，则 ``interpreter``
线程会阻塞，等待 ``worker`` 线程执行完当前任务并从队列取走下一个任务。

完整流程
^^^^^^^^^^^^^^

一个 ``op`` 在 ``interpreter`` 执行的完整流程为：

.. figure:: ../../_static/images/imperative_execution.png


1. ``dispatch`` 层调用 ``eval.cpp`` 的
   ``InterpreterTransformation::apply_op`` 方法，对经过
   ``transformation`` 处理的 ``io`` 指令发送到 ``interpreter`` 层。

2. ``interpreter`` 线程调用 ``ChannelImpl::apply_op``
   方法（\ ``imperative/src/impl/interpreter/interpreter_impl.cpp``\ ），先对接收到的
   ``op`` 指令做一些处理：

   1. 调用 ``OpDef::infer_output_attrs_fallible`` 方法，检查输出
      ``tensor`` 的属性是否符合限制。

      1. 对于在 ``imperative/src/impl/ops/xxx.cpp`` 下实现并用
         ``OP_TRAIT_REG`` 宏注册了的 ``infer_output_attrs_fallible``
         方法，使用其特化实现。
      2. 否则，调用 ``imperative/src/impl/proxy_graph.cpp`` 下的泛化实现
         ``ProxyGraph::infer_output_attrs_fallible``\ 。

   2. 调用 ``OpDef::decide_dispatch_mode`` 方法，决定 ``dispatch_mode``
      是 ``DEFAULT_CPU`` 还是 ``KERNEL``\ 。前者对应 ``host``
      计算，后者对应 ``device`` 计算。

      1. 对于 ``host`` 计算，\ ``interpreter`` 线程无需发送 ``op``
         指令给 ``worker`` 线程，直接调用
         ``OpDef::apply_on_device_tensornd`` 和对应 ``op`` 的
         ``apply_on_physical_tensor`` 进行计算。

      2. 对于 ``device`` 计算，\ ``interpreter`` 将 ``op``
         指令发送到异步队列 ``WorkQueue`` 上（\ ``WorkQueue::add_task``
         方法），\ ``worker``
         线程空闲时从队列中取走一个任务（\ ``WorkQueue::process_one_task``
         方法）。

         ``Worker`` 线程在进行计算操作前处理一些相关命令：\ ``Tensor``
         相关（\ ``Put``\ 、\ ``Del``\ 、\ ``GetValue``\ ）、\ ``DTR``
         相关命令（\ ``Drop``\ ）、\ ``Profiler``
         相关命令（\ ``StartProfile``\ 、\ ``StopProfile``\ 、\ ``PushScope``\ 、\ ``PopScope``\ ）。

         调用 ``OpDef::apply_on_physical_tensor``\ ，进一步调用泛化的
         ``proxy_graph_detail.cpp`` 的
         ``apply_on_physical_tensor``\ （也就是复用 ``MegBrain``
         中的静态实现） 或者对应 ``op`` 的 ``cpp`` 文件特化了的
         ``XXX::apply_on_physical_tensor`` 方法（取决于是否实现了特化的
         ``apply_on_physical_tensor`` 方法），完成计算操作。

         由于 ``interpreter`` 线程需要的是一个
         ``Host Tensor``\ （在内存上），而 ``worker``
         线程完成计算后的结果是一个
         ``Device Tensor``\ （在显存上），因此\ ``worker`` 线程会调用
         ``Tensor::fetch_value()``
         方法（\ ``imperative/src/impl/physical_tensor.cpp``\ ）创建一个
         ``Host Tensor`` 并将计算结果写入到该 ``Tensor``\ ，再返回给
         ``interpreter`` 线程。

在 ``interpreter`` 运行过程中需要对当前正在执行的 ``op``
有所了解。比如为了提前报错需要正确执行当前 ``op`` 的
``infer_output_attrs_fallible`` 方法，显然不同的 ``op``
对这一方法的实现是不同的。

对于 ``OpDef`` 系统，其主要提供了一层抽象，使得所有 ``op``
对外都提供相同的接口方便 ``interpreter`` 调用，并且某些方法可以复用
``MegBrain``
中的静态实现，当然我们仍然可以自定义某些方法的行为来覆盖默认实现。
 