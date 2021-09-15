.. _custom-op-guide:

=======================
自定义算子（Custom Op）
=======================

MegEngine 中提供了非常丰富的与机器学习、神经网络、张量计算等相关的函数与模块。
不过研究人员在开发模型的过程中，经常会去设计一些新的操作比如定义新的神经网络层（Neural Network Layer）等，MegEngine 需要提供给用户自定义这些操作的能力。

一般而言，研究人员可以使用 MegEngine 提供的 python 接口通过拓展 Function 和 Module 去实现其所需的功能。
同时，面向对性能要求比较高的用户，MegEngine 还另外提供给用户一套工具，可以将其自定义的 C++/CUDA 算子快速集成入 MegEngine，即 Custom Op.

下面将通过一个简单示例去展示编写 Custom Op 并将之集成入 MegEngine 的流程，之后将展示更具体的接口介绍。

整体流程
--------

现在我们需要为 MegEngine 添加一个名为 :class:`MatMulScale` 的算子，这个算子在计算时首先会对两个输入 Tensor，lhs 和 rhs 执行矩阵乘，然后再将这个矩阵乘的结果再乘以标量 Scale.

该算子数学上的执行过程的伪代码如下：

.. code-block::

   def MatMulScale(lhs, rhs, scale):
       result = lhs.dot(rhs)
       result = result * scale
       return result

对于这样的一个操作，假设我们已经为之写好了一份 CUDA kernel 代码，并提供如下的接口函数用于调用：

.. code-block:: cpp

   void matmul_scale(const float *lhs, const float *rhs, float *result, size_t M, size_t K, size_t N, float scale);

这些的参数中，``lhs``，``rhs``，以及 ``result`` 是三个 float 类型的指针，
分别代表这个 Op 的两个输入 ``Tensor`` 和一个输出 ``Tensor``，其均需要指向一片已经分配好的 cuda memory.
而 ``M``，``K``，``N`` 是矩阵的维度信息，表示一个 ``M*K`` 的矩阵乘以一个 ``K*N`` 的矩阵。
而 ``scale`` 则代表着矩阵乘的结果需要乘以的那个系数。

对于这种情况我们可以编写如下的 C++ 代码，就可以将之封装成 MegEngine 的 Op。

.. code-block:: cpp

   #include "megbrain/custom/custom.h"

   CUSTOM_OP_REG_BEGIN(MatMulScale)

   void shape_infer(const std::vector<Shape> &inputs, const Param &params, std::vector<Shape> &outputs) {
       outputs[0] = {inputs[0][0], inputs[1][1]};
   }

   void compute(const std::vector<Tensor> &inputs, const Param &params, std::vector<Tensor> &outputs) {
       matmul_scale(                       // 调用 kernel 的接口函数
           inputs[0].data<float>(),        // lhs
           inputs[1].data<float>(),        // rhs
           outputs[0].data<float>(),       // result
           inputs[0].shape()[0],           // M
           inputs[0].shape()[1],           // K
           inputs[1].shape()[1],           // N
           params["scale"].as<float>()     // scale
       );
   }

    CUSTOM_OP_REG(MatMulScale)              // 定义一个名为 MatMulScale 的 Op
        .add_inputs(2)                      // 两个输入 Tensor
        .add_outputs(1)                     // 一个输出 Tensor
        .add_param("scale", 1.0f)           // 一个名为 scale 的 Parameter，默认值为 1.0f
        .set_shape_infer(shape_infer)       // 设置这个 Op 的 shape 推导函数
        .set_compute("cuda", compute);      // 设置这个 Op 的 计算函数

    CUSTOM_OP_REG_END(MatMulScale)

这段代码中，其首先 include Custom Op 头文件，然后使用两个宏 ``CUSTOM_OP_REG_BEGIN()`` 和 ``CUSTOM_OP_REG_END()`` 构建了一段 scope.
在这个 scope 中，我们可以编写 Custom Op 的主体代码，而这个主体代码分为两个部分。
第一个部分是一些函数的定义，包括输出 ``Tensor`` 属性推断函数和计算函数。
其中前者会根据输入 ``Tensor`` 的属性（比如 ``shape``）去推导输出 ``Tensor`` 的对应属性，而后者则是在其中调用 CUDA kernel，完成计算。
第二部分是 Op 的注册，主要用于定义 Op 有几个输入输出 ``Tensor``，有几个 ``Param``，并将上面定义的属性推断函数和计算函数的指针也注册给 Op.

之后可以使用 Custom Op 所提供的 makefile 模板将 CUDA kernel 以及上面的 C++ 文件一起编译成一个库文件，我们将之命名为 matmul_scale.so.
然后在 python 中，我们可以编写如下的代码去使用它：

.. code-block::

   from megengine.core._imperative_rt.core2 import apply
   from megengine.core.ops import custom
   from megengine.tensor import Tensor
   import numpy as np

   custom.load("matmul_scale.so")         # 加载我们所编译出来的库
   op = custom.MatMulScale(scale = 0.1)   # custom.your_op_name，就是我们在 C++ 中定义的那个 Op 的名字
   lhs = Tensor(np.random.uniform(size=(128, 256)))
   rhs = Tensor(np.random.uniform(size=(256, 512)))
   result = apply(op, lhs, rhs)

当然，我们也可以将 Custom Op 与 MegEngine 已有的 Python 组件如 ``autodiff.Function`` 以及 ``module.Module`` 结合起来，以支持训练和构建更大规模的模型：

.. code-block::

   from megengine.autodiff import Function
   from megengine.module import Module

   class MatMulScaleFunc(Function):        # 将我们定义的 Op 包装成 autodiff.Function 以支持反向训练
       def __init__(self, scale):
           super().__init__()
           self.scale = scale

       def forward(self, lhs, rhs):
           self.lhs = lhs
           self.rhs = rhs
           op = custom.MatMulScale(scale=self.scale)   # custom.your_op_name，就是我们在 C++ 中定义的那个 Op 的名字
           return apply(op, lhs, rhs)

       def backward(self, ograd):                              # 这里假设我们又定义了另一个 Custom Op MatMulScaleBackward
           op = custom.MatMulScaleBackward(scale=self.scale)   # 其完成了 MatMulScale 的反向计算，出于篇幅限制就不展示其 C++ 代码
           return apply(op, ograd, self.lhs, self.rhs)

   class MatMulScaleModule(Module):                            # 进一步将上面的 autodiff.Function 封装成 Module
       def __init__(self, ic, oc, scale, **kwargs):
           super().__init__(**kwargs)
           self.scale = scale
           self.weight = Parameter(np.zeros(shape=(ic, oc), dtype=np.float32))
           self.func = MatMulScaleFunc(scale=scale)

       def forward(self, inp):
           return self.func(inp, self.weight)


接口介绍
--------

属性推断函数
~~~~~~~~~~~~

Custom Op 的输出 ``Tensor`` 属性推导主要是根据输入 ``Tensor`` 的一些属性（``Shape``，``DType``，``Device``）以及 Op 的参数来计算输出 ``Tensor`` 的对应相关属性。
其中 ``Shape`` 代表的是 ``Tensor`` 维度信息，``DType`` 对应 Tensor 的数据类型，``Device`` 表示这个 ``Tensor`` 在什么设备（cpu/gpu）上。
比如卷积中我们可以根据输入 ``Tensor`` 的 ``Shape`` 以及 ``stride``，``padding`` 等参数计算出输出 ``Tensor`` 的 ``Shape`` 信息。

这些输出属性推导的过程目前需要使用者以 C++ 函数的形式给出，而这些函数的函数签名（即函数的输入参数与返回值的类型）是固定的，其分别如下：

.. code-block:: cpp

   void(*)(const std::vector<Device>&, const Param&, std::vector<Device>&);    // device infer
   void(*)(const std::vector<Shape>&,  const Param&, std::vector<Shape>&);     // shape infer
   void(*)(const std::vector<DType>&,  const Param&, std::vector<DType>&);     // dtype infer

我们编写自己 Custom Op 的相关属性推导函数时需要确保自己的相关函数的函数签名应该与上述例子中对应函数的函数签名保持一致。
这几个函数的函数签名基本是类似的，以 ``Shape`` 推导来说，其参数传入了输入的 ``Tensor`` 的 ``Shape`` 信息和其 ``param``，以及输出 ``Shape`` 的引用。
其中这两个 ``vector`` 的长度即分别为输入 ``Tensor`` 的数量和输出 ``Tensor`` 的数量。
我们在这个函数中可以计算出输出 ``Tensor`` ``Shape``，并将之赋值给对应引用。

**Device**

目前 Custom Op 支持的 ``Device`` 支持的设备类型包括 ``x86`` 和 ``cuda``.
我们可以像使用字符串的方式去使用它，下面是几个 ``Device`` 的使用案例。

.. code-block:: cpp

   Device device = "x86";                  // 创建一个 x86 这种设备类型
   device = "cuda";                        // 设备类型改为 cuda
   bool equal = (device == "cuda");        // 判断某个 device 是否是 cuda
   std::string device_str = device.str();  // 获取 device 对应的可读的字符串表示

而 Custom Op 还为输出 ``Tensor`` 的 ``Device`` 类型推导提供了一种默认的行为，即所有输出 ``Tensor`` 的 ``Device`` 都与第 0 个输入 ``Tensor`` 的 ``Device`` 类型相等。
如果没有输入 ``Tensor``，则所有输出 ``Tensor`` 的 ``Device`` 都为 ``x86``.
而在上面的 ``MatMulScale`` 的例子中，我们并没有为之定义 ``Device`` 推导函数，故而其就使用了这种默认的 ``Device`` 推导行为。

**DType**

目前 Custom Op 支持的 ``DType`` 支持的设备类型包括 ``float16``，``bfloat16``，``float32``，``uint8``，``int8``，``int16``，``uint16``，``int32``，
以及四种量化类型``qint8``，``quint8``，``qint16``，``qint32``.其中 ``quint8`` 是非对称量化数据类型，而其他三者是对称量化数据类型。
我们也可以像使用字符串的方式去使用它，下面是几个 ``DType`` 的使用案例。

.. code-block:: cpp

   DType dtype1 = "float32", dtype2 = "int8";  // 定义两个 dtype
   bool equal = (dtype1 == dtype2);            // 判断这两个 dtype 是否相等
   dtype1 = "int16";                           // 修改 dtype1 的数据类型
   std::string dtype_str = dtype1.str();       // 获取 dtype1 对应的可读的字符串类型表示

   DType dtype3("qint8", 0.32);                // 创建一个 scale 为 0.32 的对称 8bit 量化的数据类型
   DType dtype4("quint8", 0.32, 32);           // 创建一个 scale 为 0.32，zero_point 为 32 的非对称 8bit 量化的数据类型

   float scale = dtype3.scale();               // 获取 dtype3 的 scale
   uint8_t zero_point = dtype4.zero_point();   // 获取 dtype4 的 zero_point

与 ``Device`` 类似，而 Custom Op 也为输出 ``Tensor`` 的 ``DType`` 类型推导提供了一种默认的行为，即所有输出 ``Tensor`` 的 ``DType`` 都与第 0 个输入 ``Tensor`` 的 ``DType`` 类型相等。
如果没有输入 ``Tensor``，则所有输出 ``Tensor`` 的 ``DType`` 都为 ``float32``.
而在上面的 ``MatMulScale`` 的例子中，我们同样并没有为之定义 ``DType`` 推导函数，故而其也使用了这种默认的 ``DType`` 推导行为。

**Shape**

在 Custom Op 中我们可以以类似于 vector 或 C++ 原生数组的方式去构建和使用 ``Shape``，下面是几个 ``Shape`` 的使用案例。

.. code-block:: cpp

   Shape shape1 = {16, 3, 224, 224}, shape2 = {16, 32};    // 创建两个 shape
   bool equal = (shape1[3] == 224);                        // 获取 shape1 中第 3 个维度的长度，并进行比较
   shape2[1] = 16;                                         // 对 shape2 中第 2 个维度的长度进行修改
   shape1 = {16, 16};                                      // 让 shape1 等于一个新的 shape 值
   bool equal = (shape1 == shape2);                        // 判断两个 shape 是否相等
   size_t ndim = shape1.ndim();                            // 获取 shape1 一共有几个维度

Custom Op 也为 ``Shape`` 推导提供的默认的行为是，让所有输出 ``Tensor`` 的 ``Shape`` 都与第 0 个输入 ``Tensor`` 的 ``Shape`` 类型相等。
如果没有输入 ``Tensor``，则所有输出 ``Tensor`` 的 ``Shape`` 都为 ``[1]``.
而在上面的 ``MatMulScale`` 的例子中，显然默认的 ``Shape`` 推导函数不符合我们的需求，所以我们自行定义了我们同样并没有为之定义 ``DType`` 推导函数，故而其也使用了这种默认的 ``DType`` 推导行为。

计算函数
~~~~~~~~

Custom Op 的计算函数的主要功能其实就是如何调用我们已经编写好的 Kernel 的接口函数。
这些过程也是需要使用者以 C++ 函数的形式给出，而这个函数的函数签名也是固定的：

.. code-block:: cpp

   void(*)(const std::vector<Tensor>&, const Param&, std::vector<Tensor>&);

同样的 Custom Op 的计算函数并无返回值，该函数传入输入 ``Tensor`` 以及 ``Param``，然后计算出输出 ``Tensor`` 的值并将之作为引用返回。
这里主要涉及到两个概念，分别是 ``Tensor`` 和 ``Param``，下面将分别对其进行介绍。

**Tensor**

Custom Op中的 ``Tensor`` 可以视为数据（``data``）以及数据的属性（即上面 ``Device``，``DType``，``Shape``）的集合。
我们可以用下面的代码去获取 ``Tensor`` 的相关信息：

.. code-block:: cpp

   Device device = tensor.device();                    // 获取 tensor 的 device 信息
   DType dtype = tensor.dtype();                       // 获取 tensor 的 dtype 信息
   Shape shape = tensor.shape();                       // 获取 tensor 的 shape 信息

   size_t size = tensor.size();                        // 获取 tensor 中元素的数量
   std::vector<ptrdiff_t> strides = tensor.stride();   // 获取 tensor 中各个维度的 stride
   float scale = tensor.scale();                       // 获取 tensor 中数据的 scale，只在量化数据中有效
   uint8_t zero_point = tensor.zero_point();           // 获取 tensor 中数据的 zero_point，只在非对称量化数据中有效

我们使用上述函数获取 ``Tensor`` 的相关属性如 ``Device``，``DType``，``Shape``，或者是一些更细节的信息如 ``Tensor`` 中元素的数量，``Tensor`` 中各个维度的 ``stride`` 等。
然后我们可以利用这些信息来帮助我们进行 kernel 的编写。

另外我们可以使用下面的代码去获取 ``Tensor`` 中所存储的数据：

.. code-block:: cpp

   void *data = tensor.data();
   float *float_data = tensor.data<float>();

这里提供了两个 ``data()`` 函数，分别是不支持模板参数的和支持模板参数的，这两者均会返回实际数据的指针。

其中前者返回的是 ``void*`` 类型，我们使用时可以将之强制成转换成自己所需的实际类型，这提供给我们自行定义自己数据类型的能力。

而后者返回的是模板参数所指定的类型的指针，比如在此例中模板参数是 ``float``，所以其返回 ``float*`` 类型的指针。
在这种情况下，Custom Op 会检测模板参数类型的正确性，即此时 ``Tensor`` 中实际存储的数据类型也必须是 ``float`` 类型，否则就会出错。
而获取到的指针则指向一片这个 ``Tensor`` 的 ``Device`` 属性所对应的设备上的内存。

在获取到这个原始指针之后，结合上面可以获取的诸如 ``Shape``，``stride`` 之类的信息，我们就可以去正常的去计算各个元素的下标，读取/存储数据，编写 kernel，完成计算。
不过下标计算总是繁琐而容易出错的，故而 Custom Op 中还提供了一个叫 ``TensorAccessor`` 的工具，允许我们可以以类似于 C++ 数组的方式访问 ``Tensor`` 中的对应元素。
下面这段代码展示了如何使用 ``TensorAccessor`` 去访问一个 4 维 ``Tensor`` 中第 ``(n, c, h, w)`` 个元素

.. code-block:: cpp

   auto accessor = tensor.accessor<float, 4>();        // 获取 accessor
   accessor[n][c][h][w] = 1.f;                         // 根据 accessor 访问对应的元素
   float val = accessor[n][c][h][w];

这里的 ``accessor()`` 函数一般需要提供两个模板参数，其中第一个参数表示 ``Tensor`` 的数据类型，第二个参数表示 ``Tensor`` 的维度。
在此例中，因为 ``tensor`` 是一个 ``float`` 类型的 4 维 ``Tensor``，故而此处这两个模板参数分别为 ``float`` 和 ``4``.

如果想要使用 ``TensorAccessor`` 的话，我们可以将之作为 kernel 的参数传递给 kernel，然后在 kernel 内部去使用 accessor 去访问数据。
当然，使用 ``TensorAccessor`` 相对于自行计算元素下标会引入一点额外的 overhead，大家可以根据自己的需要选择是否使用 ``TensorAccessor``.

最后需要强调的一件事情是，为了方便进行内存管理，目前在 Custom Op 的代码中是不允许自己构造 ``Tensor`` 的。
MegEngine 中会自动的为 Custom Op 构造 ``Tensor``，分配内存，然后将构造好的 ``Tensor`` 传递给我们，我们再调用上述接口对 ``Tensor`` 进行操作。

**Param**

``Param`` 用于记录 Custom Op 的一些非 ``Tensor`` 的输入，比如卷积中的 padding，stride 等等。
其实际上是一个 ``map``，其 ``key`` 为 ``std::string`` 类型，表示某个 ``param`` 元素的名字,
而 ``value`` 为 ``ParamVal`` 类型，这个类可视为一个支持有限类型的 Any.
通过下面的代码可以简单的展示 ``ParamVal`` 的一些特性：

.. code-block:: cpp

   ParamVal a = 1, b = 1.0, c = true, d = "string";    // 可以将各种类型的数据直接赋值给 ParamVal
   ParamVal e = {1, 2, 3, 4};                          // 支持 std::vector

   ParamVal f = a + b;                                 // ParamVal 可以进行四则运算，计算结果仍然是 ParamVal类型
   ParamVal g = d + "abc";                             // ParamVal 可以和 C++ 内置类型直接进行计算

   bool equal = (a == b);                              // ParamVal 可以进行比较运算，计算结果是 bool 类型
   a = "string";                                       // ParamVal 在运行时改变其元素的实际类型
   std::string str = a.as<std::string>();              // ParamVal 转成 C++ 类型

目前 ``ParamVal`` 支持的类型包括 ``int32_t``，``uint32_t``，``int64_t``，``uint64_t``，``float``，``double``，``bool``，``std::string``，以及这些类型对应的 ``std::vector`` 类型（比如 ``std::vector<int32_t>``）。

``Param`` 可以使用 ``[]`` 运算符去根据名字获取 ``Param`` 中对应元素（``ParamVal`` 类型），我们可以以如下的方式去读写其中的数据：

.. code-block:: cpp

   param["scale"] = 0.1;                       // 将 param 中名为 scale 的元素值置为 1
   float scale = param["scale"].as<float>();   // 用 param 中名为 scale 的元素为 float 进行赋值

Custom Op 的注册
~~~~~~~~~~~~~~~~

上面我们为 Custom Op 定义了诸如属性推导函数，计算函数等信息，然而这些信息是彼此孤立的，Custom Op 的注册会将这些信息组合成一个整体。

**Op 的注册**

我们为 Custom Op 提供了一个宏，``CUSTOM_OP_REG(your_op_name)``，使用这个宏我们可以定义一个指定名字的 Custom Op.

.. code-block:: cpp

   CUSTOM_OP_REG(MatMulScale);     // 定义了一个名为 MatMulScale 的 Op

**为 Op 添加输入输出**

我们可以使用 ``add_input()`` 函数为 Op 添加一个输入 ``Tensor``，使用 ``add_output()`` 函数为 Op 添加输出 ``Tensor`` 的信息。
也可以使用 ``add_inputs()`` 和 ``add_outputs()`` 去批量添加输入输出。

.. code-block:: cpp

   CUSTOM_OP_REG(MatMulScale)
       .add_input("lhs", {"float32"}, 2)       // 为 Op 添加一个输入，名为 lhs，数据类型为 float32，维度为 2
       .add_input("rhs")                       // 使用 add_input 的默认行为，数据类型为 float32，维度为 -1，表示可以是任意维度
       .add_output("result", {"float32"}, 2)   // 为 Op 添加一个输出

   // 另一种注册输入输出 Tensor 的方式，批量注册
   CUSTOM_OP_REG(MatMulScale)
       .add_inputs(2)      // 为 Op 添加两个默认的输入，数据类型为 float32，维度为 -1
       .add_outputs(1)     // 为 Op 添加一个默认的输出，数据类型为 float32，维度为 -1

**为 Op 添加 Param**

我们可以使用 ``add_param()`` 函数为 Op 添加一个 ``Param`` 元素，其示例代码如下：

.. code-block:: cpp

   CUSTOM_OP_REG(MatMulScale)
       .add_param("scale", 1.0f);  // 为 Op 添加一个名为 scale 的参数，其默认值为 1.0f

在这里我们为 ``MatMulScale`` Op 添加了一个名为 "scale" 的参数，其默认值为 1.0f，
之后我们就可以在我们的相关属性推导函数和计算函数中使用 param["scale"] 去访问这个参数。

**为 Op 添加属性推导与计算函数**

对于属性推导函数的添加，Custom Op 提供了 ``set_shape_infer()``，``set_device_infer()``，
``set_dtype_infer()`` 三个函数分别用于设置 ``Shape``，``Device``，``DType`` 的属性推导函数。
而对于计算函数，Custom Op 提供了 ``set_compute()`` 函数用于设置进行设置。
其中属性推导函数只可以调用相关接口添加一次，而 ``set_compute()`` 函数则可以多次调用以添加不同平台上的计算函数。
相关示例代码如   

.. code-block:: cpp

   CUSTOM_OP_REG(MatMulScale)
       .set_shape_infer(matmul_scale_shape_infer)      // 为 Op 添加 Shape 推导函数
       .set_dtype_infer(matmul_scale_dtype_infer)      // 为 Op 添加 DType 推导函数
       .set_device_infer(matmul_scale_device_infer)    // 为 Op 添加 Device 推导函数
       .set_compute("x86", matmul_scale_compute_x86)   // 为 Op 添加 x86 上的计算函数
       .set_compute("cuda", matmul_scale_compute_cuda) // 为 Op 添加 cuda 上的计算函数

在这里 ``MatMulScale`` 算子并未使用默认的属性推导函数，而是分别调用相关接口为 ``Shape``，``Device``，``DType`` 的属性推导函数另行做了设置。
同时，这里还分别设置了 ``MatMulScale`` 在 ``x86`` 和 ``cuda`` 上的计算函数。
