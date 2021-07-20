.. _megdnn:

======
MegDNN
======

MegDNN 是 MegEngine 的底层计算引擎，位于 :src:`dnn` 目录。
它提供了与 dense tensor 相关的计算原语，比如 Convolution, Pooling, MatrixMul, Transpose 等。
它是跨平台的计算库，支持 x86 (with SSE4.2), arm (with NEON), CUDA (Kepler/Maxwell/Pascal)， OpenCL(mali/adreno/powervr/cuda/intel, etc), Hexagon 等。

.. _megdnn-organize:

MegDNN 文件结构
---------------

.. code-block:: shell

   dnn
   ├── atlas-stub
   ├── cuda-stub 
   ├── include
   │   ├── megdnn 
   │   │   ├── config
   │   │   ├── dtype # 定义数据类型
   │   │   ├── internal  # 仅供 MegDNN 内部实现使用的头文件
   │   │   │   ├── defs.h  # 定义了与 TensorND 的维数（ndim）相关的宏
   │   │   │   ├── opr_header_epilogue.h  # 定义 MegDNN operator 需要的宏
   │   │   │   ├── opr_header_prologue.h  # 将 opr_header_prologue.h 中的宏 undef 掉
   │   │   │   ├── visibility_epilogue.h  # 定义了与 visibility 相关的宏
   │   │   │   └── visibility_prologue.h  # 将 visibility_prologue.h 中定义的宏 undef 掉
   │   │   ├── oprs  # 定义了 MegDNN 的所有 operator
   │   │   ├── thin  # 相较于 std 的一些简化实现
   │   │   ├── arch.h  # 定义了一些平台与编译器相关的宏
   │   │   ├── basic_types.h  # 定义了一些 MegDNN 基础类型，如 ErrorHandler/TensorShape/TensorLayout/TensorND/Workspace 等
   │   │   ├── common.h
   │   │   ├── cuda.h  # 定义了一些 CUDA 特有的 API
   │   │   ├── dtype.h
   │   │   ├── handle.h 
   │   │   ├── opr_param_defs.h  # 定义了一些结构体，表示 operator 的 param （由 scripts/gen_param_defs.py 自动生成）
   │   │   ├── opr_result_defs.h  # 定义了一些结构体，表示 operator 的 result （由 scripts/gen_param_defs.py 自动生成）
   │   │   ├── oprs.h  # 包含了 oprs 文件夹下的所有头文件
   │   │   ├── tensor_format.h 
   │   │   ├── tensor_iter.h # 定义了 TensorIter 类，方便枚举 TensorND 内的所有元素。 
   │   │   └── version.h  # 定义了 MegDNN 的 version
   │   ├── hip_header.h 
   │   ├── megcore_*.h  # MegCore 是一个跨平台线程管理库，将 CUDA "device" 和 "stream" 的概念与 CPU NUMA（尚未实现）和线程的概念抽象出来。
   │   └── megdnn.h # MegDNN 的主头文件，MegDNN 的用户应使用此头文件。 
   ├── scripts 
   ├── src 
   │   ├── aarch64
   │   ├── arm_common
   │   ├── armv7
   │   ├── atlas
   │   ├── cambricon
   │   ├── common # 定义了各个平台的公共代码，比如各个 operator 的 deduce_layout, OperatorBase, Handle 的方法定义。
   │   ├── cuda
   │   ├── fallback
   │   ├── naive
   │   ├── rocm
   │   ├── x86
   │   └── CMakeLists.txt
   ├── test
   └── CMakeLists.txt
