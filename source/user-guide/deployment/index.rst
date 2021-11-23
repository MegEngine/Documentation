.. _deployment:

=====================
将模型部署到 C++ 环境
=====================

目前不再使用 MegEngine 的 C++ 接口直接作为推理框架，使用 MegEngine 的 C++ 接口作为推理框架有以下缺点：

* 接口过于复杂，MegEngine 的 C++ 接口分散在各个文件夹下面，内容和目录结构都很复杂。
* MegEngine 的 C++ 接口对于推理暴露了很多内部的概念，不单纯。
* MegEngine 的很多优化的功能使用 MegEngine 的 C++ 接口过于复杂。

所以目前推理都推荐使用 MegEngine Lite，使用 MegEngine Lite 参考 :ref:`fast-develope-cpp`