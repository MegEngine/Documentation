.. _lar-basic-usage:

Load and run 简单入门
==================================

Load and run 作为 MegEnigne 提供的模型推理工具，在模型性能分析，精度抖动排查，模型推理优化等推理相关场景中应用广泛。

这里主要介绍一些 Load and run 的简单使用方法。

使用 Load and run 对给定输入的模型进行推理
----------------------------------------------

Load and run 可以提供快速的模型推理，获取相关推理性能。
使用时首先需要获取可用模型，参考 :ref:`get-model` 。
得到模型后需要设置合理的输入，如果不设置，Load and run 默认会给一个初始输入，输入的值不确定。

.. warning::

   部分模型输入不合理时，可能会运行时出错。

输入相关信息可以通过 ``--model-info`` 查看。

.. code:: bash

   load_and_run <model_path> --model-info 

用户可以使用相关信息自定义合适的输入数据并给到 Load and run 进行推理。有关输入数据的更多信息，参考 :ref:`basic-options`

.. code:: bash

   load_and_run <model_path> --input "input_data_name:input_data;...;input_data_name_n:input_data_n"

.. note::

   默认情况下，推理过程会包括一次模型预热以及 10 次正式推理作为统计推理信息的基本设置。

推理时会展示如下信息：

.. code:: bash

   [15 12:16:26 @:19][WARN] creat mdl model use XPU as default comp node
   load model: 234.236ms
   [15 12:16:26 @:121][WARN] run testcase: 0 
   === prepare: 1.509ms; going to warmup
   
   warm up 0  355.658ms
   iter 0/10: 141.249 ms (exec=1.066 ms)
   iter 1/10: 128.852 ms (exec=1.016 ms)
   iter 2/10: 126.601 ms (exec=1.003 ms)
   iter 3/10: 134.260 ms (exec=0.967 ms)
   iter 4/10: 160.764 ms (exec=1.082 ms)
   iter 5/10: 132.226 ms (exec=1.306 ms)
   iter 6/10: 126.292 ms (exec=0.947 ms)
   iter 7/10: 127.044 ms (exec=0.855 ms)
   iter 8/10: 126.430 ms (exec=1.006 ms)
   iter 9/10: 127.292 ms (exec=0.849 ms)

   === finished test #0: time=1331.010 ms avg_time=133.101 ms sexec=10.829 ms min=126.292 ms max=160.764 ms

   === total time: 1331.010ms

主要信息包括运行时默认的推理后端（这里为 ``XPU``，即 CPU 或 GPU，优先使用 GPU )，
模型加载用时，模型预热用时，模型多次迭代中各次迭代的用时以及算子执行用时，模型正式推理的统计信息（包括 ``总用时``，``平均用时`` ，``用时标准差`` ，``用时最大最小值`` ）。


设置模型推理以及预热次数
----------------------------------------------
有些模型在运行时，因为硬件环境原因（如其他进程计算资源高占用，CPU降频等），测试得到的数据不一定能反应真正的推理性能。
增加预热以及正式推理的次数，有些时候能够更加真实的反应特定硬件环境下的模型推理性能。

Load and run 提供了 ``--iter`` , ``--warmup-iter`` 用于设置模型预热和正式推理的次数。

.. code:: bash

   load_and_run <model_path> --input "input_data_name:input_data;..." --iter <iter_number> --warmup-iter <warmup_number>

这时会执行给定次数的模型预热过程以及模型推理过程，

.. note::

   一般模型预热次数不会设置很大， ``2~3`` 次的预热足以达到预期的预热效果。

使用 Lite 接口进行模型推理
----------------------------------------------

.. versionadded:: 1.7

   Load and run 集成了对 MegEngine Lite 接口的支持，可以使用 Load and run 对 MegEngine Lite 相关的模型进行推理测试, 使用时，只需要设置 ``--Lite`` 接口即可。

.. code:: bash

   load_and_run <model_path> --input "input_data_name:input_data;..." --Lite

这时会调用 MegEngine Lite 提供的 C++ 接口进行模型推理。

.. note::

   - Load and run 默认情况下使用 MegEngine 相关的接口进行推理，使用时推理后端默认为 ``XPU``，推理运行时采用 ``host-device`` 的结构作为后端架构。会根据后端硬件特征自动选择相应的推理设备。
   - Load and run 使用 MegEngine Lite 进行推理时，默认的推理后端是 CPU，相应后端设置需要通过相关选项自主定义（ ``--Lite`` 要同时开启），参考 ``device-options`` 。
   - MegEngine Lite 推理接口，对于部分设置选项目前没有相应实现。相关实现请关注 MegEngine Lite 的接口更新。