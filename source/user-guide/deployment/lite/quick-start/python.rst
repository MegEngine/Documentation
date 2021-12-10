.. _lite-quick-start-python:

======================================
MegEngine Lite Python 部署模型快速上手
======================================

.. note::

   * Lite 的 Python 包是和 MegEngine 本体绑定在一起的，所以只需要 :ref:`安装 MegEngine 包 <install-with-pip>` 即可使用。
   * 相较于 C++ 推理，该方法可以直接加载导出的模型并执行推理，无需经历复杂的编译环节。

.. warning::

   但是这种方式目前只支持 Windows/MacOS/Linux x86 和 CUDA 版本，不支持 Android Arm.

本文将从获取一个训练好的 ``shufflenet_v2`` 模型出发，讲解如何使用 MegEngine Lite 的 Python 接口将其部署到 Linux x86 中环境下运行。
主要分为以下小节：

* :ref:`lite-model-dump-python`
* :ref:`lite-infer-code-python`

.. _lite-model-dump-python:

导出已经训练好的模型
--------------------

首先我们需要 :ref:`dump` ，主要使用 MegEngine 的 :class:`~.jit.trace` 和 :meth:`~.trace.dump` 功能将动态图转为静态图，
同时对静态图进行 Inference 相关的图优化。
下面将要用到的模型为 MegEngine 预训练的模型，来自 `模型中心 <https://megengine.org.cn/model-hub>`_ 。 
安装 MegEngine 之后运行下面的 Python 脚本将 dump 出一个预训练的 ``shufflenet_v2.mge`` 模型。

.. code-block:: python

   import numpy as np
   import megengine.functional as F
   import megengine.hub
   from megengine import jit, tensor

   if __name__ == "__main__":
       net = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
       net.eval()

       @jit.trace(symbolic=True, capture_as_const=True)
       def fun(data, *, net):
           pred = net(data)
           pred_normalized = F.softmax(pred)
           return pred_normalized

       data = tensor(np.random.random([1, 3, 224, 224]).astype(np.float32))

       fun(data, net=net)
       fun.dump("shufflenet_v2.mge", arg_names=["data"])

上面代码最后 dump 模型时将模型命名为 ``shufflenet_v2.mge`` 并设置输入 Tensor 的名字为 ``data``, 后续将通过这个名字获取模型的输入 Tensor.

.. _lite-infer-code-python:

编写并执行 Inference 代码
-------------------------

首先创建一个 ``inference.py``, 在这个文件中将直接调用 MegEngine Lite 的 Python 接口运行 ``shufflenet_v2.mge`` 模型，
注意输入数据 ``input_tensor`` 是随机生成的，不用在乎计算结果。

.. code-block:: python
   :linenos:

   from megenginelite import *
   import numpy as np
   import argparse
   
   def main():
      parser = argparse.ArgumentParser(description='test the lite python interface')
      parser.add_argument('input', help='the inference model file')
      args = parser.parse_args()

      network = LiteNetwork()
      network.load(args.input)
   
      input_tensor = network.get_io_tensor("data")
      # copy input data to input_tensor of the network
      input_data = np.random.uniform(0, 1, (1, 3, 224, 224)).astype("float32")
      input_tensor.set_data_by_copy(input_data)
   
      # forward the model
      network.forward()
      network.wait()
   
      output_names = network.get_all_output_name()
      output_tensor = network.get_io_tensor(output_names[0])
   
      output_data = output_tensor.to_numpy()
      print('shufflenet output max={}, sum={}'.format(output_data.max(), output_data.sum()))
   
   if __name__ == '__main__':
      main()

上面代码主要完成了几个步骤，包括：

* 创建默认配置的 Network （第 10 行）；
* 载入模型，MegEngine Lite 将读取并解析模型文件，并创建计算图（第 11 行）；
* 通过输入 Tensor 的名字获取模型的输入 Tensor, 并设置随机数作为输入数据（第 13~16 行）
* 执行 Inference 逻辑（第 19~20 行）；
* 获取模型输出 Tensor, 并处理输出数据（第 22~426 行）。

这样这个调用 MegEngine Lite 的 Python 接口的 demo 就完成了。

