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

请参考 :ref:`get-model`。

.. _lite-infer-code-python:

编写并执行 Inference 代码
-------------------------

首先创建一个 ``inference.py``, 在这个文件中将直接调用 MegEngine Lite 的 Python 接口运行 ``shufflenet_v2.mge`` 模型，
注意输入数据 ``input_tensor`` 是随机生成的，不用在乎计算结果。

.. code-block:: python

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

* 创建默认配置的 Network；
* 载入模型，MegEngine Lite 将读取并解析模型文件，并创建计算图；
* 通过输入 Tensor 的名字获取模型的输入 Tensor, 并设置随机数作为输入数据；
* 执行 Inference 逻辑；
* 获取模型输出 Tensor, 并处理输出数据。

这样这个调用 MegEngine Lite 的 Python 接口的 demo 就完成了。

使用 CUDA 进行推理
-----------------------------------------------------
下面将通过 CUDA 运行 shufflenet.mge 来展示如何使用 TensorBatchCollector 来攒 batch，攒好之后传递到 network 的输入 tensor 中进行推理。

.. code-block:: python

   from megenginelite import *
   import numpy as np
   import argparse

   def main():
      parser = argparse.ArgumentParser(description='test the lite python interface')
      parser.add_argument('input', help='the inference model file')
      args = parser.parse_args()

      # construct LiteOption
      net_config = LiteConfig(device_type=LiteDeviceType.LITE_CUDA)

      network = LiteNetwork(config=net_config)
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

上面示例主要演示在 CUDA 设备上进行 Inference 的接口调用过程。

使用 TensorBatchCollector 辅助完成 CUDA 推理
-----------------------------------------------------
下面将通过 CUDA 运行 shufflenet.mge 来展示如何使用 TensorBatchCollector 来攒 batch，攒好之后传递到 network 的输入 tensor 中进行推理。

.. code-block:: python

   from megenginelite import *
   import numpy as np
   import os

   def test_network():
      model_path = "shufflenet.mge"
      batch = 4

      # construct LiteOption
      net_config = LiteConfig(device_type=LiteDeviceType.LITE_CUDA)

      # constuct LiteIO, is_host=False means the input tensor will use device memory
      # set the input tensor "data" memory is not from host, but from device
      ios = LiteNetworkIO(inputs=[["data", is_host=False]])

      network = LiteNetwork(config=net_config, io=ios)
      network.load(model_path)

      dev_input_tensor = network.get_io_tensor("data")

      # read input to input_data
      input_layout = dev_input_tensor.layout
      shape = list(input_layout.shapes)[0 : input_layout.ndim]
      arr = np.ones(shape[1:], "float32")

      shape[0] = batch
      print(shape)
      batch_tensor = TensorBatchCollector(
            shape, dtype=LiteDataType.LITE_FLOAT, device_type=LiteDeviceType.LITE_CUDA
      )
      for time in range(3):
         batch_tensor.free(range(batch))
         for i in range(batch):
            batch = batch_tensor.collect(arr)
            print("collect batch id = {}".format(batch))
            arr += 1

         # set device input data to input_tensor of the network without copy
         dev_input_tensor.share_memory_with(batch_tensor.get())
         network.forward()
         network.wait()

         output_names = network.get_all_output_name()
         output_tensor = network.get_io_tensor(output_names[0])
         output_data = output_tensor.to_numpy()
         print('shufflenet output shape={}, max={}, sum={}'.format(output_data.shape, output_data.max(), output_data.sum()))

   test_network()

上面示例主要做了以下事情：

* 通过 :ref:`lite_config` 和 :ref:`lite_io` 来创建一个运行在 CUDA 上的 Network，并配置该 Network 中输入名字为 "data" 的 Tensor 在 CUDA 上，这样用户可以直接将 CUDA device 上的内存 share 给它。
* 通过该 Network 加载 shufflenet 模型，并获取名字为 "data" 的输入 Tensor，以及它的 layout 信息。
* 通过输入 tensor 的 layout 信息和 batch 信息，将创建一个在 CUDA 上的 TensorBatchCollector，并循环攒了 4 个 batch。
* 然后将 TensorBatchCollector 中的 tensor 和 Network 的输入 tensor 通过 share_memory_with 进行内存 share。
* 执行推理，获取输出数据

.. note::

   上面通过 share_memory_with 进行内存共享，将不会产生多余的数据 copy，其中 TensorBatchCollector 的使用请参考 :ref:`lite_utils_api`。