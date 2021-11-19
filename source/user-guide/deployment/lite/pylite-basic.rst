.. _pylite-basic:

===================================
MegEngine Lite Python 基础实例
===================================


在CPU上做推理的一个例子
-----------------------

接下来，我们将逐步讲解一个使用MegEngine Lite Python在CPU上做推理的简单例子。 


0. 准备输入数据和模型文件
~~~~~~~~~~~~~~~~~~~~~~~~~

示例代码如下：

.. code-block:: python
	
	from megenginelite import *
	import numpy as np
	import os
	import time

	source_dir = os.getenv('LITE_TEST_RESOUCE')
	input_data_path = os.path.join(source_dir, 'input_data.npy')
	# read input to input_data
	assert os.path.exists(input_data_path), '{} not found.'.format(
		input_data_path)

	input_data = np.load(input_data_path)
	model_path = os.path.join(source_dir, "shufflenet.mge")

	assert os.path.exists(model_path), '{} not found.'.format(model_path)

在本例中，输入数据为 *input_data.npy* ，模型文件为 *shufflenet.mge* ，这两个文件存放在 :src:`/lite/test/resource/lite` 目录下，但需要通过通过 *git lfs pull* 来获得。

1. 加载模型
~~~~~~~~~~~~~

模型文件将被作为参数，传给 **LiteNetwork** 的 ``load`` 函数，以构建一个 **LiteNetwork** 实体。模型加载后，即可拿到模型所有输入、输出的名字，以及输入Tensor的一些信息。

.. warning::

   输出Tensor的对应信息需要在模型执行过推理之后才能拿到。

.. code-block:: python

	network = LiteNetwork()
	network.load(model_path)
	print('==> Model loaded.'.format(model_path))

	input_names = network.get_all_input_name()
	output_names = network.get_all_output_name()

	print('\tmodel_path: {}'.format(model_path))
	print('\tcpu_thread_num: {}'.format(network.threads_number))
	for input_name in input_names:
		tensor = network.get_io_tensor(input_name)
		print(
			'\tinput_tensor[{0:d}]: name={1:s} shape={2} dtype={3} is_continuous={4}'
			.format(input_names.index(input_name), input_name,
					tensor.layout.shapes[:tensor.layout.ndim],
					LiteDataType(tensor.layout.data_type).name,
					tensor.is_continue))

2. 加载输入数据
~~~~~~~~~~~~~~~~

通过 ``get_io_tensor()`` 接口，获取指定的 **LiteTensor** 实体，然后用 ``set_data_by_copy()`` 把numpy数据拷贝给该 **LiteTensor**：

.. code-block:: python

	input_tensor = network.get_io_tensor(input_names[0])

	# copy input data to input_tensor of the network
	input_tensor.set_data_by_copy(input_data)
	print('==> Input data {0:s} set to tensor \"{1:s}\".'.format(
		input_data_path, input_names[0]))


.. note:: 

	**LiteTensor** 获取数据的方式除了 ``set_data_by_copy()`` 之外，还有 ``share_memory_with()``,  ``copy_from()`` 等。详情请参考 :ref:`pylite-advanced`


3. 推理
~~~~~~~~~~~~

利用 **LiteNetwork** 的 ``forward()`` 和 ``wait()`` 接口完成网络的推理，相关代码如下：

.. code-block:: python

	print('==> Start to inference.')
	start_time = time.time()
	network.forward()
	network.wait()
	exec_time = time.time() - start_time
	print('==> Inference finished within {0:.3f}ms.'.format(exec_time * 1000))


4. 获取输出数据
~~~~~~~~~~~~~~~~~

推理完成后，网络的输出Tensor便可以通过 **LiteNetwork** 的 ``get_io_tensor()`` 函数获取。**LiteTensor** 的 ``to_numpy()`` 接口可以把 **LiteTensor** 的数据转存为numpy数据的形式，具体代码演示如下：

.. code-block:: python

	for output_name in output_names:
		tensor = network.get_io_tensor(output_name)
		print(
			'\toutput_tensor[{0:d}]: name={1:s} shape={2} dtype={3} is_continuous={4}'
			.format(output_names.index(output_name), output_name,
					tensor.layout.shapes[:tensor.layout.ndim],
					LiteDataType(tensor.layout.data_type).name,
					tensor.is_continue))

	output_tensor = network.get_io_tensor(output_names[0])
	print('==> Output tensor \"{}\" extracted.'.format(output_names[0]))

	output_data = output_tensor.to_numpy()
	print('\toutput size={} max_id={} max_val={}, sum={}'.format(
		output_data.size, np.argmax(output_data), output_data.max(),
		output_data.sum()))


