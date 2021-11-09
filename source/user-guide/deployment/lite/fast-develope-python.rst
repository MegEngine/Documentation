.. _fast-develope-python:

================================
快速上手MegEngine Lite Python 部署模型
================================

MegEngine Lite 提供 Python 接口，用户可以直接安装预编译的 python whl 包，然后使用 python 接口进行 inference，该方法可以直接加载 trace 之后的模型并执行推理，
减少使用 C++ 进行推理时候复杂的编译环节，MegEngine Lite 的 whl 包是和 MegEngine 的 whl 包绑定在一起的，所以只需要安装 MegEngine python whl 包即可。
本文将从获取一个训练好的 shufflenet_v2 模型出发，讲解如何使用 MegEngine Lite 的 python 接口将其部署到 Linux x86 中环境下运行。

模型准备
--------

首先我们需要 :ref:`dump` ，这里主要使用 MegEngine 的 trace dump 功能将动态图 trace 为静态图，同时对静态图进行 Inference 相关的图优化。
下面将要用到的模型为 MegEngine 预训练的模型，来自 `模型中心 <https://megengine.org.cn/model-hub>`_ 。 安装 MegEngine 之后运行下面的 python 脚本将 dump 一个预训练的 shufflenet_v2.mge 模型。

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

上面代码最后 dump 模型时将模型命名为 shufflenet_v2.mge 并设置输入 Tensor 的名字为 "data"，后续将通过这个名字获取模型的输入 Tensor。

编写 Inference 代码
-------------------

首先创建一个 inference.py，在这个 python 文件中将直接调用 MegEngine Lite 的 python 接口运行 shufflenet_v2.mge 模型，输入数据是随机生成的，不用在乎计算结果。

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
        input_data = np.random.uniform(0,1, (1,3,224,224)).astype("float32")
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
 * 创建默认配置的 Network ：这个 Network 所有配置都保持默认
 * Load 模型，MegEngine 将读取模型文件，并解析模型文件并创建计算图
 * 通过名字 "data" 获取输入 Tensor，并将随机生成的 numpy 数据拷贝到获取的输入 Tensor中
 * 执行推理
 * 获取输出 Tensor 这里通过获得所有输出 Tensor 的名字，再通过获取的名字去获取对应的输出 Tensor，输出 Tensor 可以直接转换到 numpy 的 array，然后进行数据处理。

这样这个调用 MegEngine Lite 的 python 接口的 demo 就完成了。但是 python 接口目前只支持 windows/macos/linux x86 和 cuda 版本，不支持 Android Arm。
