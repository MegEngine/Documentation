.. _api-example:

=====================
TracedModule 接口介绍
=====================
.. note::

   注意：TracedModule API 在未来一段时间会根据使用反馈进行调整，请关注 github release note 获取变更。欢迎在文档或 Github 提交使用反馈，一起让模型到应用更快更便捷！

以 resnet18 为例介绍 TracedModule 的使用方式，model.py 可从 :models:`official/vision/classification/resnet/model.py` 下载。
通过 :py:func:`~.trace_module` 方法将 Module 转为 TracedModule，接口形式如下：

.. code-block:: python

    def trace_module(mod: Module, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> TracedModule:
        """
        module: 要被 trace 的原 Module
        args/kwargs: 运行原 Module 所需要的输入
        """
        ...
        return traced_module

将自定义的 resnet18（Module）转换为 TracedModule：

.. code-block:: python

    import megengine.functional as F
    import megengine.module as M
    import megengine as mge
    from model import resnet18

    # resnet : Module
    resnet = resnet18()
    
    import megengine.traced_module as tm
    inp = F.zeros(shape=(1,3,224,224))

    # traced_resnet : TracedModule
    traced_resnet =  tm.trace_module(resnet, inp)

.. _node-expr-method:

TracedModule 的常用方法
=======================

graph
------------------
graph 属性是 TracedModule 最重要的属性，其返回一个 InternalGraph，描述了该 TracedMdoule 的执行过程。

**示例：**

>>> graph = traced_resnet.graph
>>> graph
ResNet.Graph (self, x) {
        %2:     conv1 = getattr(self, "conv1") -> (Conv2d)
        %3:     conv1_out = conv1(x, )
        %4:     bn1 = getattr(self, "bn1") -> (BatchNorm2d)
        %5:     bn1_out = bn1(conv1_out, )
        %6:     relu_out = nn.relu(bn1_out, )
        %7:     maxpool = getattr(self, "maxpool") -> (MaxPool2d)
        %8:     maxpool_out = maxpool(relu_out, )
        %9:     layer1 = getattr(self, "layer1") -> (Module)
        %10:    layer1_out = layer1(maxpool_out, )
        %47:    layer2 = getattr(self, "layer2") -> (Module)
        %48:    layer2_out = layer2(layer1_out, )
        %91:    layer3 = getattr(self, "layer3") -> (Module)
        %92:    layer3_out = layer3(layer2_out, )
        %135:   layer4 = getattr(self, "layer4") -> (Module)
        %136:   layer4_out = layer4(layer3_out, )
        %179:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, None, 0, average_count_exclude_padding, )
        %180:   flatten_out = tensor.flatten(avg_pool2d_out, 1, -1, )
        %181:   fc = getattr(self, "fc") -> (Linear)
        %182:   fc_out = fc(flatten_out, )
        return fc_out
}

如 ``traced_resnet.graph`` 所示，``ResNet.forward`` 中的 ``x = self.conv1(x)`` 将会被解析为以下两个操作:

1. 获取 ``conv1 = self.conv1``, 对应的 Expr 为 ``%2: conv1 = getattr(self, "conv1") -> (Conv2d)``
2. 执行 ``conv1_out = conv1(x)``, 对应的 Expr 为 ``%3: conv1_out = conv1(x, )``

其中 ``%`` 后的数字为 Expr 的 id。

resnet18 中使用的所有的自定义子 Module 都将会被转换为 TracedModule，
例如 layer1 被转换 TracedModule 后有相应的名为 "ResNet_layer1" 的 Graph 记录其 forward 执行过程。

>>> traced_resnet.layer1.graph
ResNet_layer1.Graph (self, inp) {
        %13:    _0 = getattr(self, "0") -> (Module)
        %14:    _1 = getattr(self, "1") -> (Module)
        %15:    _0_out = _0(inp, )
        %31:    _1_out = _1(_0_out, )
        return _1_out
}

可以通过 ``"{:i}".format(graph)`` 方式查看 Node 的 id。 
例如 ``%2_conv1`` 中的 2 表示 ``conv_1`` 这个 Node 的 id 为 ``2``。

.. dropdown:: "{:i}".format(graph)

    >>> print("{:i}".format(graph))
    ResNet.Graph (%0_self, %1_x) {
            %2:     %2_conv1 = getattr(%0_self, "conv1") -> (Conv2d)
            %3:     %3_conv1_out = %2_conv1(%1_x, )
            %4:     %4_bn1 = getattr(%0_self, "bn1") -> (BatchNorm2d)
            %5:     %5_bn1_out = %4_bn1(%3_conv1_out, )
            %6:     %6_relu_out = nn.relu(%5_bn1_out, )
            %7:     %7_maxpool = getattr(%0_self, "maxpool") -> (MaxPool2d)
            %8:     %8_maxpool_out = %7_maxpool(%6_relu_out, )
            %9:     %9_layer1 = getattr(%0_self, "layer1") -> (Module)
            %10:    %10_layer1_out = %9_layer1(%8_maxpool_out, )
            %47:    %47_layer2 = getattr(%0_self, "layer2") -> (Module)
            %48:    %48_layer2_out = %47_layer2(%10_layer1_out, )
            %91:    %91_layer3 = getattr(%0_self, "layer3") -> (Module)
            %92:    %92_layer3_out = %91_layer3(%48_layer2_out, )
            %135:   %135_layer4 = getattr(%0_self, "layer4") -> (Module)
            %136:   %136_layer4_out = %135_layer4(%92_layer3_out, )
            %179:   %179_avg_pool2d_out = nn.avg_pool2d(%136_layer4_out, 7, None, 0, average_count_exclude_padding, )
            %180:   %180_flatten_out = tensor.flatten(%179_avg_pool2d_out, 1, -1, )
            %181:   %181_fc = getattr(%0_self, "fc") -> (Linear)
            %182:   %182_fc_out = %181_fc(%180_flatten_out, )
            return %182_fc_out
    }

flatten
-------
该方法可去除 InternalGraph 的中的层次结构（将子 graph 展开，去除自定义子 Module 的 graph）, 并返回一个新的 TracedModule。

:py:meth:`.TracedModule.flatten` ``()``
    返回一个新的 TracedModule，其所对应的 Graph 无层次结构

拍平后的 InternalGraph 仅包含内置 Module 或 function 的 Expr，此时可以直观的得到数据之间的连接关系。

**示例：**

.. dropdown:: flatten

    >>> flattened_resnet = traced_resnet.flatten()
    >>> flattened_resnet.graph
    ResNet.Graph (self, x) {
            %2:     conv1 = getattr(self, "conv1") -> (Conv2d)
            %3:     conv1_out = conv1(x, )
            %4:     bn1 = getattr(self, "bn1") -> (BatchNorm2d)
            %5:     bn1_out = bn1(conv1_out, )
            %6:     relu_out = nn.relu(bn1_out, )
            %7:     maxpool = getattr(self, "maxpool") -> (MaxPool2d)
            %8:     maxpool_out = maxpool(relu_out, )
            %9:     layer1__0_conv1 = getattr(self, "layer1.0.conv1") -> (Conv2d)
            %10:    layer1__0_conv1_out = layer1__0_conv1(maxpool_out, )
            %11:    layer1__0_bn1 = getattr(self, "layer1.0.bn1") -> (BatchNorm2d)
            %12:    layer1__0_bn1_out = layer1__0_bn1(layer1__0_conv1_out, )
            %13:    layer1__0_relu_out = nn.relu(layer1__0_bn1_out, )
            %14:    layer1__0_conv2 = getattr(self, "layer1.0.conv2") -> (Conv2d)
            %15:    layer1__0_conv2_out = layer1__0_conv2(layer1__0_relu_out, )
            %16:    layer1__0_bn2 = getattr(self, "layer1.0.bn2") -> (BatchNorm2d)
            %17:    layer1__0_bn2_out = layer1__0_bn2(layer1__0_conv2_out, )
            %18:    layer1__0_downsample = getattr(self, "layer1.0.downsample") -> (Identity)
            %19:    layer1__0_downsample_out = layer1__0_downsample(maxpool_out, )
            %20:    layer1__0_iadd_out = layer1__0_bn2_out.__iadd__(layer1__0_downsample_out, )
            %21:    layer1__0_out = nn.relu(layer1__0_iadd_out, )
            %22:    layer1__1_conv1 = getattr(self, "layer1.1.conv1") -> (Conv2d)
            %23:    layer1__1_conv1_out = layer1__1_conv1(layer1__0_out, )
            %24:    layer1__1_bn1 = getattr(self, "layer1.1.bn1") -> (BatchNorm2d)
            %25:    layer1__1_bn1_out = layer1__1_bn1(layer1__1_conv1_out, )
            %26:    layer1__1_relu_out = nn.relu(layer1__1_bn1_out, )
            %27:    layer1__1_conv2 = getattr(self, "layer1.1.conv2") -> (Conv2d)
            %28:    layer1__1_conv2_out = layer1__1_conv2(layer1__1_relu_out, )
            %29:    layer1__1_bn2 = getattr(self, "layer1.1.bn2") -> (BatchNorm2d)
            %30:    layer1__1_bn2_out = layer1__1_bn2(layer1__1_conv2_out, )
            %31:    layer1__1_downsample = getattr(self, "layer1.1.downsample") -> (Identity)
            %32:    layer1__1_downsample_out = layer1__1_downsample(layer1__0_out, )
            %33:    layer1__1_iadd_out = layer1__1_bn2_out.__iadd__(layer1__1_downsample_out, )
            %34:    layer1_out = nn.relu(layer1__1_iadd_out, )
            %35:    layer2__0_conv1 = getattr(self, "layer2.0.conv1") -> (Conv2d)
            %36:    layer2__0_conv1_out = layer2__0_conv1(layer1_out, )
            %37:    layer2__0_bn1 = getattr(self, "layer2.0.bn1") -> (BatchNorm2d)
            %38:    layer2__0_bn1_out = layer2__0_bn1(layer2__0_conv1_out, )
            %39:    layer2__0_relu_out = nn.relu(layer2__0_bn1_out, )
            %40:    layer2__0_conv2 = getattr(self, "layer2.0.conv2") -> (Conv2d)
            %41:    layer2__0_conv2_out = layer2__0_conv2(layer2__0_relu_out, )
            %42:    layer2__0_bn2 = getattr(self, "layer2.0.bn2") -> (BatchNorm2d)
            %43:    layer2__0_bn2_out = layer2__0_bn2(layer2__0_conv2_out, )
            %44:    layer2__0_downsample__0 = getattr(self, "layer2.0.downsample.0") -> (Conv2d)
            %45:    layer2__0_downsample__1 = getattr(self, "layer2.0.downsample.1") -> (BatchNorm2d)
            %46:    layer2__0_downsample__0_out = layer2__0_downsample__0(layer1_out, )
            %47:    layer2__0_downsample_out = layer2__0_downsample__1(layer2__0_downsample__0_out, )
            %48:    layer2__0_iadd_out = layer2__0_bn2_out.__iadd__(layer2__0_downsample_out, )
            %49:    layer2__0_out = nn.relu(layer2__0_iadd_out, )
            %50:    layer2__1_conv1 = getattr(self, "layer2.1.conv1") -> (Conv2d)
            %51:    layer2__1_conv1_out = layer2__1_conv1(layer2__0_out, )
            %52:    layer2__1_bn1 = getattr(self, "layer2.1.bn1") -> (BatchNorm2d)
            %53:    layer2__1_bn1_out = layer2__1_bn1(layer2__1_conv1_out, )
            %54:    layer2__1_relu_out = nn.relu(layer2__1_bn1_out, )
            %55:    layer2__1_conv2 = getattr(self, "layer2.1.conv2") -> (Conv2d)
            %56:    layer2__1_conv2_out = layer2__1_conv2(layer2__1_relu_out, )
            %57:    layer2__1_bn2 = getattr(self, "layer2.1.bn2") -> (BatchNorm2d)
            %58:    layer2__1_bn2_out = layer2__1_bn2(layer2__1_conv2_out, )
            %59:    layer2__1_downsample = getattr(self, "layer2.1.downsample") -> (Identity)
            %60:    layer2__1_downsample_out = layer2__1_downsample(layer2__0_out, )
            %61:    layer2__1_iadd_out = layer2__1_bn2_out.__iadd__(layer2__1_downsample_out, )
            %62:    layer2_out = nn.relu(layer2__1_iadd_out, )
            %63:    layer3__0_conv1 = getattr(self, "layer3.0.conv1") -> (Conv2d)
            %64:    layer3__0_conv1_out = layer3__0_conv1(layer2_out, )
            %65:    layer3__0_bn1 = getattr(self, "layer3.0.bn1") -> (BatchNorm2d)
            %66:    layer3__0_bn1_out = layer3__0_bn1(layer3__0_conv1_out, )
            %67:    layer3__0_relu_out = nn.relu(layer3__0_bn1_out, )
            %68:    layer3__0_conv2 = getattr(self, "layer3.0.conv2") -> (Conv2d)
            %69:    layer3__0_conv2_out = layer3__0_conv2(layer3__0_relu_out, )
            %70:    layer3__0_bn2 = getattr(self, "layer3.0.bn2") -> (BatchNorm2d)
            %71:    layer3__0_bn2_out = layer3__0_bn2(layer3__0_conv2_out, )
            %72:    layer3__0_downsample__0 = getattr(self, "layer3.0.downsample.0") -> (Conv2d)
            %73:    layer3__0_downsample__1 = getattr(self, "layer3.0.downsample.1") -> (BatchNorm2d)
            %74:    layer3__0_downsample__0_out = layer3__0_downsample__0(layer2_out, )
            %75:    layer3__0_downsample_out = layer3__0_downsample__1(layer3__0_downsample__0_out, )
            %76:    layer3__0_iadd_out = layer3__0_bn2_out.__iadd__(layer3__0_downsample_out, )
            %77:    layer3__0_out = nn.relu(layer3__0_iadd_out, )
            %78:    layer3__1_conv1 = getattr(self, "layer3.1.conv1") -> (Conv2d)
            %79:    layer3__1_conv1_out = layer3__1_conv1(layer3__0_out, )
            %80:    layer3__1_bn1 = getattr(self, "layer3.1.bn1") -> (BatchNorm2d)
            %81:    layer3__1_bn1_out = layer3__1_bn1(layer3__1_conv1_out, )
            %82:    layer3__1_relu_out = nn.relu(layer3__1_bn1_out, )
            %83:    layer3__1_conv2 = getattr(self, "layer3.1.conv2") -> (Conv2d)
            %84:    layer3__1_conv2_out = layer3__1_conv2(layer3__1_relu_out, )
            %85:    layer3__1_bn2 = getattr(self, "layer3.1.bn2") -> (BatchNorm2d)
            %86:    layer3__1_bn2_out = layer3__1_bn2(layer3__1_conv2_out, )
            %87:    layer3__1_downsample = getattr(self, "layer3.1.downsample") -> (Identity)
            %88:    layer3__1_downsample_out = layer3__1_downsample(layer3__0_out, )
            %89:    layer3__1_iadd_out = layer3__1_bn2_out.__iadd__(layer3__1_downsample_out, )
            %90:    layer3_out = nn.relu(layer3__1_iadd_out, )
            %91:    layer4__0_conv1 = getattr(self, "layer4.0.conv1") -> (Conv2d)
            %92:    layer4__0_conv1_out = layer4__0_conv1(layer3_out, )
            %93:    layer4__0_bn1 = getattr(self, "layer4.0.bn1") -> (BatchNorm2d)
            %94:    layer4__0_bn1_out = layer4__0_bn1(layer4__0_conv1_out, )
            %95:    layer4__0_relu_out = nn.relu(layer4__0_bn1_out, )
            %96:    layer4__0_conv2 = getattr(self, "layer4.0.conv2") -> (Conv2d)
            %97:    layer4__0_conv2_out = layer4__0_conv2(layer4__0_relu_out, )
            %98:    layer4__0_bn2 = getattr(self, "layer4.0.bn2") -> (BatchNorm2d)
            %99:    layer4__0_bn2_out = layer4__0_bn2(layer4__0_conv2_out, )
            %100:   layer4__0_downsample__0 = getattr(self, "layer4.0.downsample.0") -> (Conv2d)
            %101:   layer4__0_downsample__1 = getattr(self, "layer4.0.downsample.1") -> (BatchNorm2d)
            %102:   layer4__0_downsample__0_out = layer4__0_downsample__0(layer3_out, )
            %103:   layer4__0_downsample_out = layer4__0_downsample__1(layer4__0_downsample__0_out, )
            %104:   layer4__0_iadd_out = layer4__0_bn2_out.__iadd__(layer4__0_downsample_out, )
            %105:   layer4__0_out = nn.relu(layer4__0_iadd_out, )
            %106:   layer4__1_conv1 = getattr(self, "layer4.1.conv1") -> (Conv2d)
            %107:   layer4__1_conv1_out = layer4__1_conv1(layer4__0_out, )
            %108:   layer4__1_bn1 = getattr(self, "layer4.1.bn1") -> (BatchNorm2d)
            %109:   layer4__1_bn1_out = layer4__1_bn1(layer4__1_conv1_out, )
            %110:   layer4__1_relu_out = nn.relu(layer4__1_bn1_out, )
            %111:   layer4__1_conv2 = getattr(self, "layer4.1.conv2") -> (Conv2d)
            %112:   layer4__1_conv2_out = layer4__1_conv2(layer4__1_relu_out, )
            %113:   layer4__1_bn2 = getattr(self, "layer4.1.bn2") -> (BatchNorm2d)
            %114:   layer4__1_bn2_out = layer4__1_bn2(layer4__1_conv2_out, )
            %115:   layer4__1_downsample = getattr(self, "layer4.1.downsample") -> (Identity)
            %116:   layer4__1_downsample_out = layer4__1_downsample(layer4__0_out, )
            %117:   layer4__1_iadd_out = layer4__1_bn2_out.__iadd__(layer4__1_downsample_out, )
            %118:   layer4_out = nn.relu(layer4__1_iadd_out, )
            %119:   avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, None, 0, average_count_exclude_padding, )
            %120:   flatten_out = tensor.flatten(avg_pool2d_out, 1, -1, )
            %121:   fc = getattr(self, "fc") -> (Linear)
            %122:   fc_out = fc(flatten_out, )
            return fc_out
    }

set_watch_points & clear_watch_points
-------------------------------------
查看 TracedModule 执行时 graph 中某个 Node 对应的真正的 Tensor/Module。

:py:meth:`.TracedModule.set_watch_points` ``(nodes : Sequence[Node])``
    设置需要观察的 Node

    * ``nodes`` 待观察的 Node

:py:meth:`.TracedModule.clear_watch_points` ``()``
    清除需要观察的 Node

**示例：**

通过该方法观察 ``F.avg_pool2d`` 的输入与输出 Tensor 的 shape 变换

>>> avgpool_inp_node, avgpool_out_node = traced_resnet.graph.get_node_by_id([136,179])
>>> traced_resnet.set_watch_points([avgpool_inp_node, avgpool_out_node])
>>> inp = F.zeros(shape = (1,3,224,224))
>>> traced_resnet(inp)
>>> watched_value = traced_resnet.watch_node_value
>>> watched_value[avgpool_inp_node].shape
(1, 512, 7, 7)
>>> watched_value[avgpool_out_node].shape
(1, 512, 1, 1)

``traced_resnet.watch_node_value`` 是一个 ``Dict[Node, Union[Tensor, Module]]``，
它的 ``key`` 是已被设置要观察的 Node，``value`` 是网络运行期间 ``key`` 所对应的真正的 Tensor 或 Module。

可以看到上面的例子成功获取到了 ``F.avg_pool2d`` 的输入与输出的 shape。
当再次运行 ``traced_resnet`` 时，之前观察到的 Tensor 或 Module 将被新的值覆盖。

set_end_points & clear_end_points
---------------------------------
设置模型停止运行的位置，接受一个 ``List[Node]`` 作为输入，当网络生成所有设置的 ``Node`` 后会立即返回，不再继续往下执行。
*该方法仅支持将最顶层 graph 中的 node 设置未结束运行点。*

:py:meth:`.TracedModule.set_end_points` ``(nodes : Sequence[Node])``
    设置结束运行点

    * ``nodes`` 停止运行处的的 ``Node``

:py:meth:`.TracedModule.clear_end_points` ``()``
    清除结束运行点

**示例：**

将 ``traced_resnet`` 的输出点设置为 ``F.avg_pool2d`` 的输入与输出，当 ``F.avg_pool2d`` 执行完后，
就立即结束运行之后的 Expr，并将 ``F.avg_pool2d`` 的输入与输出作为模型返回值直接返回

>>> avgpool_inp_node, avgpool_out_node = traced_resnet.graph.get_node_by_id([136,179])
>>> traced_resnet.set_end_points([avgpool_inp_node, avgpool_out_node])
>>> inp = F.zeros(shape = (1,3,224,224))
>>> avgpool_inp, avgpool_out =  traced_resnet(inp)
>>> avgpool_inp.shape
(1, 512, 7, 7)
>>> avgpool_inp.shape
(1, 512, 1, 1)

可以看到模型的输出变成了 ``F.avg_pool2d`` 的输入与输出，并且未执行 ``F.avg_pool2d`` 之后的 Expr。

Node 、Expr 、InternalGraph 的常用方法
============================================

InternalGraph.exprs
-------------------
遍历 Graph 中的 Expr。通过访问 :py:meth:`.InternalGraph.exprs` 可按模型执行顺序得到该 Graph 中所记录 Expr 序列。

:py:meth:`.InternalGraph.exprs` ``(recursive : bool = True)``
    按 Expr 执行顺序获取 Expr 执行序列
    
    * ``recursive``:  是否获取子 Graph 中的 Expr，默认为 True

**示例：**

.. dropdown:: InternalGraph.exprs

    >>> traced_resnet.graph.exprs(recursive=False)
    <megengine.traced_module.traced_module.ExprFilter at 0x7f4aa317a470>

    >>> traced_resnet.graph.exprs(recursive=False).as_list()
    [%0:    self = Input(),
    %1:    x = Input(),
    %2:    conv1 = getattr(self, "conv1") -> (Conv2d),
    %3:    conv1_out = conv1(x, ),
    %4:    bn1 = getattr(self, "bn1") -> (BatchNorm2d),
    %5:    bn1_out = bn1(conv1_out, ),
    %6:    relu_out = nn.relu(bn1_out, ),
    %7:    maxpool = getattr(self, "maxpool") -> (MaxPool2d),
    %8:    maxpool_out = maxpool(relu_out, ),
    %9:    layer1 = getattr(self, "layer1") -> (Module),
    %10:   layer1_out = layer1(maxpool_out, ),
    %47:   layer2 = getattr(self, "layer2") -> (Module),
    %48:   layer2_out = layer2(layer1_out, ),
    %91:   layer3 = getattr(self, "layer3") -> (Module),
    %92:   layer3_out = layer3(layer2_out, ),
    %135:  layer4 = getattr(self, "layer4") -> (Module),
    %136:  layer4_out = layer4(layer3_out, ),
    %179:  avg_pool2d_out = nn.avg_pool2d(layer4_out, 7, None, 0, average_count_exclude_padding, ),
    %180:  flatten_out = tensor.flatten(avg_pool2d_out, 1, -1, ),
    %181:  fc = getattr(self, "fc") -> (Linear),
    %182:  fc_out = fc(flatten_out, )]

InternalGraph.nodes
-------------------
遍历 Graph 中的 Node。通过访问 :py:meth:`.InternalGraph.nodes` 可得到该 graph 中的 Node 序列。

:py:meth:`.InternalGraph.nodes` ``(recursive : bool = True)``
    按 id 从小到大返回 Graph 中的 Node
    
    * ``recursive``:  是否获取子 Graph 中的 Node，默认为 True

**示例：**

.. dropdown:: InternalGraph.nodes

    >>> nodes = traced_resnet.graph.nodes(recursive=False).as_list() 
    >>> for node in nodes: 
    ...     print("{:i}".format(node))
    %0_self
    %1_x
    %2_conv1
    %3_conv1_out
    %4_bn1
    %5_bn1_out
    %6_relu_out
    %7_maxpool
    %8_maxpool_out
    %9_layer1
    %10_layer1_out
    %47_layer2
    %48_layer2_out
    %91_layer3
    %92_layer3_out
    %135_layer4
    %136_layer4_out
    %179_avg_pool2d_out
    %180_flatten_out
    %181_fc
    %182_fc_out

Expr.inputs & Expr.outputs
--------------------------
通过访问 Expr 的 inputs 和 outputs 属性，可获得该 Expr 的输入和输出 Node。

:py:attr:`.Expr.inputs` ``: List[Node]``

:py:attr:`.Expr.outputs` ``: List[Node]``

**示例：**

>>> exprs = traced_resnet.graph.exprs(recursive=False).as_list()
>>> fc_expr = exprs[-1]
>>> fc_expr
%182:  fc_out = fc(flatten_out, )
>>> fc_expr.inputs
[fc, flatten_out]
>>> fc_expr.outputs
[fc_out]

Expr.args & Expr.kwargs & Expr.named_args
-----------------------------------------
在调用一个 function 时，例如 F.conv2，其输入并不是只有 Tensor，
还有一些非 Tensor 的输入，例如 kernel_size 等，我们提供了
``Expr.args``、``Expr.kwargs`` 和 ``Expr.named_args``
三种方法获取该生成该 Expr 时所传入的非 Tensor 输入。

以一个自定义的 ``MyBn`` 为例介绍在 ``trace`` 时对参数的处理，以及上述 3 个方法的使用方式。

.. code-block:: python

    import megengine.module as M
    import megengine.functional as F
    import megengine as mge
    import megengine.traced_module as tm

    class MyBn(M.Module):
        def __init__(self, ):
            super().__init__()
            self.weight = mge.Parameter(F.ones([3]))
            self.bias = mge.Parameter(F.zeros([3]))
        def forward(self, x):
            x = F.batch_norm(x, weight=self.weight, bias=self.bias, training=True)
            return x
    
    mybn = MyBn()
    inp = F.zeros(shape = [1, 3, 224, 224])

将 ``my_bn`` 转换为 TracedMdoule 后我们可以得到如下一个 graph:

>>> traced_mybn = tm.trace_module(mybn, inp)
>>> traced_mybn.graph
MyBn.Graph (self, x) {
        %2:     weight = getattr(self, "weight") -> (Tensor)
        %3:     bias = getattr(self, "bias") -> (Tensor)
        %4:     batch_norm_out = nn.batch_norm(x, None, None, weight, bias, compute_mode=default, eps=1e-05, inplace=True, momentum=0.9, param_dim=dim_1c11, training=True)
        return batch_norm_out
}

``F.batch_norm`` 的函数定义如下：

.. code-block:: python

    def batch_norm(
        inp: Tensor,
        running_mean: Tensor = None,
        running_var: Tensor = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        *,
        training: bool = False,
        momentum: float = 0.9,
        eps: float = 1e-5,
        inplace: bool = True,
        compute_mode="default",
        param_dim="dim_1c11"
    ):...

可以从 graph 中看到，在 trace 时，我们将 ``*`` 号前的参数全部转为位置参数(positional argument)，
将 ``*`` 后的参数全部转换为了关键字参数(keyword argument)，在调用函数时即使没有输入相应的参数我们也会将其默认值记录下来，
例如 ``eps=1e-5``。

**示例1：**

``Expr.args`` 返回的是 function 位置参数所对应的值。

>>> bn_expr = graph.exprs().as_list()[-1]
>>> bn_expr.args
(x, None, None, weight, bias)

可以看到当调用 ``args`` 属性时，返回了 ``*`` 号前的 5 个位置参数，分别是
``(inp, running_mean, running_var, weight, bias)``。

**示例2：**

``Expr.kwargs`` 返回的是 function 关键字参数的名字以及其所对应的值。

>>> bn_expr = graph.exprs().as_list()[-1]
>>> bn_expr.kwargs
{'compute_mode': 'default',
'eps': 1e-05,
'inplace': True,
'momentum': 0.9,
'param_dim': 'dim_1c11',
'training': True}

可以看到当调用 ``kwargs`` 属性时，返回了 ``*`` 号后的所有关键字参数，包括参数名字和实际输入的参数（或默认值）。

**示例3：**

``Expr.named_args`` 返回的是 function 的参数名字以及其所对应的输入值

该属性提供了所有参数的名字以及调用时输入的参数，可以通过该方法获取参数名字所对应的输入值。

>>> bn_expr = graph.exprs().as_list()[-1]
>>> bn_expr.named_args
{'inp': x,
'running_mean': None,
'running_var': None,
'weight': weight,
'bias': bias,
'compute_mode': 'default',
'eps': 1e-05,
'inplace': True,
'momentum': 0.9,
'param_dim': 'dim_1c11',
'training': True}

Node.expr
---------
通过访问 Node 的 expr 属性，可获得该 Node 是由哪个 Expr 生成的。

:py:attr:`.Node.expr` ``: Expr``

**示例：**

>>> nodes = traced_resnet.graph.nodes(recursive=False).as_list()
>>> fc_out_node = nodes[-1]
>>> fc_out_node.expr
%182:  fc_out = fc(flatten_out, )

Node.users
----------
通过访问 Node 的 users 属性，可获得该 Node 是将会被哪些 Expr 作为输入所使用。

:py:attr:`.Node.users` ``: Lsit[Expr]``

**示例：**

>>> nodes = traced_resnet.graph.nodes(recursive=False).as_list()
>>> fc_mnode = nodes[-2]
>>> fc_mnode.users
[%182: fc_out = fc(flatten_out, )]

ModuleNode.owner
----------------
通过访问 ModuleNode 的 owner 属性，可直接访问该 ModuleNode 所对应的 Module。

:py:attr:`.ModuleNode.owner` ``: Module``

**示例：**

>>> nodes = traced_resnet.graph.nodes(recursive=False).as_list()
>>> fc_mnode = nodes[-2]
>>> fc_mnode.owner
Linear(in_features=512, out_features=1000, bias=True)

Node.top_graph & Expr.top_graph
-------------------------------
通过访问 Node 或 Expr 的 top_graph 属性，可直获得该 Node 或 Expr 所属的 InternalGraph。

:py:attr:`.Node.top_graph` ``: InternalGraph``

:py:attr:`.Expr.top_graph` ``: InternalGraph``

**示例：**

>>> layer1_graph = traced_resnet.layer1.graph
>>> layer1_exprs = layer1_graph.exprs(False).as_list()
>>> layer1_exprs[-1].top_graph is layer1_graph
True
>>> layer1_nodes = layer1_graph.nodes(False).as_list()
>>> layer1_nodes[-1].top_graph is layer1_graph
True

InternalGraph.eval
------------------
通过访问 InternalGraph 的 eval 方法，可以直接运行该 Graph。

:py:meth:`.InternalGraph.eval` ``(*inputs)``
    将 Tensor 直接输入 Graph 并返回按 Expr 执行序列执行后的结果
    
    * ``inputs`` 模型的输入

利用 ``eval`` 执行一个 graph 时，只需要输入与 `graph.inputs[1:]` 中的 Node 相对应的实际的 Tensor 或 Module 即可执行。

**示例：**

>>> resnet_graph = traced_resnet.graph
>>> inp = mge.Tensor(np.random.random((1, 3, 224, 224)), dtype="float32")
>>> fc_out = resnet_graph.eval(inp)[0]
>>> fc_out.shape
(1, 1000)

.. _tracedmodule-find-expr-and-node:

Node 和 Expr 的查找方法
=======================

BaseFilter
----------
:py:class:`~.BaseFilter` 是一个可迭代的类，其提供了一些方法将迭代器转换为 ``list``, ``dict`` 等。

:py:class:`~.NodeFilter` 和 :py:class:`~.ExprFilter` 继承于 :py:class:`~.BaseFilter`，
NodeFilter 负责处理 Node，ExprFilter 负责处理 Expr。

* :py:meth:`.BaseFilter.as_list`  返回 Node 或 Expr 列表
* :py:meth:`.BaseFilter.as_dict`  返回 Node 或 Expr 的 id 和 Node 或 Expr 组成的字典
* :py:meth:`.BaseFilter.as_unique`  如果查找到的 Node 或 Expr 只有一个，直接返回该 Node 或 Expr, 否则报错
* :py:meth:`.BaseFilter.as_count`  返回查找到 Node 或 Expr 的数量

get_node_by_id
--------------
通过 id 从 Graph 中获取对应 id 的 Node。

:py:meth:`.InternalGraph.get_node_by_id` ``(node_id: List[int] = None, recursive=True)``
    获取 InternalGraph 中 id 为 ``node_id`` 的 Node，支持一次查找多个 Node

    * ``node_id`` 待查找 Node 的 id 
    * ``recursive`` 是否查找子 Graph 中的 Node，默认为 True

**示例：**

>>> graph = traced_resnet.graph
>>> nodes = graph.get_node_by_id([4, 8, 31]).as_list()
>>> print(nodes)
[bn1, maxpool_out, _1_out]
>>> print(["{:i}".format(n) for n in nodes])
['%4_bn1', '%8_maxpool_out', '%31__1_out']

get_expr_by_id
--------------
通过 id 从 Graph 中获取对应 id 的 Expr

:py:meth:`.InternalGraph.get_expr_by_id` ``(expr_id: List[int] = None, recursive=True)``
    获取 InternalGraph 中 id 为 expr_id 的 Expr，支持一次查找多个 Expr

    * ``expr_id`` 待查找 Expr 的 id 列表
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

**示例：**

>>> graph = traced_resnet.graph
>>> exprs = graph.get_expr_by_id([4, 8, 31]).as_list()
>>> print(exprs)
[%4:  bn1 = getattr(self, "bn1") -> (BatchNorm2d),
 %8:  maxpool_out = maxpool(relu_out, ),
 %31: _1_out = _1(_0_out, )]

get_function_by_type
--------------------
通过该方法查找 Graph 中调用了某个内置 function 的 CallFunction Expr

:py:meth:`.InternalGraph.get_function_by_type` ``(func: Callable = None, recursive=True)``
    获取 InternalGraph 中 ``self.func == func`` 的 CallFunction Expr

    * ``func`` 可调用的函数
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

**示例：**

>>> graph = traced_resnet.graph
>>> graph.get_function_by_type(F.relu, False).as_list()
[%6:   relu_out = nn.relu(bn1_out, )]

get_method_by_type
------------------
通过该方法查找 Graph 中调用了某个 method 的 CallMethod Expr

:py:meth:`.InternalGraph.get_method_by_type` ``(method: str = None, recursive=True)``
    获取 InternalGraph 中 ``self.method == method`` 的 CallMethod

    * ``method`` 待查找某对象的方法的名字（该方法是一个可调用的函数）
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

**示例：**

>>> graph = traced_resnet.graph
>>> graph.get_method_by_type("__call__", False).as_list()
[%3:    conv1_out = conv1(x, ),
 %5:    bn1_out = bn1(conv1_out, ),
 %8:    maxpool_out = maxpool(relu_out, ),
 %10:   layer1_out = layer1(maxpool_out, ),
 %48:   layer2_out = layer2(layer1_out, ),
 %92:   layer3_out = layer3(layer2_out, ),
 %136:  layer4_out = layer4(layer3_out, ),
 %182:  fc_out = fc(flatten_out, )]

get_module_by_type
------------------
通过该方法查找 Graph 中对应某种 Module 的 ModuleNode

:py:meth:`.InternalGraph.get_module_by_type` ``(module_cls: Module, recursive=True)``
    获取 InternalGraph 中对应于 ``module_cls`` 的 ModuleNode

    * ``module_cls`` Module 某个子类
    * ``recursive`` 是否查找子 Graph 中的 Expr，默认为 True

**示例：**

>>> graph = traced_resnet.graph
>>> graph.get_module_by_type(M.BatchNorm2d, False).as_list()
[bn1]

.. _tracedmodule-graph-transform-method:

图手术常用方法
==============

add_input_node
--------------
为最顶层的 InternalGraph 增加一个输入，此输入会作为一个 free_varargs 参数（即无形参名称）。
子 Graph 不支持调用该方法。

:py:meth:`.InternalGraph.add_input_node` ``(shape, dtype="float32", name="args")``
    为顶层 Graph 新增一个输入

    * ``shape`` 新增输入的 shape
    * ``dtype`` 新增输入的 dtype，默认为 "float32"
    * ``name``  新增输入的名字，默认为 "args"，若该名字在 Graph 中已存在，则会在 name 后添加后缀，以保证 name 在 Graph 在的唯一性。

**示例：**

>>> graph = traced_resnet.graph # graph : InternalGraph
>>> new_inp_node = graph.add_input_node(shape=(1,3,224,224), dtype="float32", name="new_data")
>>> traced_resnet.argspec.args.append("new_data")
>>> print(new_inp_node)
new_data
>>> print(graph)
ResNet.Graph (self, x, new_data) {
        %2:     conv1 = getattr(self, "conv1") -> (Conv2d)
        %3:     conv1_out = conv1(x, )
        %4:     bn1 = getattr(self, "bn1") -> (BatchNorm2d)
        %5:     bn1_out = bn1(conv1_out, )
        ...
}

add_output_node
---------------
为最顶层的 InternalGraph 增加一个输出，此输入会作为输出元组中的最后一个元素。
子 Graph 不支持调用该方法。

:py:meth:`.InternalGraph.add_output_node` ``(node: TensorNode)``
    将 Graph 中的某个 Node 作为 Graph 的一个输出

    * ``node`` Graph 中的某 Node

**示例：**

>>> graph = traced_resnet.graph
>>> fc_inp_node = graph.get_node_by_id(180).as_unique()
>>> graph.add_output_node(fc_inp_node)
>>> print(graph)
ResNet.Graph (self, x) {
        %2:     conv1 = getattr(self, "conv1") -> (Conv2d)
        ...
        return fc_out, fc_out
}
>>> fc_out, fc_inp = traced_resnet(inp)
>>> fc_inp.shape
(1, 512)
>>> fc_out.shape
(1, 1000)

reset_outputs
-------------
重新设置最顶层 InternalGraph 的输出。子 Graph 不支持调用该方法。

当要改变的输出较多时，一个一个调用 ``add_output_node`` 较为麻烦，通过 ``reset_outputs`` 方法一次性重置输出内容于结构。

:py:meth:`.InternalGraph.reset_outputs` ``(node: outputs)``
    重置 Graph 的输出

    * ``node`` 由 Graph 中的 TensorNode 构成的某种结构，支持 ``list``, ``dict``, ``tuple`` 等（最底层的元素必须是 TensorNode）。 

**示例：**

>>> graph = traced_resnet.graph
>>> avgpool_inp_node = graph.get_node_by_id(136).as_unique()
>>> fc_inp_node = graph.get_node_by_id(180).as_unique()
>>> fc_out_node = graph.outputs[0]

把 fc 的输入和输出以 Dict 形式输出 并与 avgppol 的输入组成 tuple

>>> new_outputs = ({"fc_inp": fc_inp_node, "fc_out": fc_out_node }, avgpool_inp_node)

将 new_outputs 作为 graph 新的输出

>>> graph.reset_outputs(new_outputs)
>>> print(graph)
ResNet.Graph (self, x) {
        ...
        return flatten_out, fc_out, layer4_out
}
>>> fc_inp_out, avgpool_inp = traced_resnet(inp)
>>> fc_inp_out["fc_inp"].shape
(1, 512)
>>> fc_inp_out["fc_out"].shape
(1, 1000)
>>> avgpool_inp.shape
(1, 512, 7, 7)

compile
-------
该方法会将 InternalGraph 与输出无关的 Expr 删除。

:py:meth:`.InternalGraph.compile` ``()``

常与 ``insert_exprs`` 和 ``replace_node`` 一起使用。

replace_node
------------
替换 InternalGraph 中的指定 Node。可用于新增 Expr 后替换一些 Node，或结合 :py:meth:`.InternalGraph.compile` 删某些 Expr。

:py:meth:`.InternalGraph.replace_node` ``(repl_dict : Dict[Node, Node])``
    替换 Graph 中的 ``key`` 替换为 ``value``

    * ``repl_dict`` 为一个 ``key`` 和 ``value`` 都为 Node 的字典，且 ``key`` 和 ``value`` 必须在同一个 Graph 中。
      在 ``value.expr`` 之后的所有将 ``key`` 作为输入的 Expr 将被替换为以 ``value`` 作为输入。 

**示例：**

以将 traced_net.layer1 中所有描述 ``F.relu`` Expr 删除为例

>>> graph = traced_resnet.layer1.graph
>>> relu_exprs = graph.get_function_by_type(F.relu).as_list()
>>> relu_exprs
[%22:   relu_out = nn.relu(bn1_out, ),
 %30:   relu_out_1 = nn.relu(iadd_out, ),
 %38:   relu_out = nn.relu(bn1_out, ),
 %46:   relu_out_1 = nn.relu(iadd_out, )]

将获取到的所有以 ``F.relu`` 的输出作为输入的 Expr 替换为以 ``F.relu`` 的输入作为输入

>>> for id, expr in enumerate(relu_exprs):
...     cur_graph = expr.top_graph
...     relu_inp_node = expr.inputs[0]
...     relu_out_node = expr.outputs[0]
...     cur_graph.replace_node({relu_out_node: relu_inp_node})
...     cur_graph.compile()

这里可以看到在 layer1 的 graph 中找不到描述 ``F.relu`` 的 Expr 了

>>> graph.get_function_by_type(F.relu).as_list()
[]

insert_exprs
------------
向 InternalGraph 中插入 Expr。
可用于插入 ``function`` 或 ``Module`` ，
在插入的过程中将这些 ``function`` 或 ``Module`` 解析为 Expr 或 TracedModule。

一般与 ``replace_node`` 和 ``compile`` 一起使用完成插入 Expr 的操作。

:py:meth:`.InternalGraph.insert_exprs` ``(expr: Optional[Expr] = None)``
    向 Graph 中插入 Expr

    * ``expr`` 在 `_exprs` 属性中的 ``expr`` 之后插入解析 ``function`` 或 ``Module`` 的 expr。
      若为 None，则会根据输入自动计算向什么位置插入 Expr。

在 ``insert_exprs`` 的作用域里，``TensorNode`` 可以当作 ``Tensor`` 使用， ``ModuleNode`` 可以当作 ``Module``。

**示例1：** 向 layer1 中的所有 ``F.relu`` 后插入一个 ``F.neg`` 函数

>>> graph = traced_resnet.layer1.graph
>>> relu_exprs = graph.get_function_by_type(F.relu).as_list()
>>> for id, expr in enumerate(relu_exprs):
...     cur_graph = expr.top_graph
...     relu_out_node = expr.outputs[0]
...     with cur_graph.insert_exprs():
...         # 此处可直接将 TensorNode 输入到 F.neg 中
...         neg_out_node = F.neg(relu_out_node)
...     # 将所有以 relu_out_node 作为输入的 Expr 替换为以 neg_out_node 作为输入
...     cur_graph.replace_node({relu_out_node: neg_out_node})
...     cur_graph.compile()

可以看到在最后一个 ``cur_graph`` 中描述 ``F.relu`` 的 Expr 后有一个新插入的描述 ``F.neg`` 的 Expr

>>> cur_graph
ResNet_layer1_1.Graph (self, x) {
        ...
        %38:    relu_out = nn.relu(bn1_out, )
        %185:   neg_out = elemwise.neg(relu_out, )
        ...
        %46:    relu_out_1 = nn.relu(iadd_out, )
        %186:   neg_out_1 = elemwise.neg(relu_out_1, )
        return neg_out_1
}

**示例2：** 将 layer1 中的所有 ``F.relu`` 替换为 ``F.relu6``

>>> graph = traced_resnet.layer1.graph
>>> relu_exprs = graph.get_function_by_type(F.relu).as_list()
>>> for id, expr in enumerate(relu_exprs):
...     cur_graph = expr.top_graph
...     relu_inp_node = expr.inputs[0]
...     relu_out_node = expr.outputs[0]
...     with cur_graph.insert_exprs():
...         # 此处可直接将 TensorNode 输入到 MegEngine 的函数中
...         relu6_out_node = F.relu6(relu_inp_node)
...     # 将所有以 relu_out_node 作为输入的 Expr 替换为以 relu6_out_node 作为输入
...     cur_graph.replace_node({relu_out_node: relu6_out_node})
...     cur_graph.compile()

可以看到在最后一个 ``cur_graph`` 中描述 ``F.relu`` 的 Expr 均变为了 ``F.relu6`` 的 Expr

>>> cur_graph
ResNet_layer1_1.Graph (self, x) {
        ...
        %189:   relu6_out = nn.relu6(bn1_out, )
        %185:   neg_out = elemwise.neg(relu6_out, )
        ...
        %190:   relu6_out_1 = nn.relu6(iadd_out, )
        %186:   neg_out_1 = elemwise.neg(relu6_out_1, )
        return neg_out_1
}

**示例3：** 向 resnet18 中插入 Module

.. code:: python

    class MyNeg(M.Module):
        def forward(self, x):
            return x * -1
    myneg = MyNeg()

向 resnet18 中插入 ``myneg`` 这个自定义的 Module，完成使模型输出乘 -1 的功能，首先
需要将 ``myneg`` 设为 ``traced_resnet`` 的一个 attribute

>>> setattr(traced_resnet, "neg", myneg)

获取 graph 的输出 Node，以及 ``traced_resnet`` 所对应的 ModuleNode

>>> graph = traced_resnet.graph
>>> self_node = graph.inputs[0] # 此 node 为 traced_resnet 所对应的 ModuleNode
>>> out_node = graph.outputs[0]

调用 ``neg`` 来将其插入到 graph 中, 在图手术模式下，``self_node`` 等价于 ``traced_resnet``

>>> with graph.insert_exprs():
...     neg_node = getattr(self_node, "neg")(out_node)
... graph.replace_node({out_node: neg_node})
... graph.compile()
>>> graph
ResNet.Graph (self, x) {
        ...
        %182:   fc_out = fc(flatten_out, )
        %183:   neg = getattr(self, "neg") -> (Module)
        %184:   neg_out = neg(fc_out, )
        return neg_out
}

可以看到成功将 ``myneg`` 插入到了 graph 中, 并且 ``MyNeg`` 这个非 MegEngine 内置
的 Module 也有其对应的名为 ``ResNet_neg`` 的 graph

>>> traced_resnet.neg.graph
ResNet_neg.Graph (self, x) {
    %187:   mul_out = x.__mul__(-1, )
    return mul_out
}


.. warning::

    由于 Tensor 的 ``__setitem__`` 比较特殊，因此在图手术模式下对 TensorNode 进行赋值时，需要特别注意要图手术结果是否符合预期。

    直接以 TensorNode 赋值结果作为输出
    
    .. code:: python

        # x_node 是一个 TensorNode , x_node 的 name 为 x_node
        x = x_node
        with graph.insert_exprs():
            # 此操作会解析为 setitem_out = x_node.__setietm__(0, 1, )
            # 此时变量 x 依然对应的是 x_node
            x[0] = 1  
            # 此操作会解析为 setitem_out_1 = setitem_out.__setietm__(0, 2, )
            # 此时变量 x 依然对应的是 x_node
            x[0] = 2  

        # 此处实际替换的 x 依然为 x_node
        graph.replace_node({* : x}) 

    以其它操作生成的 TensorNode 作为输出

    .. code:: python

        with graph.insert_exprs():
            # 此操作会解析为 setitem_out = x_node.__setietm__(0, 1, )
            #  此时变量 x 依然对应的是 x_node
            x[0] = 1
            # 此操作会解析为 mul_out = setitem_out.__mul__(1, )
            # 此时变量 x 对应的是 mul_out
            x = x * 1
        # 此处实际替换的 x 为 mul_out
        graph.replace_node({* : x})

wrap
----
有时不希望插入的函数被展开为 megengine 内置的 function, 此时可以用 :py:meth:`~.traced_module.wrap` 将自定义的函数当作 megengine 内置函数处理，
即不再 ``trace`` 到自定义函数内部。

:py:meth:`~.traced_module.wrap` ``(func: Callable)``
    将自定义函数注册为内置函数

    * ``func`` 为一个可调用的函数。 

**示例：**

将 layer1 中的所有 ``F.relu`` 替换为自定义的 ``my_relu6``

.. code:: python

    @tm.wrap
    def my_relu6(x):
        x = F.minimum(F.maximum(x, 0), 6)
        return x

与替换为 ``F.relu6`` 类似，只调用 ``my_relu6`` 就完成了 ``trace`` 并将新的 Expr 插入到 Graph 中

>>> graph = traced_resnet.layer1.graph
>>> relu_exprs = graph.get_function_by_type(F.relu).as_list()
>>> for id, expr in enumerate(relu_exprs):
...     cur_graph = expr.top_graph
...     relu_inp_node = expr.inputs[0]
...     relu_out_node = expr.outputs[0]
...     with cur_graph.insert_exprs():
...         # 此处可直接将 TensorNode 输入到 MegEngine 的函数中
...         relu6_out_node = my_relu6(relu_inp_node)
...     # 将所有以 relu_out_node 作为输入的 Expr 替换为以 relu6_out_node 作为输入
...     cur_graph.replace_node({relu_out_node: relu6_out_node})
...     cur_graph.compile()

可以看到在最后一个 ``cur_graph`` 中描述 ``F.relu`` 的 Expr 均变为了 ``my_relu6`` 的 Expr

>>> cur_graph
ResNet_layer1_1.Graph (self, x) {
        ...
        %185:   my_relu6_out = __main__.my_relu6(bn1_out, )
        ...
        %186:   my_relu6_out_1 = __main__.my_relu6(iadd_out, )
        return my_relu6_out_1
}

.. warning::

    * 被 ``wrap`` 的函数的返回值必须仅为 Tensor 或内部元素为 Tensor 的容器

    * 需要注意的是，当自定义的 function 或 Module 未被 ``trace`` 到 function 或 Module 内部时，
      序列化后的 TracedModule 可以脱离源码被 load，但无法运行

.. _tracedmodule_graph_optimize:

TracedMdoule 内置模型优化
=========================

.. warning::

    内置模型优化的实现与接口持续完善中，欢迎在文档或 Github 提交使用反馈。

我们提供了一些常用图手术实现来优化模型，包括：

* FuseConvBn：将 BatchNorm 融合到 Convolution 中
* FuseAddMul：融合连续的常量加法或常量乘法
* BackwardFoldScale：将卷积之后的常量乘法融合到卷积中

使用这些优化的接口统一为 ``optimize``：

.. code-block:: python

    def optimize(
        module: TracedModule, enabled_pass: List[str] = ["FuseConvBn"],
    ) -> TracedModule:...

该函数传入一个 TracedMdoule，一个待优化选项的列表 enabled_pass，在函数内部会将传入的优化选项一一作用至 TracedMdoule 上，
并返回优化后的 TracedMdoule。需要注意的是，我们不会在原 ``module`` 上进行优化，而是在原 ``module`` 的副本上进行优化。

下面将通过一些例子来介绍如何使用该接口。

FuseConvBn
----------

将 BatchNorm 融合到 Convolution 中是模型加速的一个非常有效的手段。
我们实现的 FuseConvBn 支持将内置 ``F.batchnorm`` 或 `M.BatchNorm2d` 融合至 ``F.conv2d`` 或 ``M.Conv2d`` 中。

如下列的例子，将 resnet18 中的 bn 都融合至 conv 中：

>>> optimized_resnet = tm.optimize(traced_resnet, enabled_pass="FuseConvBn")
>>> getattr(optimized_resnet.layer1,"0").graph 
ResNet_layer1_0.Graph (self, x) {
        %18:    conv1 = getattr(self, "conv1") -> (Conv2d)
        %220:   conv1_out = conv1(x, )
        %22:    relu_out = nn.relu(conv1_out, )
        %23:    conv2 = getattr(self, "conv2") -> (Conv2d)
        %218:   conv2_out = conv2(relu_out, )
        %27:    downsample = getattr(self, "downsample") -> (Identity)
        %28:    downsample_out = downsample(x, )
        %29:    iadd_out = conv2_out.__iadd__(downsample_out, )
        %30:    relu_out_1 = nn.relu(iadd_out, )
        return relu_out_1
}

调用 FuseConvBn 选项后，会将图中类似 ``bn(conv(x))`` 的表达式进行融合。

.. warning::

    * 该优化目前仅支持 2d 的 conv 和 bn
    * 当一个 conv module 被调用多次时，我们将会对其拷贝，并设置一个新的 name，以使其转变为仅被调用一次
    
    例如，对如下的计算过程中使用的 conv 和 bn 进行融合时

    .. code-block:: python

        x = conv_0(x1)
        y1 = bn_0(x)

        x = conv_0(x2)
        y2 = bn_0(x)
        y = y1 + y2
        
    由于 ``conv_0`` 被使用了两次，因此我们将会将 ``conv_0`` 进行拷贝得到一个新的 module 为 ``conv_0_1``，
    同时第一次调用 ``conv_0`` 将变成调用 ``conv_0_1``，以保证融合结果正确。

    .. code-block:: python

        x = conv_0_1(x1)
        y1 = bn_0(x)

        x = conv_0(x2)
        y2 = bn_0(x)
        y = y1 + y2

FuseAddMul
----------

FuseaddMul 是将一些连续的常量乘法或常量加法融合，使得图中的运算变少，提高模型运行速度。

对于如下运算

.. code-block:: python

    class MyModule(M.Module):
        def __init__(self, ):
            super().__init__()
            self.scale = mge.Tensor([1,2])
            
        def forward(self, x):
            x = x * self.scale[0]
            x = 3 * x
            x = 3 + x
            x = x - self.scale[1]
            return x

我们会将 ``x * self.scale[0]`` 和 ``3 * x`` 融合为 ``x * 3``, ``3 + x`` 和 ``x - self.scale[1]`` 融合为 ``x + 1``，
优化之后的 graph 如下：

>>> optimized_resnet = tm.optimize(traced_mymodule, enabled_pass="FuseaddMul")
>>> optimized_resnet.graph
MyModule.Graph (self, x) {
        %21:    const_tensor_1 = Constant() -> (Tensor)
        %22:    mul_out_1 = x.__mul__(const_tensor_1, )
        %19:    const_tensor = Constant() -> (Tensor)
        %20:    add_out_2 = mul_out_1.__add__(const_tensor, )
        return add_out_2
}

.. warning::

    目前该优化仅支持 shape 为 (1,) 的 Tensor 或数值常量

BackwardFoldScale
-----------------

BackwardFoldScale 是将卷积之后的一些常量乘法中的常量吸到卷积的参数里。

对于如下运算

.. code-block:: python

    class MyModule(M.Module):
        def __init__(self, ):
            super().__init__()
            self.conv = M.Conv2d(3,3,1,1,0)
            self.scale = mge.Tensor([1,2])
            
        def forward(self, x):
            x = self.conv(x)
            x = F.relu(x)
            x1 = x * self.scale[0]
            x2 = F.reshape(x, -1)
            x2 = x2 * self.scale[1]
            y = x1.reshape(-1)*2 + x2
            return y

我们会将 ``x1.reshape(-1)*2`` 和 ``x * self.scale[0]`` 这一路常量乘法反传至 ``self.conv``，
以及 ``x2 * self.scale[1]`` 这一路常量乘法反传至 ``self.conv``，然后将所有的常量融合至卷积里，
当遇到不同分支反传过来的常量乘法时，会检测不同分支反传的常量是否相同，不相同则反传失败。

优化后的 graph 如下:

>>> optimized_resnet = tm.optimize(traced_mymodule, enabled_pass="BackwardFoldScale")
>>> optimized_resnet.graph
MyModule.Graph (self, x) {
        %2:     conv = getattr(self, "conv") -> (Conv2d)
        %3:     conv_out = conv(x, )
        %4:     relu_out = nn.relu(conv_out, )
        %8:     reshape_out = tensor.reshape(relu_out, -1, )
        %11:    reshape_out_1 = relu_out.reshape(-1, )
        %13:    add_out = reshape_out_1.__add__(reshape_out, )
        return add_out
}

.. warning::

    * 目前该优化仅支持 shape 为 (1,) 的 Tensor 或数值常量

TracedModule 的局限
===================

* 不支持动态控制流，动态控制流是指 ``if`` 语句中的 ``condition`` 随输入 Tensor 的变化而变化，
  或者是 ``for``, ``while`` 每次运行的语句不一样。当 ``trace`` 到控制流时，
  仅会记录并解释满足条件的那个分支。

* 不支持全局变量（Tensor），即跨 Module 使用 ``Tensor`` 将会得到不可预知的结果，如下面的例子：

  .. code:: python

    g_tensor = mge.Tensor([0])
    class Mod(M.Module):
        def forward(self, x):
            x = g_tensor + 1
            return x

* 被 ``trace`` 的 Module 或 function 参数中的非 ``Tensor`` 类型，
  将会被看作是常量存储在 Expr 的 :py:attr:`~.Expr.const_val` 属性中，
  并且该值将不会再变化。

* 在模型中使用 MegEngine 内置的 function 时， **推荐** 下面这中调用方法：

  .. code:: python

    import megengine.functional as F

    def my_relu(x):
        return F.relu(x) * x

  **不推荐** 下面这中调用方法：

  .. code:: python

    from megengine.functional import relu

    def my_relu(x):
        return relu(x) * x

* 当被 ``trace`` 的自定义 Module 被调用了多次，并且每次传入参数中的非 ``Tensor`` 数据不一致时，
  将会被 ``trace`` 出多个 Graph。此时将无法通过 :py:attr:`.TracedModule.graph` 属性访问 Graph，
  只能通过对应 Moldule 的 ``CallMethod`` Expr 访问，如下面的例子：

  .. dropdown:: example

    .. code:: python

        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm

        class Mod(M.Module):
            def forward(self, x, b):
                x  = x + b
                return x

        class Net(M.Module):
            def __init__(self, ):
                super().__init__()
                self.mod = Mod()

            def forward(self, x):
                x = self.mod(x, 1)
                x = self.mod(x, 2)
                return x

        net = Net()
        inp = F.zeros(shape=(1, ))

        traced_net = tm.trace_module(net, inp)

        print(traced_net.graph)
        '''
        Net.Graph (self, x) {
                %5:     mod = getattr(self, "mod") -> (Module)
                %6:     mod_out = mod(x, 1, )
                %10:    mod_1 = getattr(self, "mod") -> (Module)
                %11:    mod_1_out = mod_1(mod_out, 2, )
                return mod_1_out
        }
        '''
        # 此时 traced_net.mod 将会被 trace 出 2 个 graph，因此无法直接访问 graph 属性
        try:
            print(traced_net.mod.graph)
        except:
            print("error")

        # 可通过 mod 的 CallMethod Expr 访问对应的 Graph
        print(traced_net.graph.get_expr_by_id(6).as_unique().graph)
        '''
        mod.Graph (self, x) {
                %9:     add_out = x.__add__(1, )
                return add_out
        }
        '''
        print(traced_net.graph.get_expr_by_id(11).as_unique().graph)
        '''
        mod_1.Graph (self, x) {
                %14:    add_out = x.__add__(2, )
                return add_out
        }
        '''
