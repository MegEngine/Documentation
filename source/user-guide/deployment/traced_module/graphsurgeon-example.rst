.. _graphsurgeon-example:

==============
图手术 Example
==============

为模型添加前后处理
==================

TracedModule 可以被反复的 trace，因此在加前后处理时，推荐以新写一个 Module 的形式给模型加前后处理。

.. admonition:: 在 Module 里加前后处理
    :class: dropdown

    .. code:: python

        import numpy as np
        import pickle
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm
        
        class Main(M.Module):
            def forward(self, x):
                return x
        
        class PreProcess(M.Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, x, y):
                x = x*y
                return x
        
        class PostProcess(M.Module):
            def __init__(self):
                super().__init__()
        
            def forward(self, x, y):
                x = x/y
                return x
        
        class Net(M.Module):
            def __init__(self, traced_module):
                super().__init__()
                self.pre_process = PreProcess()
                self.traced_module = traced_module
                self.post_process = PostProcess()
        
            def forward(self, x, y):
                x = self.pre_process(x, y)
                x = self.traced_module(x)
                x = self.post_process(x, y)
                return x
        
        if __name__ == "__main__":
            module = Main()
            x = F.zeros((1, 14, 8, 8))
            traced_module = tm.trace_module(module, x)
            obj = pickle.dumps(traced_module)
            traced_module = pickle.loads(obj)
            
            # 新写一个 module，将 之前 dump 的 TracedModule 作为该 module 的一个子 module
            new_module = Net(traced_module)
        
            x = F.zeros((1, 14, 8, 8))
            y = F.ones((1, 14, 8, 8))
            traced_module = tm.trace_module(new_module, x, y)
            predict = traced_module(x, y)
            np.testing.assert_equal(x.numpy(), predict.numpy())

当然也可以用图手术的方式添加前后处理，最终效果是一样的。

.. admonition:: 图手术加前后处理
    :class: dropdown

    .. code:: python

        import numpy as np
        import pickle
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm
        
        class Main(M.Module):
            def forward(self, x):
                return x
        
        def pre_process(x, y):
                x = x*y
                return x
        
        def post_process(x, y):
                x = x/y
                return x
        
        if __name__ == "__main__":
            module = Main()
            x = F.zeros((1, 14, 8, 8))
            traced_module = tm.trace_module(module, x)
            obj = pickle.dumps(traced_module)
            traced_module = pickle.loads(obj)
        
            graph = traced_module.graph
            x_node = graph.inputs[1]
            
        
            # 为 graph 新增一个 input
            y_node = graph.add_input_node(shape = (1, 14, 8, 8), name="y")
            
            with graph.insert_exprs():
                new_x_node = pre_process(x_node, y_node)
        
            # 将使用了 x_node 的 Expr 替换为 new_x_node
            graph.replace_node({x_node: new_x_node})
        
            # 由于后处理中涉及到 /y，不能将 y 自动生成为 0，因此特别的给 y_node 设置 value
            y = F.ones((1, 14, 8, 8))
            y_node.value = y
            orig_net_out_node = graph.outputs[0]
            with graph.insert_exprs():
                new_out_node = post_process(orig_net_out_node, y_node)
            
            # 通过 replace_node 将 graph.outputs 替换掉
            # 或者调用 graph.reset_outputs(new_out_node), 重新设置 graph 的输出
            graph.replace_node({orig_net_out_node:new_out_node})
        
            # 调用 compile 清理掉 graph 中与 outputs 无关的 Expr
            graph.compile()
            predict = traced_module(x, y)
            np.testing.assert_equal(x.numpy(), predict.numpy())


把一些常量吸收到卷积里
======================

对于一些基于 anchor 的检测算法，经常会在卷积的输出后，对卷积结果乘 ``stride`` 或除 ``anchor_size``，在推理部署时，可以将这些常量吸收到卷积里，基于 TracedModule 可以较容易的实现这些转换, 如下面的例子。

.. admonition:: 吸常量
    :class: dropdown

    .. code:: python

        import numpy as np
        import pickle
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm
        from megengine.traced_module.node import TensorNode
        import megengine as mge
        
        class Net(M.Module):
            def __init__(self,):
                super().__init__()
                self.conv = M.Conv2d(in_channels=3, out_channels=16, kernel_size=1, bias=True)
        
            def forward(self, x):
                x = self.conv(x)
                stride, anchor_size= 8, 128
                x = x * stride
                x = x / anchor_size
                return x
        
        def fuse_const():
            net = Net()
            inp = mge.Tensor(np.random.random(size = (1,3,16,16)), dtype=np.float32)
        
            traced_net = tm.trace_module(net, inp)
            obj = pickle.dumps(traced_net)
            traced_net = pickle.loads(obj)
        
            graph = traced_net.graph
        
            for div_expr in graph.get_method_by_type("__truediv__").as_list():
                div_self, div_inp = div_expr.args
                if isinstance(div_inp, TensorNode):
                    # 除数不是 TensorNode，就满足了我们的条件
                    continue
                mul_expr = div_self.expr
                mul_self, mul_inp = mul_expr.args
                call_conv_expr = mul_self.expr
        
                conv_node = call_conv_expr.inputs[0]
        
                # 直接通过 owner 访问 self.conv ，并修改其 weight 和 bias
                conv_module = conv_node.owner
                conv_module.weight = conv_module.weight * mul_inp / div_inp
                conv_module.bias = conv_module.bias * mul_inp / div_inp
        
                # 修改之后，要用 conv 的输出替换 div 的输出
                call_conv_expr.top_graph.replace_node({div_expr.outputs[0] : call_conv_expr.outputs[0]})
        
                # 把与 graph 输出无关的 expr 删掉
                call_conv_expr.top_graph.compile()
            
            gt = net(inp)
            actual = traced_net(inp)
            np.testing.assert_equal(gt.numpy(), actual.numpy())
        
        if __name__ == "__main__":
            fuse_const()

将一些 OP 转换为 fp16
=====================

对于一些计算量特别大的全连接层，会占用较多的存储资源，可以通过将其转换为 fp16 计算减少其占用的资源, 如下面的例子。

.. admonition:: 转 fp16
    :class: dropdown

    .. code:: python

        import numpy as np
        import pickle
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm
        import megengine as mge
        
        class Net(M.Module):
            def __init__(self,):
                super().__init__()
                self.linear_0 = M.Linear(3, 1024, bias = True)
                self.linear_1 = M.Linear(1024, 4096, bias=True)
        
            def forward(self, x):
                x = self.linear_0(x)
                x = self.linear_1(x)
                return x
        
        def to_fp16():
            net = Net()
            inp = mge.Tensor(np.random.random(size = (1,3)), dtype=np.float32)
        
            traced_net = tm.trace_module(net, inp)
            obj = pickle.dumps(traced_net)
            traced_net = pickle.loads(obj)
        
            graph = traced_net.graph
        
            for linear_node in graph.get_module_by_type(M.Linear).as_list():
                linear_module = linear_node.owner
                if linear_module.in_features * linear_module.out_features < 100*1024:
                    # 不满足条件的 Linear 跳过
                    continue
                # 将 weight 和 bias 转换为 fp16
                linear_module.weight = linear_module.weight.astype(np.float16)
                linear_module.bias = linear_module.bias.astype(np.float16)
        
                linear_call_expr = linear_node.users[0]
        
                # 把输入转换为 fp16
                inp_node = linear_call_expr.inputs[1]
                with linear_call_expr.top_graph.insert_exprs():
                    new_inp_node = inp_node.astype(np.float16)
                # 将 linear 的输入替换为fp16的输入
                linear_call_expr.replace_inputs({inp_node: new_inp_node})
        
                # 把输出转换为 fp16
                out_node = linear_call_expr.outputs[0]
                with linear_call_expr.top_graph.insert_exprs():
                    new_out_node = out_node.astype(np.float32)
                
                # 将 out_node 作为输入的 expr 的输入替换为 new_out_node
                linear_call_expr.top_graph.replace_node({out_node: new_out_node})
                linear_call_expr.top_graph.compile()
            
            gt = net(inp)
            actual = traced_net(inp)
            np.testing.assert_allclose(gt.numpy(), actual.numpy(), atol=5e-2)
        
        
        if __name__ == "__main__":
            to_fp16()

通过 InternalGraph  确定数据流向
================================

在量化训练时，常常会对 concat 的输入做某些约束，通过 TracedModule 可以轻易的找到这些 concat 的输入是来自于哪个内置的 function 或 Module 的输出，如下面的例子。

.. admonition:: find inputs
    :class: dropdown

    .. code:: python

        import numpy as np
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.traced_module as tm
        import megengine as mge
        
        class Net(M.Module):
            def __init__(self,):
                super().__init__()
                self.conv = M.Conv2d(3, 16, 1, bias=False)
                self.bn = M.BatchNorm2d(16)
                self.conv_bn = M.Sequential(
                    M.Conv2d(16, 16, 1,bias=False),
                    M.BatchNorm2d(16)
                )
        
            def forward(self, x):
                x = self.conv(x)
                x0 = self.bn(x)
                x1 = self.conv_bn(x0)
                x = F.concat((x0, x1), 1)
                return x
        
        
        def find_cat_inputs():
            net = Net()
            inp = mge.Tensor(np.random.random(size = (1,3, 16, 16)), dtype=np.float32)
        
            traced_net = tm.trace_module(net, inp)
            flattened_net = traced_net.flatten()
            cat_expr = flattened_net.graph.get_function_by_type(F.concat).as_unique()
            print(cat_expr)
            # _orig_name 包含了其是由哪个 builtin 的 module 输出的信息
            print([n._orig_name for n in cat_expr.inputs])
            """
            %8:     concat_out = tensor.concat((bn_out, conv_bn_out), 1, )
            ['bn_out', 'conv_bn.1_out']
            """
        
        if __name__ == "__main__":
            find_cat_inputs()

Conv 和 BN 融合
===============

在 推理 或 量化训练 时，常常需要将 Conv 和 Bn 融合到一起，基于 TracedModule 的 Graph 可以找到满足融合条件的 Conv 和 Bn，并以图手术的方式将其融合，如下面的例子。

.. admonition:: fuse bn
    :class: dropdown

    .. code:: python

        import numpy as np
        import pickle
        
        import megengine.functional as F
        import megengine.module as M
        import megengine.module.qat as Q
        import megengine.traced_module as tm
        from megengine.traced_module.expr import CallMethod
        from megengine.traced_module.node import ModuleNode
        import megengine as mge
        
        class Net(M.Module):
            def __init__(self,):
                super().__init__()
                self.conv = M.Conv2d(3,16,1, bias=False)
                self.bn = M.BatchNorm2d(16)
                self.conv_bn = M.Sequential(
                    M.Conv2d(16,16,1,bias=False),
                    M.BatchNorm2d(16)
                )
        
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = F.relu(x)
                x = self.conv_bn(x)
                return x
        
        def _fuse_conv_bn(conv : M.Conv2d, bn : M.BatchNorm2d = None):
            weight, bias = conv.weight, conv.bias
            target_cls = M.ConvBn2d
            if not conv.training:
                class FakeCls:
                    def __init__(self, conv, bn):
                        self.conv = conv
                        self.bn = bn
                    def apply_quant_weight(self, inp):
                        return inp
        
                weight, bias = Q.ConvBn2d.fold_weight_bias(
                    FakeCls(conv, bn),
                    bn.running_mean,
                    bn.running_var
                )
                target_cls = M.Conv2d
            this_module = target_cls(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                conv.bias is not None,
                conv.conv_mode,
                conv.compute_mode,
                name=conv.name,
            )
            if conv.training:
                this_module.conv.weight = weight
                this_module.conv.bias = bias
                this_module.bn = bn
            else:
                this_module.weight = weight
                this_module.bias = bias
            return this_module
        
        def fuse_bn_transform():
            net = Net()
            inp = mge.Tensor(np.random.random(size = (1,3, 16, 16)), dtype=np.float32)
        
            traced_net = tm.trace_module(net, inp)
            obj = pickle.dumps(traced_net)
            traced_net = pickle.loads(obj)
        
            graph = traced_net.graph
        
            for conv_node in graph.get_module_by_type(M.Conv2d).as_list():
                if len(conv_node.users) > 1:
                    continue
                conv_expr = conv_node.users[0]
                conv_out_node = conv_expr.outputs[0]
        
                if len(conv_out_node.users) > 1:
                    # conv -> bn，conv 的输出只能被 bn 使用
                    continue
                
                # 判断 conv 之后的 expr 是否是 bn
                bn_expr = conv_out_node.users[0]
                if not isinstance(bn_expr, CallMethod):
                    continue
                bn_node = bn_expr.inputs[0]
                if not isinstance(bn_node, ModuleNode) or bn_node.module_type != M.BatchNorm2d:
                    continue
                
                conv_module = conv_node.owner
                bn_module = bn_node.owner
        
                new_module = _fuse_conv_bn(conv_module, bn_module)
        
                cur_graph = conv_node.top_graph
                self_node = cur_graph.inputs[0]
                self_module = self_node.owner
                name = conv_module._name
        
                # 将 fuse 后的 module 设置到 调用 conv 的 module 上
                setattr(self_module, conv_module._name, new_module)
                inp_node = conv_expr.inputs[1]
                bn_out_node = bn_expr.outputs[0]
        
                # 将 fuse 后的 module 以图手术的方式 insert 到 graph 中
                with cur_graph.insert_exprs():
                    fused_conv_out = getattr(self_node, name)(inp_node)
        
                cur_graph.replace_node({bn_out_node: fused_conv_out})
                cur_graph.compile()
        
            gt = net(inp)
            actual = traced_net(inp)
            np.testing.assert_allclose(gt.numpy(), actual.numpy(), atol=5e-2)
        
        
        if __name__ == "__main__":
            fuse_bn_transform()
