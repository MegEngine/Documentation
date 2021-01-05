# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ------------------------------------------------------------------------------
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
# ------------------------------------------------------------------------------

import math
import megengine.hub as hub
import megengine.module as M
import megengine.functional as F


class ConvBlock(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=M.ReLU()):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.use_bn = use_bn

        self.conv = M.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = M.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if not self.activation is None:
            x = self.activation(x)
        return x


class DwConvBlock(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=M.ReLU()):
        super(DwConvBlock, self).__init__()
        self.conv = ConvBlock(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation,
                              groups=out_channels,
                              bias=bias,
                              use_bn=use_bn,
                              bn_eps=bn_eps,
                              activation=activation)

    def forward(self, x):
        return self.conv(x)


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    return M.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias
    )


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=M.ReLU()):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=M.ReLU()):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bias=False,
                    bn_eps=1e-5,
                    activation=M.ReLU()):
    return DwConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)


def dwconv5x5_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=2,
                    dilation=1,
                    bias=False,
                    bn_eps=1e-5,
                    activation=M.ReLU()):
    return DwConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        activation=activation)


def round_channels(channels, divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor,
                           divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

class GlobalAvgPool2D(M.Module):

    def forward(self, x):
        h = x.shape[-2]
        w = x.shape[-1]
        x = F.avg_pool2d(x, [h, w])
        return x

class Swish(M.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.activate = M.Sigmoid()

    def forward(self, x):
        return x * self.activate(x)


class SEBlock(M.Module):

    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation=M.ReLU(),
                 out_activation=M.Sigmoid()):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)
        self.pool = GlobalAvgPool2D()
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True)
        self.activ = mid_activation
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels,
            bias=True)
        self.sigmoid = out_activation

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


def calc_tf_padding(x,
                    kernel_size,
                    stride=1,
                    dilation=1):
    height, width = x.size()[2:]
    oh = math.ceil(height / stride)
    ow = math.ceil(width / stride)
    pad_h = max((oh - 1) * stride + (kernel_size - 1) * dilation + 1 - height, 0)
    pad_w = max((ow - 1) * stride + (kernel_size - 1) * dilation + 1 - width, 0)
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


class EffiDwsConvUnit(M.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiDwsConvUnit, self).__init__()
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)

        self.dw_conv = dwconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation)
        self.se = SEBlock(
            channels=in_channels,
            reduction=4,
            mid_activation=activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.tf_mode:
            # TODO 未实现
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3))
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class EffiInvResUnit(M.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 exp_factor,
                 se_factor,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInvResUnit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.tf_mode = tf_mode
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor
        dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=activation)
        self.conv2 = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=(0 if tf_mode else (kernel_size // 2)),
            bn_eps=bn_eps,
            activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                mid_activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=self.kernel_size, stride=self.stride))
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class EffiInitBlock(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 activation,
                 tf_mode):
        super(EffiInitBlock, self).__init__()
        self.tf_mode = tf_mode

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            padding=(0 if tf_mode else 1),
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        if self.tf_mode:
            x = F.pad(x, pad=calc_tf_padding(x, kernel_size=3, stride=2))
        x = self.conv(x)
        return x

class EfficientNet(M.Module):

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 kernel_sizes,
                 strides_per_stage,
                 expansion_factors,
                 dropout_rate=0.2,
                 tf_mode=False,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(EfficientNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        activation = Swish()

        self.features = []
        init_block = EffiInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            activation=activation,
            tf_mode=tf_mode)
        self.features.append(init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            kernel_sizes_per_stage = kernel_sizes[i]
            expansion_factors_per_stage = expansion_factors[i]
            stage = []
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                stride = strides_per_stage[i] if (j == 0) else 1
                if i == 0:
                    unit = EffiDwsConvUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode)
                    stage.append(unit)
                else:
                    unit = EffiInvResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        exp_factor=expansion_factor,
                        se_factor=4,
                        bn_eps=bn_eps,
                        activation=activation,
                        tf_mode=tf_mode)
                    stage.append(unit)
                in_channels = out_channels
            self.features += stage
        final_block = conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bn_eps=bn_eps,
            activation=activation)
        self.features.append(final_block)
        in_channels = final_block_channels
        final_pool = GlobalAvgPool2D()
        self.features.append(final_pool)
        self.features = M.Sequential(*self.features)

        self.output = []
        if dropout_rate > 0.0:
            dropout = M.Dropout(dropout_rate)
            self.output.append(dropout)
        fc = M.Linear(
            in_features=in_channels,
            out_features=num_classes)
        self.output.append(fc)
        self.output = M.Sequential(*self.output)

    def forward(self, x):
        x = self.features(x)
        x = F.mean(x, axis=3, keepdims=True)
        x = x.reshape(x.shape[0], -1)
        x = self.output(x)
        return x


def get_efficientnet(version,
                     in_size,
                     tf_mode=False,
                     bn_eps=1e-5,
                     **kwargs):
    if version == "b0":
        assert (in_size == (224, 224))
        depth_factor = 1.0
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b1":
        assert (in_size == (240, 240))
        depth_factor = 1.1
        width_factor = 1.0
        dropout_rate = 0.2
    elif version == "b2":
        assert (in_size == (260, 260))
        depth_factor = 1.2
        width_factor = 1.1
        dropout_rate = 0.3
    elif version == "b3":
        assert (in_size == (300, 300))
        depth_factor = 1.4
        width_factor = 1.2
        dropout_rate = 0.3
    elif version == "b4":
        assert (in_size == (380, 380))
        depth_factor = 1.8
        width_factor = 1.4
        dropout_rate = 0.4
    elif version == "b5":
        assert (in_size == (456, 456))
        depth_factor = 2.2
        width_factor = 1.6
        dropout_rate = 0.4
    elif version == "b6":
        assert (in_size == (528, 528))
        depth_factor = 2.6
        width_factor = 1.8
        dropout_rate = 0.5
    elif version == "b7":
        assert (in_size == (600, 600))
        depth_factor = 3.1
        width_factor = 2.0
        dropout_rate = 0.5
    elif version == "b8":
        assert (in_size == (672, 672))
        depth_factor = 3.6
        width_factor = 2.2
        dropout_rate = 0.5
    else:
        raise ValueError("Unsupported EfficientNet version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 2, 3, 3, 4, 1]
    downsample = [1, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 40, 80, 112, 192, 320]
    expansion_factors_per_layers = [1, 6, 6, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 3, 5, 3, 5, 5, 3]
    strides_per_stage = [1, 2, 2, 2, 1, 2, 1]
    final_block_channels = 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                          zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])
    strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(strides_per_stage, layers, downsample), [])
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_eps=bn_eps,
        in_size=in_size,
        **kwargs)

    return net


def check(func):
    import megengine
    import numpy as np
    def _wrapped(*args, **kwargs):
        print("check", func.__name__)
        m = func()
        x = megengine.tensor(np.zeros([1,3,*m.in_size], dtype=np.float32))
        y = m(x)
        print("logits shape", y.shape)
        return m
    return _wrapped

@check
def efficientnet_b0():
    return get_efficientnet("b0", (224, 224))


@check
def efficientnet_b1():
    return get_efficientnet("b1", (240, 240))


@check
def efficientnet_b2():
    return get_efficientnet("b2", (260, 260))


@check
def efficientnet_b3():
    return get_efficientnet("b3", (300, 300))


@check
def efficientnet_b4():
    return get_efficientnet("b4", (380, 380))


@check
def efficientnet_b5():
    return get_efficientnet("b5", (456, 456))


@check
def efficientnet_b6():
    return get_efficientnet("b6", (528, 528))


@check
def efficientnet_b7():
    return get_efficientnet("b7", (600, 600))


@check
def efficientnet_b8():
    return get_efficientnet("b8", (672, 672))


if __name__ == "__main__":

    import megengine as mge
   
    m = efficientnet_b0()
    m = efficientnet_b1()
    m = efficientnet_b2()
    m = efficientnet_b3()
    m = efficientnet_b4()
    m = efficientnet_b5()
    m = efficientnet_b6()
    m = efficientnet_b7()
    m = efficientnet_b8()

