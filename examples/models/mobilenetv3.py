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

import os
import megengine.hub as hub
import megengine.module as M
import megengine.functional as F


def round_channels(channels,
                   divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


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
        bias=bias)


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


class SEBlock(M.Module):

    def __init__(self,
                 channels,
                 reduction=16,
                 round_mid=False,
                 mid_activation=M.ReLU(),
                 out_activation=M.Sigmoid()):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)
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
        w = F.mean(x, axis=3, keepdims=True)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ReLU6(M.Module):

    def forward(self, x):
        return F.maximum(x, 6)


class HSwish(M.Module):

    def __init__(self):
        super(HSwish, self).__init__()
        self.actv = ReLU6()

    def forward(self, x):
        return x * self.actv(x + 3.0) / 6.0


class HSigmoid(M.Module):

    def __init__(self):
        super(HSigmoid, self).__init__()
        self.actv = ReLU6()

    def forward(self, x):
        return self.actv(x + 3.0) / 6.0


class MobileNetV3Unit(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_channels,
                 stride,
                 use_kernel3,
                 activation,
                 use_se):
        super(MobileNetV3Unit, self).__init__()
        assert (exp_channels >= out_channels)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se
        self.use_exp_conv = exp_channels != out_channels
        mid_channels = exp_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                round_mid=True,
                out_activation=HSigmoid())
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV3FinalBlock(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se):
        super(MobileNetV3FinalBlock, self).__init__()
        self.use_se = use_se

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=HSwish())
        if self.use_se:
            self.se = SEBlock(
                channels=out_channels,
                reduction=4,
                round_mid=True,
                out_activation=HSigmoid())

    def forward(self, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class MobileNetV3Classifier(M.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 dropout_rate):
        super(MobileNetV3Classifier, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.activ = HSwish()
        if self.use_dropout:
            self.dropout = M.Dropout(dropout_rate)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class MobileNetV3(M.Module):

    def __init__(self,
                 channels,
                 exp_channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 use_relu,
                 use_se,
                 first_stride,
                 final_use_se,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = []
        init_block = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation=HSwish())
        self.features.append(init_block)

        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = []
            for j, out_channels in enumerate(channels_per_stage):
                exp_channels_ij = exp_channels[i][j]
                stride = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                use_kernel3 = kernels3[i][j] == 1
                activation = M.ReLU() if use_relu[i][j] == 1 else HSwish()
                use_se_flag = use_se[i][j] == 1
                unit = MobileNetV3Unit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    exp_channels=exp_channels_ij,
                    use_kernel3=use_kernel3,
                    stride=stride,
                    activation=activation,
                    use_se=use_se_flag)
                stage.append(unit)
                in_channels = out_channels
            self.features += stage
        final_block = MobileNetV3FinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            use_se=final_use_se)
        self.features.append(final_block)
        in_channels = final_block_channels
        final_pool = M.AvgPool2d(
            kernel_size=7,
            stride=1)
        self.features.append(final_pool)
        self.features = M.Sequential(*self.features)

        self.output = MobileNetV3Classifier(
            in_channels=in_channels,
            out_channels=num_classes,
            mid_channels=classifier_mid_channels,
            dropout_rate=0.2)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.reshape(x.shape[0], -1)
        return x


def get_mobilenetv3(version,
                    width_scale,
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    net = MobileNetV3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
        **kwargs)

    return net


def mobilenetv3_small_w7d20(**kwargs):
    return get_mobilenetv3(version="small", width_scale=0.35, **kwargs)


def mobilenetv3_small_wd2(**kwargs):
    return get_mobilenetv3(version="small", width_scale=0.5, **kwargs)


def mobilenetv3_small_w3d4(**kwargs):
    return get_mobilenetv3(version="small", width_scale=0.75, **kwargs)


def mobilenetv3_small_w1(**kwargs):
    return get_mobilenetv3(version="small", width_scale=1.0, **kwargs)


def mobilenetv3_small_w5d4(**kwargs):
    return get_mobilenetv3(version="small", width_scale=1.25, **kwargs)


def mobilenetv3_large_w7d20(**kwargs):
    return get_mobilenetv3(version="large", width_scale=0.35, **kwargs)


def mobilenetv3_large_wd2(**kwargs):
    return get_mobilenetv3(version="large", width_scale=0.5, **kwargs)


def mobilenetv3_large_w3d4(**kwargs):
    return get_mobilenetv3(version="large", width_scale=0.75, **kwargs)


def mobilenetv3_large_w1(**kwargs):
    return get_mobilenetv3(version="large", width_scale=1.0, **kwargs)


def mobilenetv3_large_w5d4(**kwargs):
    return get_mobilenetv3(version="large", width_scale=1.25, **kwargs)



