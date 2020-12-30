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

import megengine.module as M
import megengine.functional as F


class FireConv(M.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(FireConv, self).__init__()
        self.conv = M.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.activation = M.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class FireUnit(M.Module):

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels, residual):
        super(FireUnit).__init__()

        self.residual = residual
        self.squeeze = FireConv(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            padding=0
        )
        self.expand1x1 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1,
            padding=0)
        self.expand3x3 = FireConv(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.squeeze(x)
        y1 = self.expand1x1(x)
        y2 = self.expand3x3(x)
        out = F.concat([y1, y2], axis=1)
        if self.residual:
            out = out + identity
        return out


class SqueezeInitBlock(M.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SqueezeInitBlock, self).__init__()
        self.conv = M.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2)
        self.activ = M.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class SqueezeNet(M.Module):

    def __init__(self,
                 channels,
                 residuals,
                 init_block_kernel_size,
                 init_block_channels,
                 maxpool_pad,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.feature = []
        init_block = SqueezeInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size
        )
        self.feature.append(init_block)
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = []

            pool = M.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=maxpool_pad[i]
            )
            stage.append(pool)

            for j, out_channels in enumerate(channels_per_stage):
                expand_channels = out_channels // 2
                squeeze_channels = out_channels // 8
                unit = FireUnit(
                    in_channels=in_channels,
                    squeeze_channels=squeeze_channels,
                    expand1x1_channels=expand_channels,
                    expand3x3_channels=expand_channels,
                    residual=((residuals is not None) and (residuals[i][j] == 1)))
                stage.append(unit)
                in_channels = out_channels
            self.feature += stage
        self.feature.append(M.Dropout(drop_prob=0.5))
        self.feature = M.Sequential(*self.feature)

        self.output = []
        final_conv = M.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1
        )
        self.output.append(final_conv)
        final_activ = M.ReLU()
        self.output.append(final_activ)
        final_pool = M.AvgPool2d(kernel_size=13, stride=1)
        self.output.append(final_pool)
        self.output = M.Sequential(*self.output)

    def forward(self, x):
        x = self.feature(x)
        x = self.output(x)
        x = x.reshape(x.shape[0], -1)
        return x


def get_squeezenet(version, residual=False, **kwargs):
    if version == '1.0':
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        maxpool_pad = [1, 0, 0]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == '1.1':
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        maxpool_pad = [1, 0, 0]
        init_block_kernel_size = 3
        init_block_channels = 64
    else:
        raise ValueError("Unsupported SqueezeNet version {}".format(version))

    if not residual:
        residuals = None

    net = SqueezeNet(
        channels=channels,
        residuals=residuals,
        init_block_kernel_size=init_block_kernel_size,
        init_block_channels=init_block_channels,
        maxpool_pad=maxpool_pad,
        **kwargs)

    return net


def squeezenet_v1_0(**kwargs):
    return get_squeezenet(version="1.0", residual=False, **kwargs)


def squeezenet_v1_1(**kwargs):
    return get_squeezenet(version="1.1", residual=False, **kwargs)


def squeezeresnet_v1_0(**kwargs):
    return get_squeezenet(version="1.0", residual=True, **kwargs)


def squeezeresnet_v1_1(**kwargs):
    return get_squeezenet(version="1.1", residual=True, **kwargs)


