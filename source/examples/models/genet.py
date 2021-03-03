#-*- coding:utf-8 -*-
#!/etc/env python
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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F
from collections import OrderedDict

__all__ = ["GENet", "genet_small", "genet_normal", "genet_large"]

class XXBlock(M.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        super(XXBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if activation is None:
            activation = M.ReLU()
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = M.Sequential(
            M.Conv2d(in_ch, expansion_out_ch,ksize, stride=stride, padding=ksize // 2),
            norm_layer(expansion_out_ch),
            activation,
            M.Conv2d(expansion_out_ch, out_ch, ksize,stride=1, padding=(ksize-1) // 2, bias=bias),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = M.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                self.shortcut = M.Sequential(
                    M.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = M.Sequential(
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )


    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class BottleBlock(M.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        super(BottleBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if activation is None:
            activation = M.ReLU()
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = M.Sequential(
            M.Conv2d(in_ch, expansion_out_ch, 1, stride=1, padding=0),
            norm_layer(expansion_out_ch),
            activation,
            M.Conv2d(expansion_out_ch, expansion_out_ch, ksize, stride=stride, padding=(ksize-1) // 2, bias=bias),
            norm_layer(expansion_out_ch),
            activation,
            M.Conv2d(expansion_out_ch, out_ch, 1, stride=1, padding=0),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = M.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                self.shortcut = M.Sequential(
                    M.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = M.Sequential(
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )


    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class DwBlock(M.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        super(DwBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if activation is None:
            activation = M.ReLU()
        expansion_out_ch = round(out_ch*expansion)
        self.conv_block = M.Sequential(
            M.Conv2d(in_ch, expansion_out_ch, 1, stride=1, padding=0),
            norm_layer(expansion_out_ch),
            activation,
            M.Conv2d(expansion_out_ch, expansion_out_ch, ksize,stride=stride, padding=ksize // 2, bias=bias),
            norm_layer(expansion_out_ch),
            activation,
            M.Conv2d(expansion_out_ch, out_ch, 1, stride=1, padding=0),
            norm_layer(out_ch)
        )
        self.activation = activation
        self.shortcut = M.Sequential()
        if stride > 1 or in_ch != out_ch:
            if stride > 1:
                #official code not use avg downsample, we add it in there
                self.shortcut = M.Sequential(
                    M.AvgPool2d(kernel_size=stride+1, stride=stride, padding=stride//2),
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )
            else:
                self.shortcut = M.Sequential(
                    M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias),
                    norm_layer(out_ch)
                )

    def forward(self, x):
        identify = x
        net = self.conv_block(x)
        net = net + self.shortcut(identify)
        net = self.activation(net)
        return net

class ConvBlock(M.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, expansion=1.0, bias=False,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        super(ConvBlock, self).__init__()
        assert (expansion - 1) < 1e-6, ValueError("The expansion of the conv block caMot greater than 1")
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if activation is None:
            norm_layer = M.ReLU()
        expansion_out = int(round(out_ch*expansion))
        self.conv_block = M.Sequential(
            M.Conv2d(in_ch, expansion_out, ksize, stride=stride, bias=bias),
            norm_layer(expansion_out),
            activation
        )

    def forward(self, x):
        net = self.conv_block(x)
        return net

class GENet(M.Module):
    def __init__(self, cfg, in_chs=3, num_classes=1000, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), features_only=False):
        super(GENet, self).__init__()
        self.current_chaMels = in_chs
        features = OrderedDict()
        for i in range(len(cfg)):
            features[f"layer_{i}"] = self._make_layer(cfg[i], norm_layer, activation)
        self.features = M.Sequential(features)
        self.global_avg = M.AdaptiveAvgPool2d(1)
        self.classifier = M.Linear(self.current_chaMels, num_classes)
        self.features_only = features_only

    def _make_layer(self, cfg, norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        block_type = cfg[0]
        block = self._get_block(block_type)
        sub_layers = cfg[1]
        out_chaMels = cfg[2]
        current_stride = cfg[3]
        ksize = cfg[4]
        expansion = cfg[5]
        layers = []
        for i in range(sub_layers):
            current_stride = current_stride if i < 1 else 1
            layers.append(block(self.current_chaMels, out_chaMels, ksize, stride=current_stride,
                                expansion=expansion, norm_layer=norm_layer, activation=activation))
            self.current_chaMels = out_chaMels
        return M.Sequential(*layers)

    def _get_block(self, block_type):
        if block_type.lower() == "conv":
            return ConvBlock
        elif block_type.lower() == "xx":
            return XXBlock
        elif block_type.lower() == "dw":
            return DwBlock
        elif block_type.lower() == "bl":
            return BottleBlock

    def forward(self, x):
        net = self.features(x)
        if self.features_only:
            return net
        net = self.global_avg(net)
        net = F.flatten(net)
        net = self.classifier(net)
        return net



cfg = {
    "genet_small":[
        ["conv", 1, 13, 2, 3, 1],
        ["xx",1, 48, 2, 3, 1],
        ["xx", 3, 48, 2, 3, 1],
        ["bl",7, 384, 2, 3, 0.25],
        ["dw", 2, 560, 2, 3, 3],
        ["dw", 1, 256, 1, 3, 3],
        ["conv", 1, 1920, 1, 1, 1]
        ],
    "genet_normal":[
        ["conv", 1, 32, 2, 3, 1],
        ["xx",1, 128 , 2, 3, 1],
        ["xx", 2, 192, 2, 3, 1],
        ["bl", 6, 640, 2, 3, 0.25],
        ["dw", 4, 640, 2, 3, 3],
        ["dw", 1, 640, 1, 3, 3],
        ["conv", 1, 2560, 1, 1, 1]
        ],
    "genet_large":[
        ["conv", 1, 32, 2, 3, 1],
        ["xx",1, 128, 2, 3, 1],
        ["xx", 2, 192, 2, 3, 1],
        ["bl",6, 640, 2, 3, 0.25],
        ["dw", 5, 640, 2, 3, 3],
        ["dw", 4, 640, 1, 3, 3],
        ["conv", 1, 2560, 1, 1, 1]
        ]
}


def get_cfg(model_name):
    return cfg[model_name]


def _create_model(model_name, **kwargs):
    cfg = get_cfg(model_name)
    model = GENet(cfg, **kwargs)
    return model

def genet_small(**kwargs):
    return _create_model("genet_small", **kwargs)

def genet_normal(**kwargs):
    return _create_model("genet_normal", **kwargs)

def genet_large(**kwargs):
    return _create_model("genet_large", **kwargs)

if __name__ == "__main__":
    x = mge.random.normal(0, 1, (1, 3, 224, 224))
    model = genet_large(features_only=True)
    out = model(x)
    print(out.shape)