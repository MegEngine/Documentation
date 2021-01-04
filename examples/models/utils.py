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
from __future__ import division
from __future__ import absolute_import

import megengine.module.init as init
import megengine.module as M
import math

class SEModule(M.Module):
    '''
        Implementation of semodule in SENet and MobileNetV3, there we use 1x1 conv replace the linear layer.
        SENet:"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
        MobileNetV3: "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>
    '''
    def __init__(self, in_ch, reduction=16,  norm_layer=None, nolinear=M.ReLU(), sigmoid=M.Sigmoid()):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of input,
            @reduction: int, the coefficient of dimensionality reduction
            @sigmoid: M.Module, the sigmoid function, in MobilenetV3 is H-Sigmoid and in SeNet is sigmoid
            @norm_layer: M.Module, the batch normalization moldule
            @nolinear: M.Module, the nolinear function module
            @sigmoid: M.Module, the sigmoid layer
        '''
        super(SEModule, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d

        if nolinear is None:
            nolinear = M.ReLU()
        
        if sigmoid is None:
            sigmoid = M.Sigmoid()

        self.avgpool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0),
            norm_layer(in_ch // reduction),
            nolinear,
            M.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0),
            norm_layer(in_ch),
            sigmoid,
        )

    def forward(self, x):
        net = self.avgpool(x)
        net = self.fc(net)
        return net

def kaiming_norm_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
        Kaiming initilization.
        Args:
            tensor: an n-dimensional megengine.Tensor
            a: the negtive slope
            mode: either 'fan_in' or 'fan_out'
            nonlinearity: the non-linear function (the name of the non-linear function)
    '''
    fan = init.calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return init.normal_(tensor, 0, std)

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
        Kaiming initilization.
        Args:
            tensor: an n-dimensional megengine.Tensor
            a: the negtive slope
            mode: either 'fan_in' or 'fan_out'
            nonlinearity: the non-linear function (the name of the non-linear function)
    '''
    fan = init.calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) / math.sqrt(fan)
    return init.uniform_(tensor, -bound, bound) 
