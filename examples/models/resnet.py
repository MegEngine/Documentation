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


import megengine as mge
import megengine.module as M
import megengine.functional as F

from utils import SEModule,SplAtConv2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
          'wide_resnet50_2', 'wide_resnet101_2', 'seresnet18', 'seresnet34',
          'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
          'seresnext101_32x8d', 'resnest50', 'resnest101', 'resnest200', 'resnest269'
]

model_urls = {
    'resnet18': '',
    'resnet34': '',
    'resnet50': '',
    'resnet101': '',
    'resnet152': '',
    'resnext50_32x4d': '',
    'resnext101_32x8d': '',
    'wide_resnet50_2': '',
    'wide_resnet101_2': '',
    'seresnet18' : '', 
    'seresnet34' : '',
    'seresnet50' : '', 
    'seresnet101' : '', 
    'seresnet152' : '',
    'seresnext50_32x4d' : '',
    'seresnext101_32x8d' : '',
    'resnest14' : '',
    'resnest26' : '',
    'resnest50' : '',
    'resnest101' : '',
    'resnest200' : '',
    'resnest269' : ''
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''
        Conv3x3 with padding
        Args:
            in_planes (int): the input channels of the conv layer
            out_planes (int): the number of channels of the outputs (or the number of kernels)
            stride (int or tuple or list): the stride of the conv
            dilation (int): the dilation rate of the conv
    '''
    return M.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, 
                        padding=dilation, groups=groups, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    '''
        1x1 convolutional layer
        Args:
            in_planes (int): the input channels of the conv layer
            out_planes (int): the number of channels of the outputs (or the number of kernels)
            stride (int or tuple or list): the stride of the conv
    '''
    return M.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False, padding=0)


class BasicBlock(M.Module):
    expansion = 1 #note that the expansion of basic block in resnet is 1
    
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, groups=1, downsample=None,
            base_width=64, norm_layer=None, se_module=None, radix=2, reduction=4, avd=False, avd_first=False):
        '''
            Implementation of the basic block.
            Args:
                inplanes (int): the number of channels of input
                outplanes (int): the number of channels of output (the number of kernels of conv layers)
                stride (int, tuple or list): the stride of the first conv3x3 layer
                dilation (int):the dilation rate of the first conv layer of the block
                groups (int): the number of groups for the first conv3x3 layer
                downsample (megendine.module.Module or None): if not None, will do the downsample for x
                base_width (int): the basic width of the layer
                norm_layer (None or megendine.module.Module): the normalization layer of the block, default is batch normalization 
                se_module (SEModule or None): the semodule from SENet
        '''
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supportes in BasicBlock")
        #self.downsample  and self.conv1 layer will do the downsample of the input  both when stride != 1
        #layer1
        self.conv1 = conv3x3(inplanes, outplanes, stride=stride, dilation=dilation, groups=groups)
        self.bn1 = norm_layer(outplanes)
        #activation layer
        self.relu = M.ReLU()
        #layer2
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        #downsample layer
        self.downsample = downsample
        #semodule
        self.se = se_module

        self.stride = stride
    
    def forward(self, x):
        identify = x
        
        net = self.relu(self.bn1(self.conv1(x)))
        net = self.bn2(self.conv2(net))
        if self.downsample is not None:
            identify = self.downsample(x)
        if self.se is not None:
            net = net * self.se(net)
        net = identify + net #residual
        net = self.relu(net)
        return net

class Bottleneck(M.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # the expansion of the bottleneck block in resnet is 4
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, groups=1, downsample=None,
            base_width=64, norm_layer=None, se_module=None, radix=2, reduction=4, avd=False, avd_first=False, is_first=False):
        '''
            Implementation of the basic block.
            Args:
                inplanes (int): the number of channels of input
                outplanes (int): the number of channels of output (the number of kernels of conv layers)
                stride (int, tuple or list): the stride of the second conv layer
                dilation (int):the dilation rate of the second conv layer of the block
                groups (int): the number of groups for the second conv layer
                downsample (megendine.module.Module or None): if not None, will do the downsample for x
                base_width (int): the basic width of the layer
                norm_layer (None or megendine.module.Module): the normalization layer of the block, default is batch normalization
                se_module (SEModule):  the Squeeze Excitation Module
                radix (int): the radix index
                reduction (int): the reduction factor
                avd (bool): whether use the avd layer
                avd_first (bool): whether use the avd layer befo conv2
                is_first (bool): whether is the first block of the stage 
        '''
        super(Bottleneck, self).__init__()
        width = int((base_width / 64) * outplanes) * groups
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = M.AvgPool2d(3, stride, padding=1)
            stride=1
        self.radix = radix
        #layer1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        #layer2
        if self.radix >= 1:
            self.conv2 = SplAtConv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, radix=radix, reduction=reduction)
        else:
            self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
            self.bn2 = norm_layer(width)
        #layer3
        self.conv3 = conv1x1(width, outplanes*self.expansion)
        self.bn3 = norm_layer(outplanes*self.expansion)

        #activation layer
        self.relu = M.ReLU()
        
        #downsample layer
        self.downsample = downsample
        #se module
        self.se = se_module

        #stride
        self.stride = stride
    
    def forward(self, x):
        identify = x

        #layer1 forward
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        if self.avd and self.avd_first:
            net = self.avd_layer(net)
        #layer2 forward
        if self.radix > 1:
            net = self.conv2(net)
        else:
            net = self.conv2(net)
            net = self.bn2(net)
            net = self.relu(net)

        if self.avd and not self.avd_first:
            net = self.avd_layer(net)
        #layer3 forward
        net = self.conv3(net)
        net = self.bn3(net)
        
        #if semodule
        if self.se is not None:
            net = self.se(net) * net
        #if need downsample
        if self.downsample is not None:
            identify = self.downsample(x)
        
        net = net + identify #residual
        net = self.relu(net)
        return net

def get_layers(num_layers):
    '''
        Get the number of blocks for each stage in resnet
        Args:
            num_layers (int): the number of layers for resnet 
        Reference:
            "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    blocks = []
    if num_layers == 14:
        blocks = [1, 1, 1, 1]
    elif num_layers == 18 or num_layers == 26:
        blocks = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
        blocks = [3, 4, 6, 3]
    elif num_layers == 101:
        blocks = [3, 4, 23, 3]
    elif num_layers == 152:
        blocks = [3, 4, 36, 3]
    elif num_layers == 200:
        blocks = [3, 24, 36, 3]
    elif num_layers == 269:
        blocks = [3, 30 ,48, 8]
    else:
        raise ValueError("Unknown number of layers {}".format(num_layers))
    return blocks

class ResNet(M.Module):
    def __init__(self,  block, blocks,in_ch=3, num_classes=1000, first_stride=2, light_head=False, zero_init_residual=False, 
            groups=1, width_per_group=64, strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1],multi_grids=[1, 1, 1], norm_layer=None, 
            se_module=None, reduction=16, radix=0, avd=False, avd_first=False, avg_layer=False, avg_down=False, stem_width=64):
        '''
            Modified resnet according to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
            Implementate  ResNet and the variation of ResNet.
            Args:
                in_ch: int, the number of channels of the input
                block: BasicBlock or Bottleneck.The block of the resnet

                num_classes: int, the number of classes to predict
                first_stride: int, the stride of the first conv layer
                light_head: boolean, whether use conv3x3 replace the conv7x7 in first conv layer
                zero_init_residual: whether initilize the residule block's batchnorm with zero
                groups: int, the number of groups for the conv in net
                width_per_group: int, the width of the conv layers
                strides: list, the list of the strides for the each stage
                dilations: list, the dilations of each block
                multi_grids: list, implementation of the multi grid layer in deeplabv3
                norm_layer: megengine.module.Module, the normalization layer, default is batch normalization
                se_module: SEModule, the Squeeze Excitation Module
                radix: int, the radix index from ResNest
                reduction: int, the reduction rate
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
                stem_width: int, the channels of the conv3x3 when use 3 conv3x3 replace conv7x7
            References:
                "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
                "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
                https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
                deeplab v3: https://arxiv.org/pdf/1706.05587.pdf
                deeplab v3+: https://arxiv.org/pdf/1802.02611.pdf
                "Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
                "ResNeSt: Split-Attention Networks"<https://arxiv.org/pdf/2004.08955.pdf>
        '''
        super(ResNet, self).__init__()

        if len(dilations) != 4 :
            raise ValueError("The length of dilations must be 4, but got {}".format(len(dilations)))
        
        if len(strides) != 4 :
            raise ValueError("The length of dilations must be 4, but got {}".format(len(strides)))
        
        if len(multi_grids) > blocks[-1]:
            multi_grids = multi_grids[:blocks[-1]]
        elif len(multi_grids) < blocks[-1]:
            raise ValueError("The length of multi_grids must greater than or equal the number of blocks for last stage , but got {}/{}".format(len(multi_grids), blocks[-1]))
        
        if norm_layer is None:
            norm_layer = M.BatchNorm2d

        self.base_width = width_per_group
        self.multi_grids = multi_grids
        self.inplanes = 64
        self.groups = groups
        self.norm_layer = norm_layer
        self.avg_layer = avg_layer
        self.avg_down = avg_down

        if light_head:
            self.conv1 = M.Sequential(
                conv3x3(in_ch, stem_width, stride=first_stride),
                norm_layer(stem_width),
                M.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                M.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = M.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=first_stride, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #4 stage
        self.layer1 = self._make_layer(block, 64, blocks[0], stride=strides[0], dilation=dilations[0], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer2 = self._make_layer(block, 128, blocks[1], stride=strides[1], dilation=dilations[1], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer3 = self._make_layer(block, 256, blocks[2], stride=strides[2], dilation=dilations[2], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer4 = self._make_grid_layer(block, 512, blocks[3], stride=strides[3], dilation=dilations[3], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)

        #classification part
        self.avgpool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, M.BatchNorm2d):
                M.init.fill_(m.weight, 1)
                M.init.zeros_(m.bias)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, se_module=None, reduction=16, radix=0, avd=False, avd_first=False):
        '''
            Implementation of the stage in resnet.
            Args:
                block: megengine.module.Module, the block module
                planes: int, the base channels
                blocks: int, the number of blocks for this stage
                stride: int, the stride for the first block in the stage
                dilation: int, the rate of the dilation(atrous)
                reduction: int, the reduction rate
                radix: int, the radix index from ResNest
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
        '''
        norm_layer = self.norm_layer
        downsample = None
        se = None
        if se_module is not None:
            se = se_module(planes*block.expansion, reduction, norm_layer=self.norm_layer)
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            down_stride = stride
            if self.avg_layer:
                if self.avg_down:
                    avg_layer = M.AvgPool2d(kernel_size=down_stride, stride=down_stride, padding=0)
                    down_stride = 1
                else:
                    avg_layer = M.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_layers += [conv1x1(self.inplanes, planes*block.expansion, down_stride), norm_layer(planes*block.expansion)]
            downsample = M.Sequential(*down_layers)      
        layers = []
        layers.append(block(self.inplanes, planes, groups=self.groups, downsample=downsample, stride=stride,
                            base_width=self.base_width, dilation=dilation, norm_layer=norm_layer, se_module=se, 
                            radix=radix, reduction=reduction, avd=avd, avd_first=avd_first, is_first=True))
        self.inplanes = planes * block.expansion
        if se_module is not None:
            se = se_module(self.inplanes, reduction, norm_layer=self.norm_layer)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                    dilation=dilation, norm_layer=norm_layer, se_module=se, reduction=reduction, avd=avd, avd_first=avd_first))
        return M.Sequential(*layers)
    
    def _make_grid_layer(self, block, planes, blocks, stride=1, dilation=1, se_module=None, reduction=16, radix=0, avd=False, avd_first=False):
        '''
            Implementation of the Multi-grid Method in deeplabv3
            Args:
                block: megengine.module.Module, the block module
                planes: int, the base channels
                blocks: int, the number of blocks for this stage
                stride: int, the stride for the first block in the stage
                dilation: int, the rate of the dilation(atrous)
                se_module: SEModule or None, the semodule from SENet
                reduction: int, the reduction rate
                radix: int, the radix index from ResNest
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
            Reference:
                "Rethinking Atrous Convolution for Semantic Image Segmentation"<https://arxiv.org/abs/1706.05587>
        '''
        norm_layer = self.norm_layer
        downsample = None
        se = None
        if se_module is not None:
            se = se_module(planes*block.expansion, reduction, norm_layer=self.norm_layer)
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_layer:
                if self.avg_down:
                    stride=1
                    avg_layer = M.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
                else:
                    avg_layer = M.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_layers += [conv1x1(self.inplanes, planes*block.expansion, stride), norm_layer(planes*block.expansion)]
            downsample = M.Sequential(*down_layers)      
        layers = []
        layers.append(block(self.inplanes, planes, groups=self.groups, stride=stride, downsample=downsample,
                            base_width=self.base_width, dilation=dilation*self.multi_grids[0], norm_layer=norm_layer, 
                            se_module=se, radix=radix, avd=avd, avd_first=avd_first))
        self.inplanes = planes * block.expansion
        if se_module is not None:
            se = se_module(self.inplanes, reduction, norm_layer=self.norm_layer)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                    dilation=dilation*self.multi_grids[i], norm_layer=norm_layer, se_module=se,
                    radix=radix, avd=avd, avd_first=avd_first))
        return M.Sequential(*layers)
    
    def _forward_impl(self, x):
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.maxpool(net)
        
        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)

        net = self.avgpool(net)
        net = F.flatten(net, 1)
        net = self.fc(net)
        return net
    
    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, blocks, pretrained=False, progress=True, **kwargs):
    '''
        Wraper to create the resnet.
        Args:
            arch (str):the arch to download the pretrained weights
            pretrained (bool): if True, download and load the pretrained weights
            progress (bool): if True, display the download progress
    '''
    model = ResNet(block, blocks,**kwargs)
    if pretrained:
        state_dict = mge.hub.load_serialized_obj_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)



def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_50_2",Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_101_2",Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def seresnet18(pretrained=False, progress=True, **kwargs):
    """SEResNet-18 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)

def seresnet34(pretrained=False, progress=True, **kwargs):
    """SEResNet-34 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)

def seresnet50(pretrained=False, progress=True, **kwargs):
    """SEResNet-50 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def seresnet101(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def seresnet152(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)

def seresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-50 32x4d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    kwargs["se_module"] = SEModule
    return _resnet("seresnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def seresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-101_32x8d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    kwargs["se_module"] = SEModule
    return _resnet("seresnext101_32x8d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnest14(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-14 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1]
    kwargs["light_head"] = True
    return _resnet("resnest14", Bottleneck, get_layers(269), pretrained, progress, **kwargs)

def resnest26(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-26 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 2
    kwargs["light_head"] = True
    return _resnet("resnest26", Bottleneck, get_layers(269), pretrained, progress, **kwargs)

def resnest50(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-50 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnest101(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-101 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def resnest200(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-200 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest200", Bottleneck, get_layers(200), pretrained, progress, **kwargs)

def resnest269(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-269 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 8
    kwargs["light_head"] = True
    return _resnet("resnest269", Bottleneck, get_layers(269), pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = resnest200()
    x = mge.random.normal(size=(1, 3, 224, 224))
    model.eval()
    pred = model(x)
    print(pred.shape)
=======
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


import megengine as mge
import megengine.module as M
import megengine.functional as F

from utils import SEModule,SplAtConv2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
          'wide_resnet50_2', 'wide_resnet101_2', 'seresnet18', 'seresnet34',
          'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50_32x4d',
          'seresnext101_32x8d', 'resnest50', 'resnest101', 'resnest200', 'resnest269'
]

model_urls = {
    'resnet18': '',
    'resnet34': '',
    'resnet50': '',
    'resnet101': '',
    'resnet152': '',
    'resnext50_32x4d': '',
    'resnext101_32x8d': '',
    'wide_resnet50_2': '',
    'wide_resnet101_2': '',
    'seresnet18' : '', 
    'seresnet34' : '',
    'seresnet50' : '', 
    'seresnet101' : '', 
    'seresnet152' : '',
    'seresnext50_32x4d' : '',
    'seresnext101_32x8d' : '',
    'resnest14' : '',
    'resnest26' : '',
    'resnest50' : '',
    'resnest101' : '',
    'resnest200' : '',
    'resnest269' : ''
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''
        Conv3x3 with padding
        Args:
            in_planes (int): the input channels of the conv layer
            out_planes (int): the number of channels of the outputs (or the number of kernels)
            stride (int or tuple or list): the stride of the conv
            dilation (int): the dilation rate of the conv
    '''
    return M.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, 
                        padding=dilation, groups=groups, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    '''
        1x1 convolutional layer
        Args:
            in_planes (int): the input channels of the conv layer
            out_planes (int): the number of channels of the outputs (or the number of kernels)
            stride (int or tuple or list): the stride of the conv
    '''
    return M.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False, padding=0)


class BasicBlock(M.Module):
    expansion = 1 #note that the expansion of basic block in resnet is 1
    
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, groups=1, downsample=None,
            base_width=64, norm_layer=None, se_module=None, radix=2, reduction=4, avd=False, avd_first=False):
        '''
            Implementation of the basic block.
            Args:
                inplanes (int): the number of channels of input
                outplanes (int): the number of channels of output (the number of kernels of conv layers)
                stride (int, tuple or list): the stride of the first conv3x3 layer
                dilation (int):the dilation rate of the first conv layer of the block
                groups (int): the number of groups for the first conv3x3 layer
                downsample (megendine.module.Module or None): if not None, will do the downsample for x
                base_width (int): the basic width of the layer
                norm_layer (None or megendine.module.Module): the normalization layer of the block, default is batch normalization 
                se_module (SEModule or None): the semodule from SENet
        '''
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supportes in BasicBlock")
        #self.downsample  and self.conv1 layer will do the downsample of the input  both when stride != 1
        #layer1
        self.conv1 = conv3x3(inplanes, outplanes, stride=stride, dilation=dilation, groups=groups)
        self.bn1 = norm_layer(outplanes)
        #activation layer
        self.relu = M.ReLU()
        #layer2
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = norm_layer(outplanes)
        #downsample layer
        self.downsample = downsample
        #semodule
        self.se = se_module

        self.stride = stride
    
    def forward(self, x):
        identify = x
        

        net = self.relu(self.bn1(self.conv1(x)))
        net = self.bn2(self.conv2(net))
        if self.downsample is not None:
            identify = self.downsample(x)
        
        net = identify + net #residual
        net = self.relu(net)
        return net

class Bottleneck(M.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # the expansion of the bottleneck block in resnet is 4
    def __init__(self, inplanes, outplanes, stride=1, dilation=1, groups=1, downsample=None,
            base_width=64, norm_layer=None, se_module=None, radix=2, reduction=4, avd=False, avd_first=False, is_first=False):
        '''
            Implementation of the basic block.
            Args:
                inplanes (int): the number of channels of input
                outplanes (int): the number of channels of output (the number of kernels of conv layers)
                stride (int, tuple or list): the stride of the second conv layer
                dilation (int):the dilation rate of the second conv layer of the block
                groups (int): the number of groups for the second conv layer
                downsample (megendine.module.Module or None): if not None, will do the downsample for x
                base_width (int): the basic width of the layer
                norm_layer (None or megendine.module.Module): the normalization layer of the block, default is batch normalization
                se_module (SEModule):  the Squeeze Excitation Module
                radix (int): the radix index
                reduction (int): the reduction factor
                avd (bool): whether use the avd layer
                avd_first (bool): whether use the avd layer befo conv2
                is_first (bool): whether is the first block of the stage 
        '''
        super(Bottleneck, self).__init__()
        width = int((base_width / 64) * outplanes) * groups
        if norm_layer is None:
            norm_layer = M.BatchNorm2d
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = M.AvgPool2d(3, stride, padding=1)
            stride=1
        self.radix = radix
        #layer1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        #layer2
        if self.radix >= 1:
            self.conv2 = SplAtConv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, radix=radix, reduction=reduction)
        else:
            self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
            self.bn2 = norm_layer(width)
        #layer3
        self.conv3 = conv1x1(width, outplanes*self.expansion)
        self.bn3 = norm_layer(outplanes*self.expansion)

        #activation layer
        self.relu = M.ReLU()
        
        #downsample layer
        self.downsample = downsample
        #se module
        self.se = se_module

        #stride
        self.stride = stride
    
    def forward(self, x):
        identify = x

        #layer1 forward
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        if self.avd and self.avd_first:
            net = self.avd_layer(net)
        #layer2 forward
        if self.radix > 1:
            net = self.conv2(net)
        else:
            net = self.conv2(net)
            net = self.bn2(net)
            net = self.relu(net)

        if self.avd and not self.avd_first:
            net = self.avd_layer(net)
        #layer3 forward
        net = self.conv3(net)
        net = self.bn3(net)
        
        #if semodule
        if self.se is not None:
            net = self.se(net)
        #if need downsample
        if self.downsample is not None:
            identify = self.downsample(x)
        
        net = net + identify #residual
        net = self.relu(net)
        return net

def get_layers(num_layers):
    '''
        Get the number of blocks for each stage in resnet
        Args:
            num_layers (int): the number of layers for resnet 
        Reference:
            "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    blocks = []
    if num_layers == 14:
        blocks = [1, 1, 1, 1]
    elif num_layers == 18 or num_layers == 26:
        blocks = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
        blocks = [3, 4, 6, 3]
    elif num_layers == 101:
        blocks = [3, 4, 23, 3]
    elif num_layers == 152:
        blocks = [3, 4, 36, 3]
    elif num_layers == 200:
        blocks = [3, 24, 36, 3]
    elif num_layers == 269:
        blocks = [3, 30 ,48, 8]
    else:
        raise ValueError("Unknown number of layers {}".format(num_layers))
    return blocks

class ResNet(M.Module):
    def __init__(self,  block, blocks,in_ch=3, num_classes=1000, first_stride=2, light_head=False, zero_init_residual=False, 
            groups=1, width_per_group=64, strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1],multi_grids=[1, 1, 1], norm_layer=None, 
            se_module=None, reduction=16, radix=0, avd=False, avd_first=False, avg_layer=False, avg_down=False, stem_width=64):
        '''
            Modified resnet according to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
            Implementate  ResNet and the variation of ResNet.
            Args:
                in_ch: int, the number of channels of the input
                block: BasicBlock or Bottleneck.The block of the resnet

                num_classes: int, the number of classes to predict
                first_stride: int, the stride of the first conv layer
                light_head: boolean, whether use conv3x3 replace the conv7x7 in first conv layer
                zero_init_residual: whether initilize the residule block's batchnorm with zero
                groups: int, the number of groups for the conv in net
                width_per_group: int, the width of the conv layers
                strides: list, the list of the strides for the each stage
                dilations: list, the dilations of each block
                multi_grids: list, implementation of the multi grid layer in deeplabv3
                norm_layer: megengine.module.Module, the normalization layer, default is batch normalization
                se_module: SEModule, the Squeeze Excitation Module
                radix: int, the radix index from ResNest
                reduction: int, the reduction rate
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
                stem_width: int, the channels of the conv3x3 when use 3 conv3x3 replace conv7x7
            References:
                "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
                "Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>
                https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
                deeplab v3: https://arxiv.org/pdf/1706.05587.pdf
                deeplab v3+: https://arxiv.org/pdf/1802.02611.pdf
                "Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
                "ResNeSt: Split-Attention Networks"<https://arxiv.org/pdf/2004.08955.pdf>
        '''
        super(ResNet, self).__init__()

        if len(dilations) != 4 :
            raise ValueError("The length of dilations must be 4, but got {}".format(len(dilations)))
        
        if len(strides) != 4 :
            raise ValueError("The length of dilations must be 4, but got {}".format(len(strides)))
        
        if len(multi_grids) > blocks[-1]:
            multi_grids = multi_grids[:blocks[-1]]
        elif len(multi_grids) < blocks[-1]:
            raise ValueError("The length of multi_grids must greater than or equal the number of blocks for last stage , but got {}/{}".format(len(multi_grids), blocks[-1]))
        
        if norm_layer is None:
            norm_layer = M.BatchNorm2d

        self.base_width = width_per_group
        self.multi_grids = multi_grids
        self.inplanes = 64
        self.groups = groups
        self.norm_layer = norm_layer
        self.avg_layer = avg_layer
        self.avg_down = avg_down

        if light_head:
            self.conv1 = M.Sequential(
                conv3x3(in_ch, stem_width, stride=first_stride),
                norm_layer(stem_width),
                M.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                M.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = M.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=first_stride, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #4 stage
        self.layer1 = self._make_layer(block, 64, blocks[0], stride=strides[0], dilation=dilations[0], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer2 = self._make_layer(block, 128, blocks[1], stride=strides[1], dilation=dilations[1], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer3 = self._make_layer(block, 256, blocks[2], stride=strides[2], dilation=dilations[2], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)
        self.layer4 = self._make_grid_layer(block, 512, blocks[3], stride=strides[3], dilation=dilations[3], se_module=se_module, reduction=reduction, radix=radix, avd=avd, avd_first=avd_first)

        #classification part
        self.avgpool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, M.BatchNorm2d):
                M.init.fill_(m.weight, 1)
                M.init.zeros_(m.bias)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    M.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    M.init.zeros_(m.bn2.weight)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, se_module=None, reduction=16, radix=0, avd=False, avd_first=False):
        '''
            Implementation of the stage in resnet.
            Args:
                block: megengine.module.Module, the block module
                planes: int, the base channels
                blocks: int, the number of blocks for this stage
                stride: int, the stride for the first block in the stage
                dilation: int, the rate of the dilation(atrous)
                reduction: int, the reduction rate
                radix: int, the radix index from ResNest
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
        '''
        norm_layer = self.norm_layer
        downsample = None
        se = None
        if se_module is not None:
            se = se_module(planes*block.expansion, reduction, norm_layer=self.norm_layer)
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            down_stride = stride
            if self.avg_layer:
                if self.avg_down:
                    avg_layer = M.AvgPool2d(kernel_size=down_stride, stride=down_stride, padding=0)
                    down_stride = 1
                else:
                    avg_layer = M.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_layers += [conv1x1(self.inplanes, planes*block.expansion, down_stride), norm_layer(planes*block.expansion)]
            downsample = M.Sequential(*down_layers)      
        layers = []
        layers.append(block(self.inplanes, planes, groups=self.groups, downsample=downsample, stride=stride,
                            base_width=self.base_width, dilation=dilation, norm_layer=norm_layer, se_module=se, 
                            radix=radix, reduction=reduction, avd=avd, avd_first=avd_first, is_first=True))
        self.inplanes = planes * block.expansion
        if se_module is not None:
            se = se_module(self.inplanes, reduction, norm_layer=self.norm_layer)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                    dilation=dilation, norm_layer=norm_layer, se_module=se, reduction=reduction, avd=avd, avd_first=avd_first))
        return M.Sequential(*layers)
    
    def _make_grid_layer(self, block, planes, blocks, stride=1, dilation=1, se_module=None, reduction=16, radix=0, avd=False, avd_first=False):
        '''
            Implementation of the Multi-grid Method in deeplabv3
            Args:
                block: megengine.module.Module, the block module
                planes: int, the base channels
                blocks: int, the number of blocks for this stage
                stride: int, the stride for the first block in the stage
                dilation: int, the rate of the dilation(atrous)
                se_module: SEModule or None, the semodule from SENet
                reduction: int, the reduction rate
                radix: int, the radix index from ResNest
                avd: bool, whether use the avd layer
                avd_first: bool, whether use the avd layer before bottleblock's conv2
            Reference:
                "Rethinking Atrous Convolution for Semantic Image Segmentation"<https://arxiv.org/abs/1706.05587>
        '''
        norm_layer = self.norm_layer
        downsample = None
        se = None
        if se_module is not None:
            se = se_module(planes*block.expansion, reduction, norm_layer=self.norm_layer)
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_layer:
                if self.avg_down:
                    stride=1
                    avg_layer = M.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
                else:
                    avg_layer = M.AvgPool2d(kernel_size=1, stride=1, padding=0)
                down_layers.append(avg_layer)
            down_layers += [conv1x1(self.inplanes, planes*block.expansion, stride), norm_layer(planes*block.expansion)]
            downsample = M.Sequential(*down_layers)      
        layers = []
        layers.append(block(self.inplanes, planes, groups=self.groups, stride=stride, downsample=downsample,
                            base_width=self.base_width, dilation=dilation*self.multi_grids[0], norm_layer=norm_layer, 
                            se_module=se, radix=radix, avd=avd, avd_first=avd_first))
        self.inplanes = planes * block.expansion
        if se_module is not None:
            se = se_module(self.inplanes, reduction, norm_layer=self.norm_layer)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                    dilation=dilation*self.multi_grids[i], norm_layer=norm_layer, se_module=se,
                    radix=radix, avd=avd, avd_first=avd_first))
        return M.Sequential(*layers)
    
    def _forward_impl(self, x):
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.maxpool(net)
        
        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)

        net = self.avgpool(net)
        net = F.flatten(net, 1)
        net = self.fc(net)
        return net
    
    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, blocks, pretrained=False, progress=True, **kwargs):
    '''
        Wraper to create the resnet.
        Args:
            arch (str):the arch to download the pretrained weights
            pretrained (bool): if True, download and load the pretrained weights
            progress (bool): if True, display the download progress
    '''
    model = ResNet(block, blocks,**kwargs)
    if pretrained:
        state_dict = mge.hub.load_serialized_obj_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)



def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
   
    return _resnet("resnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_50_2",Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """
    Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet_101_2",Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def seresnet18(pretrained=False, progress=True, **kwargs):
    """SEResNet-18 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet18", BasicBlock, get_layers(18), pretrained, progress, **kwargs)

def seresnet34(pretrained=False, progress=True, **kwargs):
    """SEResNet-34 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet34", BasicBlock, get_layers(34), pretrained, progress, **kwargs)

def seresnet50(pretrained=False, progress=True, **kwargs):
    """SEResNet-50 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def seresnet101(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def seresnet152(pretrained=False, progress=True, **kwargs):
    """SEResNet-101 model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["se_module"] = SEModule
    return _resnet("seresnet152", Bottleneck, get_layers(152), pretrained, progress, **kwargs)

def seresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-50 32x4d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    kwargs["se_module"] = SEModule
    return _resnet("seresnext50_32x4d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def seresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""
    SE-ResNeXt-101_32x8d model from
    `"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    kwargs["se_module"] = SEModule
    return _resnet("seresnext101_32x8d", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnest14(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-14 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1]
    kwargs["light_head"] = True
    return _resnet("resnest14", Bottleneck, get_layers(14), pretrained, progress, **kwargs)

def resnest26(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-26 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 2
    kwargs["light_head"] = True
    return _resnet("resnest26", Bottleneck, get_layers(26), pretrained, progress, **kwargs)

def resnest50(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-50 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest50", Bottleneck, get_layers(50), pretrained, progress, **kwargs)

def resnest101(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-101 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest101", Bottleneck, get_layers(101), pretrained, progress, **kwargs)

def resnest200(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-200 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["light_head"] = True
    return _resnet("resnest200", Bottleneck, get_layers(200), pretrained, progress, **kwargs)

def resnest269(pretrained=False, progress=True, **kwargs):
    r"""
    ReNeSt-269 model from
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/pdf/2004.08955.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["radix"] = 2
    kwargs["avd"] = True
    kwargs["avg_layer"] = True
    kwargs["avg_down"] = True
    kwargs["multi_grids"] = [1] * 8
    kwargs["light_head"] = True
    return _resnet("resnest269", Bottleneck, get_layers(269), pretrained, progress, **kwargs)


if __name__ == "__main__":
    model = resnest200()
    x = mge.random.normal(size=(1, 3, 224, 224))
    model.eval()
    pred = model(x)
    print(pred.shape)