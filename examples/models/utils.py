#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: utils.py
   @Time: 2021-01-01 15:34:15
   @Version:1.0
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import megengine as mge
import megengine.module.init as init
import megengine.functional as F
import megengine.module as M
import math
from megengine.jit import trace
import numpy as np

__all__ = ["SEModule", "Droupout2d", "SplAtConv2d", "kaiming_norm_", "kaiming_uniform_", "split"]
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

class Dropout2d(M.Module):
    def __init__(self, drop_prob, kernel_size):
        '''
            Implementation of the 2d dropout.
            Args: 
                drop_prob (float): the drop out rate
                kernel_size (int): the kernel size of zhe pooling 
        '''
        super(Dropout2d, self).__init__()
        self.drop_prob = drop_prob
        self.kernel_size = kernel_size
    
    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        _, c, h, w = x.shape
        pad_h = max((self.kernel_size - 1), 0)
        pad_w = max((self.kernel_size - 1), 0)
        numel = c*h*w
        gamma = self.drop_prob * (w*h) / (self.kernel_size**2) / ((w - self.kernel_size + 1)*( h - self.kernel_size + 1))
        mask = mge.random.uniform(0, 1, size=(1, c, h, w))
        mask[mask<gamma] =  1
        mask[mask>=gamma] = 0
        mask = F.max_pool2d(mask, [self.kernel_size, self.kernel_size], stride=1, padding=(pad_h//2, pad_w//2))
        mask = 1 - mask
        x1=F.expand_dims(1.0*numel / mask.sum(axis=0), axis=0)
        y = F.matmul(F.matmul(x, mask), x1)
        return y
        

class SplAtConv2d(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                padding=0, dilation=1,  groups=1, bias=False, radix=2, reduction=4,
                norm_layer=None, dropblock_prob=0.0,**kwargs):
        '''
            Split Attention Conv2d.
            Args:
                in_channels (int): the number of  input channels
                out_channels (int): the number of output channels
                kernel_size (Union[int, Tuple[int, int]]): the kernel size of the conv
                stride (Union[int, Tuple[int, int]]): the stride of the conv
                padding (Union[int, Tuple[int, int]]): the padding size add to both side of the inputs
                dilation (Union[int, Tuple[int, int]]): the dilation rate
                groups (int): the number of groups for kernels
                bias (bool): whether use the bias
                radix (int): the branchs
                reduction (int): the reduction rate
                norm_layer (M.BatchNorm2d): the batch norm layer, if None, than do not use the norm in the module
        '''
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels*radix // reduction, 32)
        self.radix = radix
        self.cardinality = groups
        self.droupblock_prob = dropblock_prob
        self.conv = M.Conv2d(in_channels, out_channels*radix, kernel_size=kernel_size, stride=stride, 
                            padding=padding, dilation=dilation, groups=groups*radix, bias=bias, **kwargs)
        
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(radix*out_channels)
        
        self.relu = M.ReLU()
        #fc1
        self.fc1 = M.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        #fc2
        self.fc2 = M.Conv2d(inter_channels, out_channels*radix, 1, groups=self.cardinality)
        #drop out
        if dropblock_prob > 0.0:
            self.droupblock = Dropout2d(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, self.cardinality)
    
    def forward(self, x):
        #do the conv
        net = self.conv(x)
        if self.use_bn:
            net = self.bn0(net)
        
        if self.droupblock_prob > 0.0:
            net = self.droupblock(net)
            
        
        net = self.relu(net)
        #split from the channels
        batch = net.shape[0]

        if self.radix > 1:
            splited = F.split(net, self.radix , axis=1)
            gap = sum(splited)
        #calculate the attention
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).reshape(batch, -1, 1, 1)

        if self.radix > 1:
            attens = F.split(atten, self.radix, axis=1)
            
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        
        return out
        

class rSoftMax(M.Module):
    def __init__(self, radix, cardinality):
        '''
            rSoftmax.
            Args:
                radix (int):the radix index
                cardinality (int): the groups
        '''
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
    
    def forward(self, x):
        bs = x.shape[0]
        if self.radix > 1:
            x = x.reshape((bs, self.cardinality, self.radix, -1))
            x = F.softmax(x, axis=1)
            x = x.reshape(bs, -1)
        else:
            x = F.sigmoid(x)
        return x

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


    return splited
if __name__ == "__main__":
    import numpy as np
    dropout = Dropout2d(0.6, 3)
    
    numpy_x = np.random.random((1, 64, 64, 64))
    x = mge.tensor(numpy_x, dtype=np.float32)
    conv = SplAtConv2d(64, 128, 3, 2, padding=1)
    out = conv(x)