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

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    "alexnet":""
}

class AlexNet(M.Module):
    def __init__(self, in_ch=3, num_classes=1000):
        '''
            The AlexNet.
            args:
                in_ch: int, the number of channels of inputs
                num_classes: int, the number of classes that need to predict
            reference:
                "One weird trick for parallelizing convolutional neural networks"<https://arxiv.org/abs/1404.5997>
        '''
        super(AlexNet, self).__init__()
        #the part to extract feature
        self.features = M.Sequential(
            M.Conv2d(in_ch, 64, kernel_size=11, stride=4, padding=11//4),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
            M.Conv2d(64, 192, kernel_size=5, padding=2),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
            M.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
        )
        #global avg pooling
        self.avgpool = M.AdaptiveAvgPool2d((6,6))
        #classify part
        self.classifier = M.Sequential(
            M.Dropout(),
            M.Linear(256*6*6, 4096),
            M.ReLU(),
            M.Dropout(),
            M.Linear(4096, 4096),
            M.ReLU(),
            M.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = mge.functional.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, progress=True, **kwargs):
    """
        AlexNet model.
        Args:
            pretrained (bool):If True, returns a model pre-trained on ImageNet
            progress (bool): If True, display a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = mge.hub.load_serialized_obj_from_url(model_urls[arch])
        model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    net = alexnet(in_ch=3)
    x = mge.random.normal(size=[1, 3, 224, 224])
    net.eval()
    pred = net(x)
    print(pred.shape)
