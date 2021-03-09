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
import numpy as np

__all__ = ["SupConLoss"]

class SupConLoss(M.Module):
    """
    Implementation of supervised contrastive loss.
    """
    def __init__(self, temperate=0.07, base_temperate=0.07, contrast_mode="all"):
        """
        The supervised contrastive loss from
        "Supervised Contrastive Learning"<https://arxiv.org/abs/2004.11362>
        Official Pytorch Implementation:https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        Args:
            temperate: The temperature in formula
            base_temperate: The basic temperature
            contrast_mode: The mode to calculate the loss, all features or one feature
        """
        super(SupConLoss, self).__init__()
        self.temperate = temperate
        self.base_temperate = base_temperate
        self.contrast_mode = contrast_mode

    def forward(self, features, label=None, mask=None):
        """
        if label and mask both None, the loss will degenerate to
        SimSLR unsupervised loss.
        Reference:
            "A Simple Framework for Contrastive Learning of Visual Representations"<https://arxiv.org/pdf/2002.05709.pdf>
            "Supervised Contrastive Learning"<https://arxiv.org/abs/2004.11362>
        Args:
            features(tensor): The embedding feature. shape=[bs, n_views, ...]
            label(tensor): The label of images, shape=[bs]
            mask(tensor): contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        return:
            loss
        """
        if len(features.shape) < 3:
            raise ValueError("Features need have 3 dimensions at least")
        bs, num_view = features.shape[:2]
        #if dimension > 3, change the shape of the features to [bs, num_view, ...]
        if len(features.shape) > 3:
            features = features.reshape(bs, num_view, -1)

        #label and mask cannot provided at the same time
        if (label is not None) and (mask is not None):
            raise ValueError("label and mask cannot provided at the same time")
        elif (label is None) and (mask is None):
            mask = F.eye(bs, dtype="float32")
        elif label is not None:
            label = label.reshape(-1, 1)
            if label.shape[0] != bs:
                raise RuntimeError("Num of labels does not match num of features")
            mask = F.equal(label, label.T)
        else:
            mask = mask.astype("float32")

        contrast_count = features.shape[1]
        features = F.split(features, features.shape[1], axis=1)
        contrast_feature = F.squeeze(F.concat(features, axis=0), axis=1)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode:{}".format(self.contrast_mode))
        #compute logits
        anchor_dot_contrast = F.div(
            F.matmul(anchor_feature, contrast_feature.T),
            self.temperate
        )

        #for numerical stability
        logits_max = F.max(anchor_dot_contrast, axis=-1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        #tile mask
        an1, con = mask.shape[:2]
        nums = anchor_count*contrast_count
        # mask-out self-contrast cases
        mask = F.stack([mask]*nums).reshape(an1*anchor_count, con*contrast_count)
        logits_mask =  F.scatter(
            F.ones_like(mask),
            1,
            F.arange(0, int(bs*anchor_count), dtype="int32").reshape(-1, 1),
            F.zeros(int(bs*anchor_count), dtype="int32").reshape(-1, 1))
        mask = mask * logits_mask
        #compute log_prob
        exp_logits = F.exp(logits) * logits_mask
        log_prob = logits - F.log(F.sum(exp_logits, axis=1, keepdims=True)) #equation 2

        #mean
        mean_log_prob_pos = F.sum(mask * log_prob, axis=1) / F.sum(mask, axis=1)

        #loss
        loss = - (self.temperate / self.base_temperate) * mean_log_prob_pos
        loss = F.mean(loss.reshape(anchor_count, bs))
        return loss

if __name__ == "__main__":
    np.random.seed(65536)
    feature = np.ones((10, 2, 128))
    y = np.random.randint(0, 10, 10)
    feature = mge.Tensor(feature)
    y = mge.Tensor(y)
    print(SupConLoss()(feature, y))


