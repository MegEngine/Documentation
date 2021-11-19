from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

cover_channel = (-128, 3, 64, 128, 256, 512, 512)
gen_in_channel = (512, 1024, 512, 256, 128)
gen_out_channel = (512, 256, 128, 64, 3)

class UNet(M.Module):
    def __init__(self, cc = cover_channel, gic = gen_in_channel, goc = gen_out_channel):
        '''
            A kind of UNet, which use full convolution instaed of pooling.
            args:
                cc: The input channel of each layer during convolution.
                gic: The input channel of each layer during deconvolution.
                goc: The output channel of each layer during deconvolution.
        '''
        super().__init__()
        self.cc = cc
        self.gic = gic
        self.goc = goc
        # the modules used in UNet
        self.conv1 = M.Conv2d(self.cc[1], self.cc[2], kernel_size=2, stride=2, padding=0)
        self.conv2 = M.Conv2d(self.cc[2], self.cc[3], kernel_size=2, stride=2, padding=0)
        self.conv3 = M.Conv2d(self.cc[3], self.cc[4], kernel_size=2, stride=2, padding=0)
        self.conv4 = M.Conv2d(self.cc[4], self.cc[5], kernel_size=2, stride=2, padding=0)
        self.conv5 = M.Conv2d(self.cc[5], self.cc[6], kernel_size=2, stride=2, padding=0)

        self.deconv1 = M.ConvTranspose2d(self.gic[0], self.goc[0], kernel_size=2, stride=2, padding=0)
        self.deconv2 = M.ConvTranspose2d(self.gic[1], self.goc[1], kernel_size=2, stride=2, padding=0)
        self.deconv3 = M.ConvTranspose2d(self.gic[2], self.goc[2], kernel_size=2, stride=2, padding=0)
        self.deconv4 = M.ConvTranspose2d(self.gic[3], self.goc[3], kernel_size=2, stride=2, padding=0)
        self.deconv5 = M.ConvTranspose2d(self.gic[4], self.goc[4], kernel_size=2, stride=2, padding=0)

        self.relu = M.ReLU()
        self.leaky_relu = M.LeakyReLU()
        self.sigmoid = M.Sigmoid()

        self.conv_bn2 = M.BatchNorm2d(self.cc[3])
        self.conv_bn3 = M.BatchNorm2d(self.cc[4])
        self.conv_bn4 = M.BatchNorm2d(self.cc[5])

        self.gen_bn1 = M.BatchNorm2d(self.goc[0])
        self.gen_bn2 = M.BatchNorm2d(self.goc[1])
        self.gen_bn3 = M.BatchNorm2d(self.goc[2])
        self.gen_bn4 = M.BatchNorm2d(self.goc[3])
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.leaky_relu(x1)
        x2 = self.conv2(x1)
        x2 = self.conv_bn2(x2)
        x2 = self.leaky_relu(x2)
        x3 = self.conv3(x2)
        x3 = self.conv_bn3(x3)
        x3 = self.leaky_relu(x3)
        x4 = self.conv4(x3)
        x4 = self.conv_bn4(x4)
        x4 = self.leaky_relu(x4)
        x5 = self.conv5(x4)
        x5 = self.relu(x5)

        g = self.deconv1(x5)
        g = self.gen_bn1(g)
        g = F.concat((x4, g), axis=1)
        g = self.relu(g)
        g = self.deconv2(g)
        g = self.gen_bn2(g)
        g = F.concat((x3, g), axis=1)
        g = self.relu(g)
        g = self.deconv3(g)
        g = self.gen_bn3(g)
        g = F.concat((x2, g), axis=1)
        g = self.relu(g)
        g = self.deconv4(g)
        g = self.gen_bn4(g)
        g = F.concat((x1, g), axis=1)
        g = self.relu(g)
        g = self.deconv5(g)
        g = self.sigmoid(g)

        return g

if __name__ == "__main__":
    net = UNet()
    x = mge.random.normal(size=[4, 3, 256, 256])
    net.eval()
    pred = net(x)
    print(pred.shape)
