# From https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/
import torch.nn as nn
import torch

import architecture.xdlayers as xd

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
}
import numpy as np

def emulate_conv_with_k3(image, dims):
    if dims == 1:
        return image


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, bn_momentum=0.5, dims=3, no_bn=False):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.bn_momentum = bn_momentum
        self.conv1 = xd.Conv(dims, inplanes, planes, kernel_size=1, bias=no_bn)
        self.bn1 = xd.BatchNorm(dims, planes, momentum=self.bn_momentum, no_bn=no_bn)
        self.conv2 = xd.Conv(dims, planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=no_bn) # changed padding from (kernel_size - 1) // 2
        self.bn2 = xd.BatchNorm(dims, planes, momentum=self.bn_momentum, no_bn=no_bn)
        self.conv3 = xd.Conv(dims, planes, planes * 4, kernel_size=1, bias=no_bn)
        self.bn3 = xd.BatchNorm(dims, planes * 4, momentum=self.bn_momentum, no_bn=no_bn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def add_residual(self, out, residual):
        diffs = [r-o for r,o in zip(residual.shape[2:], out.shape[2:])]
        if any([d!=0 for d in diffs]):
            slices = (slice(None),)*2 + tuple([slice(None,-diff) for diff in diffs])
            residual = residual[slices]
        out += residual
        return out

    def forward(self, x, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add_residual(out, residual)
        out = self.relu(out)
        return out


class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], bn_momentum=0.5, num_classes=1,
                 avg_pool=True, dims=3, no_bn=False, base_channels=64,
                 input_channels=3, nr_outputs=None):
        super(BagNet, self).__init__()
        if nr_outputs is None:
            nr_outputs = [2]

        self.inplanes = base_channels
        self.no_bn = no_bn
        self.bn_momentum = bn_momentum
        self.dims = dims
        self.conv1 = xd.Conv(dims, input_channels, base_channels, kernel_size=1, stride=1, padding=0,
                               bias=self.no_bn)
        self.conv2 = xd.Conv(dims, base_channels, base_channels, kernel_size=3, stride=1, padding=0,
                               bias=self.no_bn)
        self.bn1 = xd.BatchNorm(dims, base_channels, momentum=self.bn_momentum, no_bn=self.no_bn)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avg_pool = avg_pool
        self.block = block

        linear_list = [xd.Conv(dims, base_channels*32, nr, kernel_size=1, stride=1, padding=0) for nr in nr_outputs]
        self.linear_list = torch.nn.ModuleList(linear_list)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                if self.no_bn:
                    m.weight.data.normal_(0, 1)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                xd.EmulateNonPaddedConvWithK3(self.dims, kernel3),
                xd.Conv(self.dims, self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=self.no_bn),
                xd.BatchNorm(self.dims, planes * block.expansion, momentum=self.bn_momentum, no_bn=self.no_bn),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel, bn_momentum=self.bn_momentum, dims=self.dims, no_bn=self.no_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel, bn_momentum=self.bn_momentum, dims=self.dims, no_bn=self.no_bn))

        return nn.Sequential(*layers)

    def forward(self, x):
        # pad x to min shape
        minshape = np.asarray([33, 33, 33])
        imshape = x.shape[2:]
        to_pad = np.repeat(minshape - imshape, 2)[::-1]
        to_pad[to_pad < 0] = 0
        nn.functional.pad(x, to_pad, "constant", 0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.avg_pool:
            x = xd.AvgPool(self.dims, x.size()[2:2+self.dims], stride=1)(x)
        x = [linear(x) for linear in self.linear_list]
        return x


def bagnet33(strides=[2, 2, 2, 1], layers=[3, 4, 6, 3], dims=3, **kwargs):
    model = BagNet(Bottleneck, layers, strides=strides, kernel3=[1,1,1,1], dims=dims, **kwargs)
    return model


def bagnet17(strides=[2, 2, 2, 1], layers=[3, 4, 6, 3], dims=3, **kwargs):
    model = BagNet(Bottleneck, layers, strides=strides, kernel3=[1,1,1,0], dims=dims, **kwargs)
    return model


def bagnet9(strides=[2, 2, 2, 1], layers=[3, 4, 6, 3], dims=3, **kwargs):
    model = BagNet(Bottleneck, layers, strides=strides, kernel3=[1,1,0,0], dims=dims, **kwargs)
    return model