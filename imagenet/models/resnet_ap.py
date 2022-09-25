# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch.nn.functional as F
import torch.nn as nn
import math


def conv_stride1(in_planes, out_planes, kernel_size=3):
    "3x3 convolution with padding"
    layer = nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False)
    return layer


def normalization(inplanes, norm_type):
    if norm_type == 'batch':
        bn = nn.BatchNorm2d(inplanes)
    elif norm_type == 'instance':
        bn = nn.GroupNorm(inplanes, inplanes)
    else:
        raise AssertionError(f"Check normalization type! {norm_type}")
    return bn


class IntroBlock(nn.Module):

    def __init__(self, planes, norm_type):
        super(IntroBlock, self).__init__()
        self.conv1 = conv_stride1(3, planes, kernel_size=7)
        self.bn1 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_type='instance', stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_stride1(inplanes, planes, kernel_size=3)  # Modification
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = conv_stride1(planes, planes, kernel_size=3)
        self.bn2 = normalization(planes, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_type='instance', stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)  # modification
        self.bn2 = normalization(planes, norm_type)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * Bottleneck.expansion, norm_type)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.stride != 1:  # modification
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAP(nn.Module):

    def __init__(self, depth, num_classes=1000, norm_type='instance'):
        super(ResNetAP, self).__init__()
        self.norm_type = norm_type

        # print(f"ResNetAP-{depth} norm: {self.norm_type}")
        blocks = {
            10: BasicBlock,
            18: BasicBlock,
            34: BasicBlock,
            50: Bottleneck,
            101: Bottleneck,
            152: Bottleneck,
            200: Bottleneck
        }
        layers = {
            10: [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
            200: [3, 24, 36, 3]
        }
        assert layers[depth], 'invalid detph for ResNet'

        self.inplanes = 64
        self.layer0 = IntroBlock(self.inplanes, norm_type)
        nc = self.inplanes
        self.layer1 = self._make_layer(blocks[depth], nc, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], nc * 2, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], nc * 4, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], nc * 8, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_stride1(self.inplanes, planes * block.expansion, kernel_size=1),
                nn.AvgPool2d(kernel_size=stride, stride=stride),
                normalization(planes * block.expansion, self.norm_type),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  norm_type=self.norm_type,
                  stride=stride,
                  downsample=downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.shape[-1])
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
