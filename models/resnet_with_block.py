import torch.nn as nn
from .layer_blocks import SELayer, SRMLayer

__all__ = [
    'resnet20',
    'resnet32',
    'se_resnet20',
    'se_resnet32',
    'srm_resnet20',
    'srm_resnet32'
]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, layer_block=None,
                 reduction=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if layer_block is not None:
            self.layer_block = layer_block(planes, reduction)
        else:
            self.layer_block = None

        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.layer_block is not None:
            out = self.layer_block(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, layer_block=None,
                 reduction=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if layer_block is not None:
            self.layer_block = layer_block(planes * self.expansion, reduction)
        else:
            self.layer_block = None

        if inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes,
                                                      planes * self.expansion,
                                                      kernel_size=1,
                                                      stride=stride,
                                                      bias=False),
                                            nn.BatchNorm2d(
                                                planes * self.expansion))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.layer_block is not None:
            out = self.layer_block(out)

        out += residual
        out = self.relu(out)

        return out


class CifarResNetWithBlock(nn.Module):
    def __init__(self, n_size, num_classes=10, layer_block=None,
                 reduction=None):
        super(CifarResNetWithBlock, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, blocks=n_size, stride=1,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.layer2 = self._make_layer(32, blocks=n_size, stride=2,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.layer3 = self._make_layer(64, blocks=n_size, stride=2,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, layer_block, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.inplane, planes, stride, layer_block,
                                     reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetWithBlock(nn.Module):
    def __init__(self, layers, num_classes=1000, layer_block=None,
                 reduction=None):
        super(ResNetWithBlock, self).__init__()
        self.inplane = 64
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=layers[0], stride=1,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.layer2 = self._make_layer(128, blocks=layers[1], stride=2,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.layer3 = self._make_layer(256, blocks=layers[2], stride=2,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.layer4 = self._make_layer(512, blocks=layers[3], stride=2,
                                       layer_block=layer_block,
                                       reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, layer_block, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.inplane, planes, stride, layer_block,
                                     reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, **kwargs)
    return model


def cifar_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, **kwargs)
    return model


def cifar_se_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, layer_block=SELayer, reduction=16, **kwargs)
    return model


def cifar_se_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, layer_block=SELayer, reduction=16, **kwargs)
    return model


def cifar_srm_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, layer_block=SRMLayer, **kwargs)
    return model


def srm_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, layer_block=SRMLayer, **kwargs)
    return model


def resnet_34(**kwargs):
    model = ResNetWithBlock([3, 4, 6, 3], **kwargs)
    return model


def se_resnet_34(**kwargs):
    model = ResNetWithBlock([3, 4, 6, 3], layer_block=SELayer, **kwargs)
    return model


def srm_resnet_34(**kwargs):
    model = ResNetWithBlock([3, 4, 6, 3], layer_block=SRMLayer, **kwargs)
    return model