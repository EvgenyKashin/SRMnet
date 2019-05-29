import torch.nn as nn
from .layer_blocks import SELayer, SRMLayer
from torchvision.models import ResNet


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


def basic_block_factory(layer_block=None):
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     reduction=None):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

            if layer_block is not None:
                self.layer_block = layer_block(planes, reduction)
            else:
                self.layer_block = None

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.layer_block is not None:
                out = self.layer_block(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    return BasicBlock


def bottleneck_factory(layer_block=None):
    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None,
                     reduction=16):
            super(Bottleneck, self).__init__()
            self.conv1 = conv1x1(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes, stride=stride)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = conv1x1(planes, planes * self.expansion)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

            if layer_block is not None:
                self.layer_block = layer_block(planes * self.expansion,
                                               reduction)
            else:
                self.layer_block = None

        def forward(self, x):
            residual = x
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

            if self.downsample:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    return Bottleneck


class CifarResNetWithBlock(nn.Module):
    def __init__(self, n_size, num_classes=10, layer_block=None,
                 reduction=None):
        super(CifarResNetWithBlock, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.block = basic_block_factory(layer_block=layer_block)
        self.layer1 = self._make_layer(16, blocks=n_size, stride=1,
                                       reduction=reduction)
        self.layer2 = self._make_layer(32, blocks=n_size, stride=2,
                                       reduction=reduction)
        self.layer3 = self._make_layer(64, blocks=n_size, stride=2,
                                       reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, reduction):
        downsample = None
        strides = [stride] + [1] * (blocks - 1)
        layers = []

        for stride in strides:
            if self.inplanes != planes:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes,
                                                   stride=stride),
                                           nn.BatchNorm2d(planes))

            layers.append(self.block(self.inplanes, planes, stride,
                                     downsample, reduction))
            self.inplanes = planes

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


# ImageNet ResNets34
def resnet34(num_classes=1000):
    model = ResNet(basic_block_factory(), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1000):
    model = ResNet(basic_block_factory(layer_block=SELayer), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def srm_resnet34(num_classes=1000):
    model = ResNet(basic_block_factory(layer_block=SRMLayer), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# ImageNet ResNets50
def resnet50(num_classes=1000):
    model = ResNet(bottleneck_factory(), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1000):
    model = ResNet(bottleneck_factory(layer_block=SELayer), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def srm_resnet50(num_classes=1000):
    model = ResNet(bottleneck_factory(layer_block=SRMLayer), [3, 4, 6, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# ImageNet ResNets101
def resnet101(num_classes=1000):
    model = ResNet(bottleneck_factory(), [3, 4, 23, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes=1000):
    model = ResNet(bottleneck_factory(layer_block=SELayer), [3, 4, 23, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def srm_resnet101(num_classes=1000):
    model = ResNet(bottleneck_factory(layer_block=SRMLayer), [3, 4, 23, 3],
                   num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# Cifar10 ResNets32
def cifar_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, **kwargs)
    return model


def cifar_se_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, layer_block=SELayer, reduction=16, **kwargs)
    return model


def cifar_srm_resnet20(**kwargs):
    model = CifarResNetWithBlock(3, layer_block=SRMLayer, **kwargs)
    return model


# Cifar10 ResNets32
def cifar_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, **kwargs)
    return model


def cifar_se_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, layer_block=SELayer, reduction=16,
                                 **kwargs)
    return model


def cifar_srm_resnet32(**kwargs):
    model = CifarResNetWithBlock(5, layer_block=SRMLayer, reduction=16,
                                 **kwargs)
    return model

