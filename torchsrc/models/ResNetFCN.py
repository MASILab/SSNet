import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

import math


class GCN(nn.Module):
    def __init__(self, inplanes, planes):
        super(GCN, self).__init__()
        # self.bn = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.conv1(x)
        return x

#
# class Refine(nn.Module):
#     def __init__(self, planes):
#         super(Refine, self).__init__()
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         residual = x
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#
#         out = residual + x
#         return out


class ResNetFCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFCN, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.num_classes)
        self.gcn2 = GCN(1024, self.num_classes)
        self.gcn3 = GCN(512, self.num_classes)
        self.gcn4 = GCN(64, self.num_classes)
        self.gcn5 = GCN(64, self.num_classes)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.gcn1(fm4)
        gcfm2 = self.gcn2(fm3)
        gcfm3 = self.gcn3(fm2)
        gcfm4 = self.gcn4(pool_x)
        gcfm5 = self.gcn5(conv_x)

        fs1 = F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2
        fs2 = F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3
        fs3 = F.upsample_bilinear(fs2, pool_x.size()[2:]) + gcfm4
        fs4 = F.upsample_bilinear(fs3, conv_x.size()[2:]) + gcfm5
        out = F.upsample_bilinear(fs4, input.size()[2:])

        return out

        # return out, fs4, fs3, fs2, fs1, gcfm1