from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


class CNN_PB(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=10, s=2, encs=None, args=None):
        super(CNN_PB, self).__init__()
        self.args = args
        self.scale =  args.scale#64*8
        fc_inner = args.fc_inner
        num_classes = args.num_classes

        self.enc_0 = encs[0]

        self.scale = s
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128 * 3 * 3, fc_inner)
        self.dce = dce_loss(num_classes, fc_inner)

    def forward(self, x, return_features=False):
        x = self.enc_0(x)
        # print(x.shape)
        xf = x.view(-1, 128 * 3 * 3)


        x1 = self.preluip1(self.ip1(xf))
        centers, x = self.dce(x1)
        # output = F.log_softmax(self.scale * x, dim=1)
        # return x1, centers, x, output
        if return_features:
            return self.scale * x, xf
        return self.scale * x

class CNN_PB_part(nn.Module):
    def __init__(self, encs=None, args=None):
        super().__init__()
        self.args = args

        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        return x

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance
