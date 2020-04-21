
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import os
import cv2
from  torchsummary import summary
import numpy as np
import torchvision

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Branch(nn.Module):
    def __init__(self, in_channels,activation='soft'):
        super(Branch, self).__init__()

        self.conv=nn.Conv2d(in_channels,256,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(256)
        self.activation=nn.ReLU()
        self.dense=nn.Conv2d(256,1024,kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self,x,activation="soft"):
        x=self.conv(x)
        x=self.norm(x)
        x=self.activation(x)
        #x = x.view(x.size()[0], -1)
        x=self.dense(x)
        if activation=='relu':
            x=F.relu(x)
        else:
            x=F.softmax(x,dim=1)
        return x


class DL_model (nn.Module):

    def __init__(self, in_channels):
        super(DL_model, self).__init__()

        self.resnet=[]
        for i in range(8):
            self.resnet.append(torchvision.models.resnet.BasicBlock(in_channels,in_channels))
        self.resnet=nn.Sequential(*self.resnet)
        self.po_branch=Branch(1,activation='soft')
        self.value_branch=Branch(1,activation='relu')

    def forward(self,value,policy):
        value=self.resnet(value)
        policy=self.resnet(policy)
        policy=self.po_branch(policy)
        value=self.value_branch(value)
        return policy,value
