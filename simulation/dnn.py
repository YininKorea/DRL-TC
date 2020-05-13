import torch
import torch.nn as nn
import torchvision


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels, activation='soft'):
        super(Branch, self).__init__()

        self.conv=nn.Conv2d(in_channels,256,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(256)
        if activation=='relu'
        self.activation=nn.ReLU()
        self.dense=nn.Linear(400, out_channels)

    def forward(self,x,activation="soft"):
        x=self.conv(x)
        x=self.norm(x)
        x=self.activation(x)
        x=torch.flatten(x, 1)
        x=self.dense(x)
        if activation=='relu':
            x=F.relu(x)
        else:
            x=F.softmax(x, dim=1)
        return x


class DL_model(nn.Module):

    def __init__(self):
        super(DL_model, self).__init__()

        self.resnet=[]
        for i in range(8):
            self.resnet.append(torchvision.models.resnet.BasicBlock(1,256))
        self.resnet=nn.Sequential(*self.resnet)
        self.policy_branch=Branch(256,activation='soft')
        self.value_branch=Branch(256,activation='relu')

    def forward(self,state):
        out = self.resnet(state)
        policy=self.policy_branch(out)
        value=self.value_branch(out)
        return policy,value