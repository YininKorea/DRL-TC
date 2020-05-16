import torch
import torch.nn as nn
import torchvision


class Branch(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels, activation='soft'):
        super(Branch, self).__init__()

        self.conv=nn.Conv2d(in_channels,256,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(256)
        self.act1=nn.ReLU()
        if activation=='relu':
            self.act2=nn.ReLU()
        else:
            self.act2=nn.Softmax(dim=1)
        self.dense=nn.Linear(input_dim**2*256, out_channels)

    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        x=self.act1(x)
        x=torch.flatten(x, 1)
        x=self.dense(x)
        x=self.act2(x)
        return x


class Model(nn.Module):

    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.resnet=[]
        self.resnet.append(torchvision.models.resnet.BasicBlock(1,256))
        for i in range(8):
            self.resnet.append(torchvision.models.resnet.BasicBlock(256,256))
        self.resnet=nn.Sequential(*self.resnet)
        self.policy_branch=Branch(input_dim, 256, 1024, activation='soft')
        self.value_branch=Branch(input_dim, 256, 1,activation='relu')

    def forward(self,state):
        out = self.resnet(state)
        policy=self.policy_branch(out)
        value=self.value_branch(out)
        return policy,value

class DNN:
    def __init__(self, input_dim, minibatch, learning_rate):
        self.input_dim = input_dim
        self.model = Model(input_dim).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataset):
        self.model.train()
        # for dataset minibatch
        # run network
        # run loss
        # apply optimizer
        
    def eval(self, input):
        self.model.eval()
        tensor = torch.tensor(input).double().unsqueeze(0).unsqueeze(0)
        raw_policy, raw_value = self.model(tensor)
        #output policy dist is long vector, reshape to matrix
        return raw_policy.detach().numpy()[:,:self.input_dim**2].reshape(self.input_dim, -1), raw_value