import torch
import torch.nn as nn
import numpy as np
import torchvision

import torch.utils.data as data

class PolicyBranch(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels):
        super(PolicyBranch, self).__init__()

        self.conv=nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(2)
        self.act1=nn.ReLU()
        self.dense=nn.Linear(input_dim**2*2, out_channels)
        self.act2=nn.Softmax(dim=1)

    def forward(self,x,mask):
        x=self.conv(x)
        x=self.norm(x)
        x=self.act1(x)
        x=torch.flatten(x, 1)
        x=self.dense(x)
        x=x*mask
        x=self.act2(x)
        return x

class ValueBranch(nn.Module):
    def __init__(self, input_dim, in_channels):
        super(ValueBranch, self).__init__()

        self.conv=nn.Conv2d(in_channels,1,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(1)
        self.act1=nn.ReLU()
        self.dense=nn.Linear(input_dim**2, 256)
        self.act2=nn.ReLU()
        self.dense2=nn.Linear(256, 1)
        self.act3=nn.ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.norm(x)
        x=self.act1(x)
        x=torch.flatten(x, 1)
        x=self.dense(x)
        x=self.act2(x)
        x=self.dense2(x)
        x=self.act3(x)
        return x


class Model(nn.Module):

    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.resnet=[]
        self.resnet.append(nn.Conv2d(1, 256, 3, padding=1))
        self.resnet.append(nn.BatchNorm2d(256))
        self.resnet.append(nn.ReLU(inplace=True))
        for i in range(8):
            self.resnet.append(torchvision.models.resnet.BasicBlock(256,256))
        self.resnet=nn.Sequential(*self.resnet)
        self.policy_branch=PolicyBranch(input_dim, 256, 1024)
        self.value_branch=ValueBranch(input_dim, 256)
        #nn.init.uniform_(self.value_branch.dense.weight)
        #nn.init.constant_(self.value_branch.dense.bias, 1/1024)
        #nn.init.uniform(self.policy_branch.dense.weight)

    def forward(self, state, mask):
        out = self.resnet(state)
        policy=self.policy_branch(out, mask)
        value=self.value_branch(out)
        return policy,value

class DNN:
    def __init__(self, input_dim, minibatch, learning_rate):
        self.input_dim = input_dim
        self.batch_size = minibatch
        self.model = Model(input_dim).float().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataset):
        self.model.train()
        dataloader = data.DataLoader(dataset, self.batch_size, shuffle=True)
        total_loss = 0
        for batch, (state, policy, value, mask) in enumerate(dataloader):

            state = state.float().cuda()
            policy = policy.float().cuda()
            value = value.float().cuda()
            mask = mask.cuda()

            self.optimizer.zero_grad()
            pred_policy, pred_value = self.model(state.unsqueeze(1), mask)
            loss_policy = (policy * pred_policy.log()).sum(-1).mean()
            loss_value = ((value-pred_value)**2).sum(1).mean()
            loss = loss_value - loss_policy # + reg l2 norm of all params
            total_loss += loss
            #loss_policy.backward(retain_graph=True)
            loss.backward()

            print(f'batch: {batch}, loss_policy: {loss_policy.cpu().data.numpy():.2f}, loss_value: {loss_value.cpu().data.numpy():.2f}')

            self.optimizer.step()
        print(f'loss: {total_loss/(batch+1)}')
        
    def eval(self, in_data):
        self.model.eval()
        mask = self.prepare_mask(in_data)
        tensor = torch.tensor(in_data).cuda().float().unsqueeze(0).unsqueeze(0)
        raw_policy, raw_value = self.model(tensor, mask)
        #print(raw_policy, raw_value)
        #output policy dist is long vector, reshape to matrix
        return raw_policy.cpu().data.numpy()[:,:self.input_dim**2].reshape(self.input_dim, -1), raw_value.cpu().data.numpy()[-1,-1]

    def prepare_mask(self, state):
        padded = np.zeros(1024)
        mask = valid_actions(state).flatten()
        padded[:mask.shape[0]] = mask
        return torch.from_numpy(padded).cuda()

class Dataset(data.Dataset):

    def __init__(self):
        self.data = []

    def __getitem__(self, idx):
        entry = self.data[-idx]
        state = entry[0]
        policy = np.zeros(1024)
        policy[:len(entry[1])] = entry[1] #pad zeros
        value = np.zeros(1)
        value[0] = entry[2]#/1000
        mask = np.zeros(1024)
        mask[:len(entry[1])] = valid_actions(entry[0]).flatten()
        return torch.from_numpy(state), torch.from_numpy(policy), torch.from_numpy(value), torch.from_numpy(mask)

    def __len__(self):
        return min(len(self.data), 100)

    def add(self, data):
        self.data.append(data)

def valid_actions(state):
    not_connected = np.all(state == 0, axis=0)
    return np.outer(~not_connected, not_connected)