import torch
import torch.nn as nn
import numpy as np
import torchvision
import utils
import torch.utils.data as data

action_space_size = 1024

class PolicyBranch(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels):
        super(PolicyBranch, self).__init__()

        self.conv=nn.Conv2d(in_channels,256,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(256)
        self.act1=nn.ReLU()
        self.dense=nn.Linear(input_dim**2*256, out_channels)
        self.act2=nn.Softmax(dim=-1)

    def forward(self, x):
        x=self.conv(x)
        x=self.norm(x)
        x=self.act1(x)
        x=torch.flatten(x, 1)
        x=self.dense(x)
        x=self.act2(x)
        return x

class ValueBranch(nn.Module):
    def __init__(self, input_dim, in_channels):
        super(ValueBranch, self).__init__()

        self.conv=nn.Conv2d(in_channels,256,kernel_size=3,stride=1,padding=1)
        self.norm=nn.BatchNorm2d(256)
        self.act1=nn.ReLU()
        self.dense=nn.Linear(input_dim**2*256, 1)
        self.act2=nn.ReLU()

    def forward(self, x):
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
        self.resnet.append(nn.Conv2d(1, 256, kernel_size=3, padding=1))
        self.resnet.append(nn.BatchNorm2d(256))
        self.resnet.append(nn.ReLU(inplace=True))
        for i in range(8):
            self.resnet.append(torchvision.models.resnet.BasicBlock(256,256))
        self.resnet=nn.Sequential(*self.resnet)
        self.policy_branch=PolicyBranch(input_dim, 256, action_space_size)
        self.value_branch=ValueBranch(input_dim, 256)

    def forward(self, state):
        out = self.resnet(state)
        policy = self.policy_branch(out)
        value = self.value_branch(out)
        return policy, value

class DNN:
    def __init__(self, input_dim, minibatch, learning_rate):
        self.input_dim = input_dim
        self.batch_size = minibatch
        self.model = Model(input_dim).float().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        #self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        #self.lr_scheduler = getattr(utils.lr_scheduler, 'poly')(self.optimizer, self.epochs, len(train_loader))

    def train(self, dataset,epoch):
        self.model.train()

        dataloader = data.DataLoader(dataset, self.batch_size, shuffle=True)
        self.lr_scheduler = getattr(utils.lr_scheduler, 'Poly')(self.optimizer,5, len(dataloader))
        total_loss = 0
        for batch, (state, policy, value) in enumerate(dataloader):
            self.lr_scheduler.step(epoch=epoch - 1)
            state = state.float().cuda()
            policy = policy.float().cuda()
            value = value.float().cuda()

            pred_policy, pred_value = self.model(state.unsqueeze(1)) # add one channel to state
            loss_policy = (policy * pred_policy.log()).sum(dim=-1).mean()
            loss_value = ((value-pred_value)**2).sum(dim=-1).mean()
            loss = loss_value - loss_policy # + reg l2 norm of all params
            total_loss += loss
            # normalize loss if batch is not full?!
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'batch: {batch}, loss_policy: {loss_policy.cpu().data.numpy():.2f}, loss_value: {loss_value.cpu().data.numpy():.2f}')
        print(f'loss: {total_loss/(batch+1)}')
        
    def eval(self, in_data):
        self.model.eval()
        in_data = torch.from_numpy(in_data).unsqueeze(0)#put into batch
        tensor = in_data.float().cuda()
        raw_policy, raw_value = self.model(tensor.unsqueeze(1))
        #print(raw_policy, raw_value)
        #output policy dist is long vector, reshape to matrix
        return raw_policy.cpu().data.numpy()[:,:self.input_dim**2].reshape(self.input_dim, -1), raw_value.cpu().data.numpy()[-1,-1]

class Dataset(data.Dataset):

    def __init__(self):
        self.data = []

    def __getitem__(self, idx):
        entry = self.data[idx]
        state = entry[0]
        policy = np.zeros(action_space_size)
        policy[:len(entry[1])] = entry[1] #pad zeros
        value = np.zeros(1)
        value[0] = entry[2]#/1000
        return torch.from_numpy(state), torch.from_numpy(policy), torch.from_numpy(value)

    def __len__(self):
        return len(self.data)#min(len(self.data), 100)

    def add(self, data):
        self.data.append(data)
