import torch
import torch.optim as optimizer
from Models import DL_model
import numpy as np
from torch.utils import data
import torch.utils.data as dataf
from torch.autograd import Variable


cuda=False
epoch=10
train_lr = 0.00001
train_momentum = 0.5
#batch_size=16
policy_loss=torch.nn.CrossEntropyLoss()
value_loss=torch.nn.CrossEntropyLoss()
number_node=256
train_value=np.random.rand(10,1,256,256)## assume we have 256 nodes, 10 eposides
#train_value=np.random.rand(10)##value function , can be roughly considered as a label of each state
train_policy=np.random.rand(10,1,256,256)
train_label=np.random.rand(10,1,256,256)


train_policy=torch.from_numpy(train_policy).float()
train_value=torch.from_numpy(train_value).float()
train_label=torch.from_numpy(train_label).float()

Dataloader_value = dataf.TensorDataset(train_value)
Dataloader_policy = dataf.TensorDataset(train_policy)
Values = dataf.TensorDataset(train_value)
#print(len(Dataloader_value))
Train_value=dataf.DataLoader(Dataloader_value,batch_size=1,shuffle=True)
Train_policy=dataf.DataLoader(Dataloader_policy,batch_size=1,shuffle=True)


def train():
    model=DL_model(in_channels=1).to('cpu')
    for i in range(epoch):
        for indice, (value,policy) in enumerate(zip(Train_value,Train_policy)):
            if cuda:
                value, policy = value.cuda(), policy.cuda()
            value, policy = Variable(value[0]), Variable(policy[0])
            po_output,va_output=model(policy,value)
            print(po_output.shape)
            print(va_output.shape)


if __name__=="__main__":
    train()