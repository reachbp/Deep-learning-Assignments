# coding: utf-8

#

from __future__ import print_function

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--learning-rate', type=float, default=.01, metavar='LR',
                    help='Learning rate (default: .01)')
parser.add_argument('--momentum', type=float, default=.8, metavar='MM',
                    help='Learning rate (default: .01)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default=Net, metavar='M',
                    help='Specify which model to use')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

trainset_imoprt = pickle.load(open("../data/kaggle/train_labeled.p", "rb"))
validset_import = pickle.load(open("../data/kaggle/validation.p", "rb"))

# In[5]:

train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=args.batch_size, shuffle=True)

model = args.model.Net()

# In[7]:

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)


# In[8]:

# CPU only training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


# In[9]:

for epoch in range(1, 20):
    train(epoch)
    test(epoch, valid_loader)

testset = pickle.load(open("../data/kaggle/test.p", "rb"))

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

test(1, test_loader)

label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))

label_predict

predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)

predict_label.head()

predict_label.to_csv('../data/kaggle/sample_submission.csv', index=False)
