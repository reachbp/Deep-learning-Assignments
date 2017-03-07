from __future__ import print_function
import pickle 
import time

import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import pandas as pd


####################################################################################################
# Constants
####################################################################################################


model_file = 'mnist_9946.p'

test_file = 'test.p'

cuda = False


####################################################################################################
# Model
####################################################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 512)
        self.fc2_surrogate = nn.Linear(512, 2000)
        self.fc2_mnist = nn.Linear(512, 10)

    def forward(self, x, surrogate=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        if surrogate:
            x = F.relu(self.fc2_surrogate(x))
        else:
            x = F.relu(self.fc2_mnist(x))
        return F.log_softmax(x)


class NetBN(nn.Module):
    
    def __init__(self):
        super(NetBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=0, padding=0, ceil_mode=True)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv1_drop = nn.Dropout2d(0.2)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.conv3_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2_mnist = nn.Linear(512, 10)
        self.fc2_surrogate = nn.Linear(512, surrogate_classes)

    def forward(self, x, surrogate=False):
        x = self.conv1_drop(F.relu(self.maxpool(self.conv1_bn(self.conv1(x)))))
        x = self.conv2_drop(F.relu(self.maxpool(self.conv2_bn(self.conv2(x)))))
        x = self.conv3_drop(F.relu(self.maxpool(self.conv3_bn(self.conv3(x)))))
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        if surrogate:
            x = F.relu(self.fc2_surrogate(x))
        else:
            x = F.relu(self.fc2_mnist(x))
        return F.log_softmax(x)


####################################################################################################
# Load data
####################################################################################################


testset = pickle.load(open("../data/kaggle/" + test_file, "rb"))
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

print('Completed loading data.')

####################################################################################################
# Load model
####################################################################################################


model = pickle.load(open(model_file, 'rb'))

print('Completed loading the model.')


####################################################################################################
# Predict
####################################################################################################


label_predict = np.array([])
model.eval()
for data, target in test_loader:
    if cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.cpu().max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))


####################################################################################################
# Write results
####################################################################################################


predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)
predict_label.to_csv('../data/kaggle/' + model_file + '.csv', index=False)

print('Completed generating the submission file.')
