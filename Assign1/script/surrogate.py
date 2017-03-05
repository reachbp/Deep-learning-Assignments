from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import augmentations
import time

####################################################################################################
# Constants
####################################################################################################

surrogate_classes = 2000
variations_per_image = 20

unlabeled_train_iterations = 5
unlabeled_train_epochs = 5

train_epochs = 10

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
# Surrogate class data generator
####################################################################################################

def generate_data():
    model.train()

    global variations_per_image
    global surrogate_classes

    images_to_process = surrogate_classes / variations_per_image
    images_processed = 0
    
    augmented_images = []
    surrogate_labels = []
    
    for batch_idx, (data, _) in enumerate(train_unlabeled_loader):
        
        for i in range(len(data)):
            
            images_processed += 1
            
            if images_processed > images_to_process:
                break
                
            new_data = []
            
            for j in range(variations_per_image):
                new_image = transforms.Compose([
                    transforms.Lambda(lambda x: augmentations.to_npimg(x)),
                    transforms.Lambda(lambda x: augmentations.random_translate(x)),
                    transforms.Lambda(lambda x: augmentations.random_rotate(x)),
                    transforms.Lambda(lambda x: augmentations.random_scale(x)),
                    transforms.Lambda(lambda x: augmentations.to_tensor(x)),
                    ])(data[i])

                c, h, w = new_image.size()
                new_image.resize_(1, c, h, w)
                new_data.append(new_image)
                surrogate_labels.append(images_processed)
            
            augmented_images.append(torch.cat(new_data, 0))
        
    augmented_images_tensor = torch.cat(augmented_images, 0)
    surrogate_labels_tensor = torch.LongTensor(surrogate_labels)
    
    return (augmented_images_tensor, surrogate_labels_tensor)

####################################################################################################
# Train and Test
####################################################################################################

def train(epoch, data_loader, surrogate=False):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, surrogate)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

####################################################################################################
# Load data
####################################################################################################

trainset_imoprt = pickle.load(open("../data/kaggle/train_labeled.p", "rb"))
validset_import = pickle.load(open("../data/kaggle/validation.p", "rb"))
trainset_unlabeled_import = pickle.load(open("../data/kaggle/train_unlabeled.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)

train_unlabeled_loader = torch.utils.data.DataLoader(
    trainset_unlabeled_import, batch_size=256, shuffle=True)
train_unlabeled_loader.dataset.train_labels = [-1 for i in range(
    len(train_unlabeled_loader.dataset.train_data))]

print('Completed loading data')

####################################################################################################
# Model and Optimizer
####################################################################################################

# model = NetBN()
model = Net()

if cuda:
    model = model.cuda()

# optim.SGD(params, lr=<object>, momentum=0, dampening=0, weight_decay=0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

# optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)

####################################################################################################
# Train on unlabeled images (surrogate data)
####################################################################################################

print('Beginning training on unlabeled data...')

# Train on augmented images and surrogate classes
for i in range(unlabeled_train_iterations):
    
    # Generate data
    data, target = generate_data()

    if cuda:
        data, target = data.cuda(), target.cuda()

    # Convert to torch dataset
    surrogate_dataset = torch.utils.data.TensorDataset(data, target)
    surrogate_dataset_loader = torch.utils.data.DataLoader(
        surrogate_dataset, batch_size=64, shuffle=True)
    
    # Train for n epochs
    for j in range(1, unlabeled_train_epochs + 1):
        train(i * unlabeled_train_iterations + j, surrogate_dataset_loader, True)

    print('Completed', i + 1, 'iteration(s) on the unlabeled data')

# Save model trained on surrogate classes
pickle.dump(model, open("surrogate_" + str(int(time.time())) + ".p", "wb"))

print('Completed training on unlabeled data.')

####################################################################################################
# Train on labeled images
####################################################################################################

print('Beginning training on labeled data...')

for epoch in range(1, train_epochs + 1):
    train(epoch, train_loader)
    test(epoch, valid_loader)

# Save model trained on labeled images
pickle.dump(model, open("mnist_" + str(int(time.time())) + ".p", "wb"))

print('Completed training on labeled data...')
