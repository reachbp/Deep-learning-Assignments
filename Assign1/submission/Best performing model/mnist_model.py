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

import matplotlib.pyplot as plt
import random
import math
import scipy
from scipy import ndimage


####################################################################################################
# Constants
####################################################################################################


surrogate_classes = 2048
variations_per_image = 16

unlabeled_train_iterations = 50
unlabeled_train_epochs = 20

train_epochs = 300

unlabeled_train_start_epoch = 40

cuda = False

save_models = True

max_accuracy = 9910

use_pretrained_model = False
model_file = 'mnist_1488751090_9917.p'

log_file_name = 'a1_' + str(int(time.time())) + '.txt'


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
# Print helpers
####################################################################################################

def print_log(log_str):
    global log_file_name
    with open(log_file_name, 'a') as f:
        f.write(str(log_str))
        print(log_str)


####################################################################################################
# Data augmentations
####################################################################################################


# Transforms a tensor into a numpy array
# ONLY FOR SINGLE CHANNEL
def to_npimg(tensor):
    t = (tensor[0]).numpy()
    return t


def random_translate(npimg, min_translate=-0.2, max_translate=0.2):
    
    h, w = npimg.shape
    min_pixels = ((h + w) / 2) * min_translate
    max_pixels = ((h + w) / 2) * max_translate
    shift = int(random.uniform(min_pixels, max_pixels))
    
    return ndimage.interpolation.shift(npimg, shift, mode='nearest')


def random_rotate(npimg, min_rotate=-30, max_rotate=30):
    
    theta = random.randint(int(min_rotate), int(max_rotate + 1))
    return ndimage.interpolation.rotate(npimg, theta, reshape=False, mode='nearest')


def random_scale(npimg, min_scale=0.7, max_scale=1.3):
    
    scale_factor = random.uniform(min_scale, max_scale)
    sc_img = clipped_zoom(npimg, scale_factor, mode='nearest')
    return sc_img
    

# Converts a HxW ndarray into a 1xHxW tensor
def to_tensor(npimg):
    h, w = npimg.shape
    npimg = npimg.reshape(1, h, w)
    return torch.from_numpy(npimg)


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out += np.amin(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


# Single channel only
def tensor_imshow(tensor):    
    npimg = tensor.numpy()[0]
    np_imshow(npimg)


def np_imshow(npimg):
    plt.figure(1)
    plt.imshow(npimg, cmap='gray')
    plt.show()


def random_gaussian_noise(npimg):
    h, w = npimg.shape
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, h * w)
    s = s.reshape((h, w))
    return npimg + s


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
                    transforms.Lambda(lambda x: to_npimg(x)),
                    transforms.Lambda(lambda x: random_translate(x)),
                    transforms.Lambda(lambda x: random_rotate(x)),
                    transforms.Lambda(lambda x: random_scale(x)),
                    transforms.Lambda(lambda x: to_tensor(x)),
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


def train(epoch, data_loader, surrogate=False, jitter=False):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if jitter:
            for i in range(len(data)):
                data[i] = transforms.Compose([
                    transforms.Lambda(lambda x: to_npimg(x)),
                    transforms.Lambda(lambda x: random_gaussian_noise(x)),
                    transforms.Lambda(lambda x: random_translate(x)),
                    transforms.Lambda(lambda x: random_rotate(x)),
                    transforms.Lambda(lambda x: random_scale(x)),
                    transforms.Lambda(lambda x: to_tensor(x)),
                ])(data[i])
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, surrogate)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
    print_log('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    check_and_save_model(correct)


def train_unlabeled(epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_unlabeled_loader):
        if cuda:
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        target = Variable(torch.LongTensor(output.data.cpu().max(1)[1].numpy().reshape(-1)).cuda())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print_log('Train Unlabeled Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_unlabeled_loader.dataset),
                       100. * batch_idx / len(train_unlabeled_loader), loss.data[0]))


def check_and_save_model(correct):
    global max_accuracy
    if correct > max_accuracy:
        print_log('Saving model as accuracy has increased to' + str(correct) + '.')
        max_accuracy = correct
        pickle.dump(model, open("mnist_" + str(int(time.time())) + "_" + str(correct) + ".p", "wb"))


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

print_log('Completed loading data.')


####################################################################################################
# Model and Optimizer
####################################################################################################

if use_pretrained_model:
    model = pickle.load(open(model_file, 'rb'))
else:
    model = NetBN()
    # model = Net()

if cuda:
    model = model.cuda()

# optim.SGD(params, lr=<object>, momentum=0, dampening=0, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

# optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


####################################################################################################
# Train on unlabeled images (surrogate data)
####################################################################################################

if use_pretrained_model == False: 
    print_log('Beginning training on surrogate class data...')

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
        surrogate_dataset.target_tensor = target
        
        # Train for n epochs
        for j in range(1, unlabeled_train_epochs + 1):
            train(i * unlabeled_train_iterations + j, surrogate_dataset_loader, surrogate=True)

        print_log('Completed' + str(i + 1) + 'iteration(s) on the unlabeled data.')

    # Save model trained on surrogate classes
    if save_models:
        pickle.dump(model, open("surrogate_" + str(int(time.time())) + ".p", "wb"))

    print_log('Completed training on surrogate class data.')


####################################################################################################
# Train on labeled images
####################################################################################################


print_log('Beginning training on labeled and unlabeled data...')

for epoch in range(1, train_epochs + 1):
    train(epoch, train_loader, surrogate=False, jitter=True)
    if epoch % 2 == 0:
        train(epoch, train_loader, surrogate=False, jitter=False)
    if epoch >= 40 and epoch % 2 == 0:
        train_unlabeled(epoch)
    test(epoch, valid_loader)

# Save model trained on labeled images
if save_models:
    pickle.dump(model, open("mnist_" + str(int(time.time())) + ".p", "wb"))

print_log('Completed training on labeled and unlabeled data.')

