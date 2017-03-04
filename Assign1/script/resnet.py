from __future__ import print_function

import argparse
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from torch.autograd import Variable
from torchvision import transforms, models

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 10)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                                           bias=False)
        self.model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.model.avgpool = torch.nn.AvgPool2d(3)

    def forward(self, x):
        x = self.model.forward(x)
        return F.log_softmax(x)

model = MyResNet()
if args.cuda:
    model = model.cuda()

trainset_labeled_import = pickle.load(open("../data/kaggle/train_labeled.p", "rb"))
trainset_unlabeled_import = pickle.load(open("../data/kaggle/train_unlabeled.p", "rb"))
validset_import = pickle.load(open("../data/kaggle/validation.p", "rb"))

train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled_import, batch_size=128, shuffle=True)
train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled_import, batch_size=256,
                                                     shuffle=True)

train_unlabeled_loader.dataset.train_labels = [-1 for i in range(len(train_unlabeled_loader.dataset.train_data))]
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=128, shuffle=True)

# optim.SGD(params, lr=<object>, momentum=0, dampening=0, weight_decay=0)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)


# optimizer = optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)


# In[152]:

# Single channel only
def tensor_imshow(tensor):
    npimg = tensor.numpy()[0]
    np_imshow(npimg)


def np_imshow(npimg):
    plt.figure(1)
    plt.imshow(npimg, cmap='gray')
    plt.show()


# In[194]:

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
        out[top:top + zh, left:left + zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out


# In[195]:

# Data Augmentations

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


def random_scale(npimg, min_scale=0.5, max_scale=1.5):
    scale_factor = random.uniform(min_scale, max_scale)
    sc_img = clipped_zoom(npimg, scale_factor, mode='nearest')
    return sc_img


def random_gaussian_noise(npimg):
    h, w = npimg.shape
    mu, sigma = 0, 0.1
    s = np.random.normal(mu, sigma, h * w)
    s = s.reshape((h, w))
    return npimg + s


# Converts a HxW ndarray into a 1xHxW tensor
def to_tensor(npimg):
    h, w = npimg.shape
    npimg = npimg.reshape(1, h, w)
    return torch.from_numpy(npimg)


# In[196]:

# CPU only training
def train_without_jitter(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_labeled_loader):
        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train - Jitterfree Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_labeled_loader.dataset),
                       100. * batch_idx / len(train_labeled_loader), loss.data[0]))


def train_with_jitter(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_labeled_loader):

        for i in range(len(data)):
            data[i] = transforms.Compose([
                transforms.Lambda(lambda x: to_npimg(x)),
                transforms.Lambda(lambda x: random_gaussian_noise(x)),
                transforms.Lambda(lambda x: random_translate(x)),
                transforms.Lambda(lambda x: random_rotate(x)),
                transforms.Lambda(lambda x: random_scale(x)),
                transforms.Lambda(lambda x: to_tensor(x)),
            ])(data[i])
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train - Jitter Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_labeled_loader.dataset),
                       100. * batch_idx / len(train_labeled_loader), loss.data[0]))


def validate(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target)

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data.cpu().numpy()[0], correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))


def train_unlabeled(epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_unlabeled_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        optimizer.zero_grad()
        output = model(data)
        target = Variable(torch.LongTensor(output.data.cpu().max(1)[1].numpy().reshape(-1)).cuda())
        #         loss = my_criterion.forward(output, target, Variable(torch.LongTensor(epoch)),
        # Variable(torch.LongTensor(2)))
        #         my_criterion.backward(loss)
        loss = F.nll_loss(output, target)
        if epoch > 70:
            loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Unlabeled Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_unlabeled_loader.dataset),
                       100. * batch_idx / len(train_unlabeled_loader), loss.data[0]))


# In[198]:


# In[199]:
#
for epoch in range(1, 100):
    train_without_jitter(epoch)
    train_with_jitter(epoch)
    train_unlabeled(epoch)

    if epoch == 99:
        pickle.dump(model, open("model_resnet.p", "wb"))

    validate(epoch, valid_loader)

# # Create Sample Submission

# In[10]:
# model = pickle.load(open("good_model.p", "rb"))

testset = pickle.load(open("../data/kaggle/test.p", "rb"))

test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

label_predict = np.array([])
model.eval()
for data, target in test_loader:
    if args.cuda:
        data = data.cuda()
        target = target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.cpu().max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))

# In[14]:

label_predict

# In[17]:

import pandas as pd

predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)

# In[18]:

predict_label.head()

# In[19]:

predict_label.to_csv('../data/kaggle/sample_submission.csv', index=False)


# In[ ]:
