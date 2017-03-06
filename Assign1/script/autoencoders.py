
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.latent_vars = 10
        self.fc11 = nn.Linear(784, 400)
        self.fc12 = nn.Linear(400, 100)
        self.fc21 = nn.Linear(100, 20)

        self.fc3 = nn.Linear(10, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc11(x))
        h2 = self.relu(self.fc12(h1))
        z = self.fc21(h2)
        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]
        return mu, log_sig

    def reparametrize(self, mu, logvar):
        
        eps = Variable(torch.randn(logvar.size()))
        z = mu + torch.exp(logvar / 2) * eps
        return z

    def decode(self, z):
        
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


### Adding convolution layers to VAE


### Adding DenoisingAE

class DenoiseAE(nn.Module):
    def __init__(self):
        super(DenoiseAE, self).__init__()
        self.latent_vars = 10
        
        self.fc11 = nn.Linear(64, 20)
        self.fc12 = nn.Linear(128, 64)
        
        self.conv1 = nn.Conv2d(1, 32, 3, 3, 1, 1, 1, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 3, 1, 1, 1, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 3, 1, 1, 1, 1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.dconv1 = nn.ConvTranspose2d(16, 16, 3,3,2,2)
        self.dconv2 = nn.ConvTranspose2d(16, 32, 3,3,2,2)
        self.dconv3 = nn.ConvTranspose2d(32, 1, 3,3,2,2)
        
        self.noiser = WhiteNoise(0, 0.5)
        self.fc3 = nn.Linear(10, 64)
        #self.fc4 = nn.Linear(400, 784)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        
        x = self.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #print("Size after encoding", x.size())
        x = x.view(-1, 16*2*2)
        #print(x.size())
        z = self.fc11(x)
        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]
      
        #print(mu.size(), log_sig.size())
        return mu, log_sig

    def reparametrize(self, mu, logvar):
        eps = Variable(torch.randn(logvar.size()))
        z = mu + torch.exp(logvar / 2) * eps
        return z

    def decode(self, x):
        #print("Input size 0", x.size())
        x = F.relu(self.fc3(x))
        #print("Checkpoint 0", x.size())
        x = x.view(-1,16, 2,2)
        #print("Checkpoint 1", x.size())
        x = self.dconv1(x)
        #print("Checkpoint 2", x.size())
        x = self.dconv2(x)
        #print("Checkpoint 3", x.size())
        x = self.dconv3(x)
        
        return x #self.sigmoid(self.fc4(h3))

    def forward(self, x):
        #Add white noise
        #noise = torch.FloatTensor(64, 1,28,28)
        #noiseV = Variable(noise)
        #x.add(noiseV)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar