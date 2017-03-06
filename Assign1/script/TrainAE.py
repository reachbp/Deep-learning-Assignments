
# coding: utf-8

# In[96]:

from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
from autoencoders import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# # Train Model

# In[136]:

trainset_imoprt = pickle.load(open("../data/kaggle/train_labeled.p", "rb"))
validset_import = pickle.load(open("../data/kaggle/validation.p", "rb"))
trainset_unlabeled_import = pickle.load(open("../data/kaggle/train_unlabeled.p", "rb"))
trainset_unlabeled_import.train_labels = [10 for i in range(len(trainset_unlabeled_import.train_data))]
train_dataset_loader = torch.utils.data.DataLoader(trainset_imoprt, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)
unlabelled_dataset_loader = torch.utils.data.DataLoader(trainset_unlabeled_import, batch_size=64, shuffle=True)


# In[199]:

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.latent_vars = 150
        
        self.fc11 = nn.Linear(64*3*3, 300)
        self.fc12 = nn.Linear(128, 64)
        
        
        self.conv1 = nn.Conv2d(1, 16, 2, 2, 1, 1, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 2, 1, 1, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1, 1, 1, 1)
    
        self.dconv1 = nn.ConvTranspose2d(16, 16, 3,3,2,2)
        self.dconv2 = nn.ConvTranspose2d(16, 32, 3,3,2,2)
        self.dconv3 = nn.ConvTranspose2d(32, 1, 3,3,2,2)
        
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc3 = nn.Linear(150, 64)
        #self.fc4 = nn.Linear(400, 784)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encode(self, x):
        #print("Input size" , x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.conv3(x))
        #print("Size after encoding", x.size())
        x = x.view(-1, 64*3*3)
        #print(x.size())
        z = self.fc11(x)
        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]
      
        #print(mu.size(), log_sig.size())
        return mu, log_sig

    def reparametrize(self, mu, logvar):
        eps = Variable(torch.randn(logvar.size()))
        z = mu + torch.exp(logvar / 2) * eps
        #print("Reparam layer ", z.size())
        return z

    def decode(self, x):
        #print("Input size 0", x.size())
        x = F.relu(self.fc3(x))
        #print("Checkpoint 0", x.size())
        x = x.view(-1,16, 2,2)
        #print("Checkpoint 1", x.size())
        x = self.bn1(self.dconv1(x))
        #print("Checkpoint 2", x.size())
        x = self.bn2(self.dconv2(x))
        #print("Checkpoint 3", x.size())
        x = self.dconv3(x)
        #print("output size", x.size())
        
        return x #self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# In[200]:

model = ConvVAE()
mse = torch.nn.MSELoss()


# In[201]:

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar):
    mean_squared =mse(X_hat, data)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mean_squared + KLD


model = VAE() #torch.load("decon-vae-50.p")
model_name = 'sdfdenoise'



def generate_train(epoch,  data_loader,isSupervised = False):
    model.train()
    train_loss = 0
    classification_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #
        data, target = Variable(data), Variable(target)
        
        # Add noise
        if model_name == 'denoise':
            noise = torch.FloatTensor(len(data), 1,28,28)
            noiseV = Variable(noise)
            noiseV.resize_as(data)
            noiseV.data.normal_(0, 1)
            data.data.add_(noiseV.data)
        
        optimizer.zero_grad()
        x_new, mu, logvar = model(data)
        kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1. - logvar)
       
        recon_loss = mse(x_new, data) + kl_loss
        class_loss = 0
        if isSupervised:
            output = F.log_softmax(mu)
            
            class_loss = F.nll_loss(output, target)
            classification_loss += class_loss.data[0]
        loss = recon_loss + class_loss
        loss.backward()
        train_loss += loss.data[0]
        
        optimizer.step()
    print('====> Epoch: {} Average reconstruction loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))
    print('====> Epoch: {} Classification loss: {:.4f}'.format(
          epoch, classification_loss / len(data_loader.dataset)))

    
def generate_test(epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        x_new, mu, logvar = model(data)
        output = F.log_softmax(mu) 
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(1, 100):
    generate_train(epoch, data_loader=unlabelled_dataset_loader, isSupervised = False)
    generate_train(epoch, data_loader=train_dataset_loader, isSupervised = True)
    generate_test(epoch, valid_loader)


# In[86]:

# Save model
torch.save(model, "decon-vae-50.p")



from  torchvision.utils import save_image
for index, (data, target) in enumerate(train_loader):
    data, target = Variable(data), Variable(target)
    gen_data, mu, logvar = model(data)
    gen_images = gen_data.view(-1, 1,28,28)
    #print(data[0])
    save_image(gen_images.data, 'images/gen{}.jpg'.format(index))

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.8)
def train(epoch):
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        gen_data, mu, logvar = model(data)
        #Check reconstruction loss 
        recon_loss = reconstruction_function(gen_data, data)
        #print("Reconstruction loss is", recon_loss)
        #print("Type of generated data", type(gen_data))
        
        optimizer.zero_grad()
        output = classifier(gen_data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch, valid_loader):
    classifier.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        gen_data, mu, logvar = model(data)
        output = classifier(gen_data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    

for epoch in range(1, 10):
    train(epoch)
    test(epoch, valid_loader)


# # Create Sample Submission

def create_test_submission():
    testset = pickle.load(open("../data/kaggle/test.p", "rb"))
    test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)
    test(1, test_loader)
    label_predict = np.array([])
    model.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        temp = output.data.max(1)[1].numpy().reshape(-1)
        label_predict = np.concatenate((label_predict, temp))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    predict_label.to_csv('../data/kaggle/sample_submission.csv', index=False)

