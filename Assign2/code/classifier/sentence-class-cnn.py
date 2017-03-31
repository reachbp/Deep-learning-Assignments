__author__ = 'bharathipriyaa'


import argparse
import torch, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


###############################################################################
# Load data
###############################################################################

def create_dataset():
    alltrain_data_list = pickle.load(open("data.pkl", "rb"))
    alltrain_labels_list = np.array(pickle.load(open("target.pkl", "rb")))
    data_list = np.ndarray((len(alltrain_data_list),args.bptt))
    for idx, data in enumerate(alltrain_data_list):
        data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))] = alltrain_data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))]
    data_list = torch.from_numpy(data_list).long()
    labels_list = torch.from_numpy(alltrain_labels_list)
    train_dataset = torch.utils.data.TensorDataset(data_list, labels_list)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_dataset_loader

vocab = pickle.load(open("vocab.p", "rb"))
trainDataset_loader = create_dataset()

###############################################################################
# Build the model
###############################################################################

class Net(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.conv1 =  nn.Conv2d(10, 10, 5, stride = 1)
        self.fc1 = nn.Linear(100*300, 10*300)
        self.fc2 = nn.Linear(10*300, 5)


    def forward(self, x):
        #print(x)
        emb = self.embedding(x)
        #print("Output from embedding layer", emb.size())
        x = emb.view(-1, 10, 10, 30)
        #print("Output after resize layer", x.size())
        #x = self.conv1(x)
        #print("Output after convolution layer", x.size())
        x = x.view(-1, 100*300)
        #print("Output after resize layer", x.size())
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return F.log_softmax(x)

model = Net(ntoken=len(vocab.keys()), ninp=300)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = F.nll_loss

###############################################################################
# Training code
###############################################################################

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainDataset_loader):
        if args.cuda:
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target[:,0])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainDataset_loader.dataset),
                100. * batch_idx / len(trainDataset_loader), loss.data[0]))

for epoch in range(10):
    train(epoch)

