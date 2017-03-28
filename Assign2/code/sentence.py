__author__ = 'bharathipriyaa'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


trainDataset = torch.randn(100, 1, 10, 100)
testDataset = torch.Tensor(10, 10, 300)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =  nn.Conv2d(1, 10, 5, stride = 1)
        self.fc1 = nn.Linear(5*60, 3)


    def forward(self, x):
        print("checkpoint 0", x.size())
        x = self.conv1(x)
        print("checkpoint 1", x.size())
        x = F.max_pool2d(x, 2, 2)
        print("checkpoint 2", x.size())
        print(x.size())
        print("checkpoint 3", x.size())
        x = x.view(-1, 5*60)
        x = self.fc1(x)

        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()

    data, target = Variable(trainDataset), Variable(trainDataset)
    optimizer.zero_grad()
    output = model(data)
    print(output.size())

for epoch in range(1, args.epochs + 1):
    train(epoch)

