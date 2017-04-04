__author__ = 'bharathipriyaa'

from sklearn import metrics
from sklearn.metrics import confusion_matrix
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
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='epochs')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--emsize', type=int, default=30,
                    help='embedding size')
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
    alltrain_data_list = pickle.load(open("data_imdb.pkl", "rb"))
    alltrain_labels_list = np.array(pickle.load(open("target_imdb.pkl", "rb")))
    data_list = np.ndarray((len(alltrain_data_list),args.bptt))
    
    for idx, data in enumerate(alltrain_data_list):
        data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))] = alltrain_data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))]
    data_list = torch.from_numpy(data_list).long()
    labels_list = torch.from_numpy(alltrain_labels_list)
    l = len(data_list)
    r = (int) (0.7 *l)
    train_dataset = torch.utils.data.TensorDataset(data_list[1:r], labels_list[1:r])
    valid_dataset = torch.utils.data.TensorDataset(data_list[r:l], labels_list[r:l])
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset_loader, val_dataset_loader

vocab = pickle.load(open("vocab_imdb.p", "rb"))
trainDataset_loader, val_dataset_loader = create_dataset()

###############################################################################
# Build the model
###############################################################################

class Net(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.conv1 =  nn.Conv1d(100, 5, 5, stride = 1)
        self.maxpool = F.max_pool2d # nn.Conv2d(10, 10, 5, stride = 1)
        self.fc1 = nn.Linear(100*300, 10*300)
        self.fc2 = nn.Linear(2*13, 5)


    def forward(self, x):
        #print(x)
        x = self.embedding(x)
        #print("Output from embedding layer", x.size())
        #x = emb.view(-1, 10, 10, args.emsize)
        #print("Output after resize layer", x.size())
        x = self.conv1(x)
        #print("Output after convolution layer", x.size())
        x = self.maxpool(x, 2, 2)
        #print("Output after maxpool layer", x.size())
        x = x.view(-1, 2*13)
        #print("Output after resize layer", x.size())
        #x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.log_softmax(x)

model = Net(ntoken=len(vocab.keys()), ninp=args.emsize)
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

def test(epoch):
    correct = 0    
    test_loss = 0
    y_true =[]
    y_pred = []
    for batch_idx, (data, target) in enumerate(val_dataset_loader):
        if args.cuda:
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target[:,0])
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        y_true.extend(target.data.cpu().numpy())
        y_pred.extend(pred.cpu().numpy().squeeze())
    print(y_true)	
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(val_dataset_loader.dataset),
    100. * correct / len(val_dataset_loader.dataset)))

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
