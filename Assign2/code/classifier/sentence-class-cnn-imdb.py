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
parser.add_argument('--bptt', type=int, default=30,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='epochs')
parser.add_argument('--lr', type=float, default=.5,
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
    alltest_data_list = pickle.load(open("data_imdb_test.pkl", "rb"))
    alltrain_labels_list = np.array(pickle.load(open("target_imdb.pkl", "rb")))
    alltest_labels_list = np.array(pickle.load(open("target_imdb_test.pkl", "rb")))
    data_list = np.ndarray((len(alltrain_data_list),args.bptt))
    data_test_list = np.ndarray((len(alltest_data_list),args.bptt))
    for idx, data in enumerate(alltrain_data_list):
        data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))] = alltrain_data_list[idx][0:min(args.bptt, len(alltrain_data_list[idx]))]

    for idx, data in enumerate(alltest_data_list):
        data_test_list[idx][0:min(args.bptt, len(alltest_data_list[idx]))] = alltest_data_list[idx][0:min(args.bptt, len(alltest_data_list[idx]))]
    data_list = torch.from_numpy(data_list).long()
    data_test_list = torch.from_numpy(data_test_list).long()
    labels_list = torch.from_numpy(alltrain_labels_list)
    labels_test_list = torch.from_numpy(alltest_labels_list)
    l = len(data_list)
    r = (int) (0.7 *l)
    train_dataset = torch.utils.data.TensorDataset(data_list[1:r], labels_list[1:r])
    valid_dataset = torch.utils.data.TensorDataset(data_list[r:l], labels_list[r:l])
    test_dataset = torch.utils.data.TensorDataset(data_test_list, labels_test_list)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset_loader, val_dataset_loader,test_dataset_loader

vocab = pickle.load(open("vocab_imdb.p", "rb"))
trainDataset_loader, val_dataset_loader, test_dataset_loader = create_dataset()
idx2word = pickle.load(open("idx2word.pkl", "rb"))
###############################################################################
# Evaluating  samples
###############################################################################

def reconstruct_wrong_sent(input, output, target) :
 #   print(input.size(),output.size(),target.size() )
  #  assert(input.size(0) == output.size(0) and input.size(0) == target.size(0))
    for idx, sentence in enumerate(input):
        sentence = ''
        for index in input[idx]:
            word = idx2word[index]
            sentence += ' ' + word
	#print(np.argmax(output[idx]), target[idx])
        predicted, i_target = np.argmax(output[idx]), target[idx]
        if predicted != i_target:
            print("Review is ", sentence )
            print("Predicted = {}{}  Target = {} ".format(predicted, output[idx], i_target))

###############################################################################
# Build the model
###############################################################################

class Net(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.conv1 =  nn.Conv1d(args.bptt, 10, 10, stride = 1)
        self.maxpool =F.max_pool1d # nn.Conv2d(10, 10, 5, stride = 1)
        self.fc1 = nn.Linear(5*145*2, 5*30*4)
        self.fc2 = nn.Linear(5*30*4, 2)


    def forward(self, x):
        #print(x)
        x = self.embedding(x)
      #  print("Output from embedding layer", x.size())
        #x = emb.view(-1, 10, 10, args.emsize)
        #print("Output after resize layer", x.size())
        x = self.conv1(x)
       # print("Output after convolution layer", x.size())
        x = self.maxpool(x,2)
       # print("Output after maxpool layer", x.size())
        x = x.view(-1, 5*145*2)
       # print("Output after resize layer", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x)

model = Net(ntoken=len(vocab.keys()), ninp=args.emsize)
if args.cuda:
    model.cuda()
print(model)
#exit()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = F.cross_entropy

###############################################################################
# Training code
###############################################################################

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(trainDataset_loader):
        if args.cuda:
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #reconstruct_wrong_sent(data.data, output.data, target.data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainDataset_loader.dataset),
                100. * batch_idx / len(trainDataset_loader), loss.data[0]))

def test(epoch, dataset_loader):
    correct = 0
    test_loss = 0
    y_true =[]
    y_pred = []
    model.eval()
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if args.cuda:
                data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        #if epoch > 10:
        #    reconstruct_wrong_sent(data.data, output.data.cpu().numpy(), target.data.cpu().numpy())
        test_loss += loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        y_true.extend(target.data.cpu().numpy())
        y_pred.extend(pred.cpu().numpy().squeeze())
    print("Classification report")
    print(metrics.classification_report(y_true, y_pred))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(dataset_loader.dataset),
    100. * correct / len(dataset_loader.dataset)))

for epoch in range(args.epochs):
    train(epoch)
    test(epoch, val_dataset_loader)


test(1, test_dataset_loader)
