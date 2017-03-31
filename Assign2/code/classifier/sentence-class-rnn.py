__author__ = 'bharathipriyaa'
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import argparse
import torch, pickle, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import rnn_classifier as rnnmodel

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--nhid', type=int, default=50,
                    help='humber of hidden units per layer')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--emsize', type=int, default=30,
                    help='size of word embeddings')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size')
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
    l = len(data_list)
    r = (int) (0.7 *l) 
    train_dataset = torch.utils.data.TensorDataset(data_list[1:r], labels_list[1:r])
    valid_dataset = torch.utils.data.TensorDataset(data_list[r:l], labels_list[r:l])
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset_loader, val_dataset_loader

vocab = pickle.load(open("vocab.p", "rb"))
trainDataset_loader, val_dataset_loader = create_dataset()

###############################################################################
# Build the model
###############################################################################

ntokens = len(vocab.keys())
#model = rnnmodel.RNNModel( ntoken = ntokens, input_size=args.emsize, hidden_size= args.nhid, output_size= 5, batchsize = args.batch_size)

model = rnnmodel.RNNModelComplex(args.model, ntoken = ntokens, ninp = args.emsize, nhid=args.nhid, nlayers=args.nlayers,
                                batchsize = args.batch_size, bptt= args.bptt)

if args.cuda:
    model.cuda()

'''
input = Variable(torch.Tensor(64,100))
hidden = model.init_hidden(64)
output = model(input, hidden)
print(output.size())
'''


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = F.nll_loss

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(epoch):
    model.train()
    train_loss = 0
    hidden = model.init_hidden(args.bptt)
    for batch_idx, (data, target) in enumerate(trainDataset_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden)
        #print(output, target)
        #print(output.size(), target.size())
        loss = criterion(output, target)
        loss.backward()

        #### Clip gradients
        clipped_lr = args.lr * clip_gradient(model, args.clip)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)

        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainDataset_loader.dataset),
                100. * batch_idx / len(trainDataset_loader), loss.data[0]))

def test(epoch):

    test_loss = 0
    hidden = model.init_hidden(args.bptt)
    correct = 0
    y_true = []
    y_pred = []	
    for batch_idx, (data, target) in enumerate(val_dataset_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, target)
        loss.backward()
        test_loss += loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
	y_true.extend(target.data.cpu().numpy())
        y_pred.extend(pred.cpu().numpy().squeeze())
    print("Classification report") 
    print(metrics.classification_report(y_true, y_pred))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(val_dataset_loader.dataset),
    100. * correct / len(val_dataset_loader.dataset)))

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)

