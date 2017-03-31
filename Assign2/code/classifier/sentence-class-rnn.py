__author__ = 'bharathipriyaa'


import argparse
import torch, pickle
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
parser.add_argument('--nhid', type=int, default=50,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lr', type=float, default=20,
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
    train_dataset = torch.utils.data.TensorDataset(data_list, labels_list)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataset_loader

vocab = pickle.load(open("vocab.p", "rb"))
trainDataset_loader = create_dataset()

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



def train(epoch):
    model.train()
    train_loss = 0
    hidden = model.init_hidden(args.bptt)
    for batch_idx, (data, target) in enumerate(trainDataset_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        model.zero_grad()
        output, hidden = model(data, hidden)
        print(output.size(), target.size())
        loss = criterion(output, target)
        loss.backward()

        #### Clip gradients
        clipped_lr = lr * clip_gradient(model, args.clip)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)


        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainDataset_loader.dataset),
                100. * batch_idx / len(trainDataset_loader), loss.data[0]))

for epoch in range(10):
    train(epoch)

