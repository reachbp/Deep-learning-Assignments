__author__ = 'bharathipriyaa'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
class RNNModel(nn.Module):
    def __init__(self, ntoken, input_size, hidden_size, output_size, batchsize):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        emb = self.encoder(input)
        print("Embedding size", emb.size())
        print("Size of input and hidden", input.size(), hidden.size())
        combined = torch.cat((input, hidden), 1)
        print("Checkpoint 1 ", combined.size(), self.input_size + self.hidden_size)
        hidden = self.i2h(combined)
        print("Checkpoint 2 ", hidden.size())
        output = self.i2o(combined)
        print("Checkpoint 3 ", output.size())
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batchsize):
        return Variable(torch.zeros(batchsize, self.hidden_size))
'''
class RNNModelComplex(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, batchsize, bptt):
        super(RNNModelComplex, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.decoder1 = nn.Linear(nhid*bptt, 5000)
        self.decoder2 = nn.Linear(5000, 5)
        print("Decoder size has to be", batchsize*nhid*bptt)
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)

        output, hidden = self.rnn(emb, hidden)
        #print("Check point 1", output.size())
        output = output.view(-1,output.size(1)* output.size(2))
        #print("Check point 2", output.size())
        decoded = F.relu(self.decoder1(output))
        decoded = F.sigmoid(self.decoder2(decoded))
        #print("Check point 3", decoded.size())
        return F.log_softmax(decoded), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
