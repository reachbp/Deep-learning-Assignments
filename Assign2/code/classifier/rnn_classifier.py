__author__ = 'bharathipriyaa'

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, ntoken, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.encoder = nn.Embedding(ntoken, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        print("Checkpoint 1 ", combined.size())
        hidden = self.i2h(combined)
        print("Checkpoint 2 ", hidden.size())
        output = self.i2o(combined)
        print("Checkpoint 3 ", output.size())
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))