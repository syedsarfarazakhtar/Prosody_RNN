import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_size)


    def forward(self, x):

        out, hidden = self.rnn(x)
        out = out.squeeze(0)
        out1 = self.fc1(out)

        return out1, hidden



