import torch
from torch import nn
from torch.nn import functional as F


class simpleBiLinear(nn.Module):
    def __init__(self, config):
        super(simpleBiLinear, self).__init__()
        self.config = config
        self.embed_dim = self.config['embed_dim']
        self.hid_dim = self.config['hid_dim']
        self.dropout = self.config['dropout']

        self.linear1 = nn.Linear(self.embed_dim, self.hid_dim, True)
        self.bn1 = nn.BatchNorm1d(self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, 1, True)

    # title: B x embed_dim
    # tag: B x embed_dim
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.linear1(x)))
        outs = torch.sigmoid(self.linear2(x))

        return outs