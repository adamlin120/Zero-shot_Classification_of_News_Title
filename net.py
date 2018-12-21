import torch
from torch import nn
from torch.nn import functional as F

  
class simpleBiLinear(nn.Module):
    def __init__(self, config):
        super(simpleBiLinear, self).__init__()
        self.config = config
        self.embed_dim = self.config['embed_dim']
        self.hid_dim = self.config['hid_dim']
        self.dropout_rate = self.config['dropout']

        self.linear1 = nn.Linear(self.embed_dim, self.hid_dim, True)
        self.bn1 = nn.BatchNorm1d(self.hid_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate) 

        self.linear2 = nn.Linear(self.hid_dim, self.hid_dim, True)
        self.bn2 = nn.BatchNorm1d(self.hid_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.linear_out = nn.Linear(self.hid_dim, 1, True)

    # title: B x embed_dim
    # tag: B x embed_dim
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)

        outs = self.linear_out(x)
        outs = torch.sigmoid(outs)

        return outs
