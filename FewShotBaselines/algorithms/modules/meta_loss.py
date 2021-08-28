import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# def weight_init(module):
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight, gain=1.0)
#         if module.bias is not None:
#             module.bias.data.zero_()

# def weight_reset(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         m.reset_parameters()

class MetaLossNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, no_mean=False, no_hidden=False, neg_init=False, no_sp=False, act=nn.ReLU, fact=F.relu):
        super().__init__()
        self.num_layers = len(hidden_dim)
        self.no_mean = no_mean
        self.no_hidden = no_hidden
        self.no_sp = no_sp
        self.act = act
        self.fact = fact

        layers = []
        if not no_hidden:
            for i in range(len(hidden_dim)):
                if i == 0:
                    indim = in_dim
                else:
                    indim = hidden_dim[i-1]
                layers.append( nn.Linear(indim, hidden_dim[i], bias=False) )
                layers.append( self.act() )
        
        self.layers = nn.Sequential(*layers)
        if self.no_hidden:
            if no_sp:
                self.loss = nn.Sequential(nn.Linear(in_dim, 1, bias=False))
            else:
                self.loss = nn.Sequential(nn.Linear(in_dim, 1, bias=False), nn.Softplus())
            if neg_init:
                torch.nn.init.constant_(self.loss[0].weight, -1)
        else:
            if no_sp:
                self.loss = nn.Sequential(nn.Linear(hidden_dim[-1], 1, bias=False))
            else:
                self.loss = nn.Sequential(nn.Linear(hidden_dim[-1], 1, bias=False), nn.Softplus())

        print(self.layers)
        #self.reset()

    def forward(self, y, weights):
        for i in range(len(weights)):
            y = F.linear(y, weight=weights[i], bias=None)
            if not i == len(weights) - 1:
                y = self.fact(y)
        if not self.no_sp:
            y = F.softplus(y)
        if self.no_mean:
            return y
        return y.mean()