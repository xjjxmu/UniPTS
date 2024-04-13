from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

from args import args as parser_args
import numpy as np
DenseConv = nn.Conv2d

class GetWeightSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, prune_rate, decay = 0.00005):
        ctx.save_for_backward(scores)
        
        out = scores.clone()
        _, idx = scores.abs().flatten().sort()
        j = int(prune_rate * scores.numel())
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        ctx.mask = out 
        ctx.decay = decay
        return out * scores

    @staticmethod
    def backward(ctx, g):
        weight, = ctx.saved_tensors
        return g + ctx.decay * (1 - ctx.mask) * weight, None, None, None
    
    
class SparseWeightConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_rate = 0

    def forward(self, x):
        w = GetWeightSubnet.apply(self.weight, self.prune_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, f=torch.sigmoid,sparse = None):
        w = GetWeightSubnet.apply(self.weight, self.prune_rate)
        temp = w.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item() * 100), temp.numel()

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate