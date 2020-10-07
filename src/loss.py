import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class criterion_margin_focal_binary_cross_entropy(nn.Module):
    def __init__(self):
        super(criterion_margin_focal_binary_cross_entropy, self).__init__()
        self.weight_pos=2
        self.weight_neg=1
        self.gamma=2
        self.margin=0.2
        self.em = np.exp(self.margin)
    def forward(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1)
        log_pos = -F.logsigmoid( logit)
        log_neg = -F.logsigmoid(-logit)

        log_prob = truth*log_pos + (1-truth)*log_neg
        prob = torch.exp(-log_prob)
        self.margin = torch.log(self.em +(1-self.em)*prob)

        weight = truth*self.weight_pos + (1-truth)*self.weight_neg
        loss = self.margin + weight*(1 - prob) ** self.gamma * log_prob

        loss = loss.mean()
        return loss