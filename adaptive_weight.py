"""
This is a library for applying adaptive weight to some cases.
-- AdaptiveWeight is for multi-loss.
-- AdaptiveParam is Energy Coefficient for SATNet.
-- More to be added.

Author: Huilin Zhang (Fanning)
"""


import torch
import torch.nn as nn


class AdaptiveWeight(nn.Module):

    def __init__(self, weight_num=2):
        super(AdaptiveWeight, self).__init__()
        # self.device = device
        self.weight_num = weight_num
        self.weight = nn.Parameter(torch.ones(weight_num))

        # self.weight_init = weight_init
        # self.weight2 = nn.Parameter(torch.cuda.FloatTensor([weight_init]))
        # self.weight3 = nn.Parameter(torch.cuda.FloatTensor([weight_init]))

    def forward(self, obj1, obj2):

        # restrict to non-negative weight by weight^2
        w1 = self.weight[0]*self.weight[0]
        w2 = self.weight[1]*self.weight[1]

        weighted_obj = w1 * obj1 + w2 * obj2

        return weighted_obj, w1, w2


class AdaptiveParam(nn.Module):

    def __init__(self):
        super(AdaptiveParam, self).__init__()

        self.raw_alpha = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

    def forward(self, x):

        # transform parameter alpha to (0,1) through sigmoid func.
        alpha = torch.sigmoid(self.raw_alpha)

        return alpha, alpha * x
