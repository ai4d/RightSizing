import torch
import torch.nn as nn

""" Two papers about the loss function 
1. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
2. Auxiliary Tasks in Multi-task Learning
"""


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        losses = []
        for i, loss in enumerate(x):
            loss = 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += loss
            losses.append(loss)
        return loss_sum, losses