import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseWeightedLoss
import pdb


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


class EvidenceLoss(BaseWeightedLoss):
    def __init__(self, num_classes, evidence='exp', loss_type='log'):
        super().__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.loss_type = loss_type
        self.eps = 1e-10

    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        return loglikelihood_err, loglikelihood_var

    def edl_loss(self, func, y, alpha, annealing_coef, target):
        losses = {}
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S

        g = y
        
        p = func(S) - func(alpha)

        A = torch.sum(g * p, dim=1, keepdim=True) 

        losses.update({'loss_cls': A})

        losses.update({'lambda': annealing_coef})

        return losses

    def _forward(self, output, target, **kwargs):

        # get evidence
        if self.evidence == 'relu':
            evidence = relu_evidence(output)
        elif self.evidence == 'exp':
            evidence = exp_evidence(output)
        elif self.evidence == 'softplus':
            evidence = softplus_evidence(output)
        else:
            raise NotImplementedError
        alpha = evidence + 1

        y = target

        annealing_coef = 1.0

        # compute the EDL loss
        if self.loss_type == 'log':
            results = self.edl_loss(torch.log, y, alpha, annealing_coef, target)
        elif self.loss_type == 'digamma':
            results = self.edl_loss(torch.digamma, y, alpha, annealing_coef, target)
        else:
            raise NotImplementedError


        uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        results.update({'uncertainty': uncertainty})

        return results

    def get_predictions(self, x):
        evidence = exp_evidence(x)
        S = evidence + torch.ones_like(x)
        p = evidence / torch.sum(S, dim=-1, keepdim=True)
        u = self.num_classes / torch.sum(S, dim=-1, keepdim=True)
        return p, u
    
    def get_alpha(self, x):
        evidence = exp_evidence(x)
        S = evidence + torch.ones_like(x)
        return S

