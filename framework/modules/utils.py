#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch import Tensor, nn
import random
from torch.nn import functional as F
from torch.optim import Optimizer


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")   
    return device

def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
        elif optimizer.lower() == "fedprox":
            optimizer = FedProxOptimizer(params, lr=lr)
            return optimizer
        elif optimizer.lower() == "scaffold":
            optimizer = SCAFFOLDOptimizer(params, lr=lr)
            return optimizer
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "bce_loss", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
        if loss in ["bpr", "bpr_loss"]:
            return bpr_loss
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try: 
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))       
    return loss_fn

def bpr_loss(pos_scores: Tensor,
    neg_scores: Tensor,)-> Tensor:
    bpr_loss = -torch.mean(torch.functional.F.logsigmoid(pos_scores - neg_scores))
    return bpr_loss

class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super(FedProxOptimizer, self).__init__(params, default)
    
    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is not None:
                    d_p = p.grad.data + group['mu'] * (p.data - global_params[g].data)
                    p.data.add_(d_p, alpha=-group['lr'])

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
    
    def step(self, global_c, client_c):
        for group in self.param_groups:
            for p, g_c, c_c in zip(group['params'], global_c, client_c):
                if p.grad is not None:
                    d_p = p.grad.data - c_c.data + g_c.data
                    p.data.add_(d_p, alpha=-group['lr'])

class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats
    
    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss
    
def get_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "swish":
            return Swish()
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

class Swish(nn.Module):
    def forward(self, input):
        return torch.sigmoid(input) * input

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer

def l2norm(x, dim=-1, eps=1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) -> Tensor:
        return l2norm(x, dim=self.dim, eps=self.eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight