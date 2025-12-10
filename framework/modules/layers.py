#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
from torch import nn
from framework.modules.utils import L2NormalizationLayer, get_activation
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans
from collections import Counter
from einops import rearrange
from torch.nn import functional as F

class MLP_Block(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None, 
                 dropout_rates=0.0,
                 batch_norm=False, 
                 bn_only_once=False, # Set True for inference speed up
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and not bn_only_once:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.mlp(inputs)

class SENet_Block(nn.Module):
    def __init__(self, 
                 input_dim,
                 reduction_ratio=16,
                 activation="ReLU",):
        super(SENet_Block, self).__init__()
        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        self.activation = get_activation(activation)
        
        self.fc1 = nn.Linear(input_dim, input_dim * reduction_ratio, bias=True)
        self.fc2 = nn.Linear(input_dim * reduction_ratio, input_dim, bias=True)
    
    def forward(self, x):
        x = x.unsqueeze(0) if len(x.shape) == 2 else x
        se = torch.mean(x, dim=1,)
        se = self.fc1(se)
        se = self.activation(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        se = se.unsqueeze(1)
        return x * se.expand_as(x)

class PQ(nn.Module):
    def __init__(self, embedding_dim, embedding_dim_latent, mc_size, device):
        super(__class__, self).__init__()
        self.d_size = int(embedding_dim/embedding_dim_latent)
        self.pq_dim = embedding_dim_latent
        self.code_embedding_tables = nn.ModuleList([nn.Embedding(mc_size, self.pq_dim, device=device) for _ in range(self.d_size)])

    def forward_pre(self, latent):
        sub_vectors = torch.split(latent, self.pq_dim, dim=0)
        quantize_book = list()
        embeddings = list()
        for i in range(self.d_size):
            sub_vector = sub_vectors[i].reshape(1,-1)
            quantize_book.append(torch.sum(torch.pow(sub_vector-self.code_embedding_tables[i].weight, 2), dim=-1).argmin())
            embeddings.append(self.code_embedding_tables[i](quantize_book[i]))

        return torch.mean(torch.stack(embeddings, dim=0), dim=0), quantize_book
    
    def forward(self, code_list):
        embeddings = list()
        for i in range(self.d_size):
            embeddings.append(self.code_embedding_tables[i](code_list[i]))
        return torch.mean(torch.stack(embeddings, dim=0), dim=0)
    
class HashEmb(nn.Module):
    def __init__(self, 
                 mc_size,
                 cb_size,
                 embedding_dim_latent,
                 device, 
                 *args, **kwargs):
        super(__class__, self).__init__()
        self.mc_size= mc_size
        self.cb_size = cb_size
        self.device = device
        self.code_list = list()
        self.code_embedding_tables = nn.Embedding(mc_size, embedding_dim_latent, device=device)
        self.hash_param = random.sample(range(1, 100), 2 * self.cb_size)

    def forward(self, item):
        return torch.stack([self.code_embedding_tables(self.code_list[i][item]) for i in range(self.cb_size)], dim=-1)

    def pre_train(self, input):
        ids = list(range(input.shape[0]))
        for i in range(self.cb_size):
            index = [((x * self.hash_param[i * 2] + self.hash_param[i * 2 + 1]) % 4096) % self.mc_size for x in ids]
            self.code_list.append(torch.tensor(index, device=self.device))

    def get_code_list(self):
        return self.code_list
    
    def set_code_list(self, code_list, device):
        self.code_list = list()
        for i in range(self.cb_size):
            self.code_list.append(code_list[i].to(device=device))

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()
   
class RPQ(nn.Module):
    code_list:list
    def __init__(self, latent_size, device, mc_size=256, cb_size=3, adapt=False, norm=False):
        super(__class__, self).__init__()
        self.mc_size= mc_size
        self.cb_size = cb_size
        self.code_list = list()
        self.code_embedding_tables = nn.ModuleList([nn.Embedding(mc_size, latent_size, device=device) for _ in range(self.cb_size)])
        self.adapt = adapt
        self.norm = norm
        self.adapt_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=False) if self.adapt else nn.Identity(),
            L2NormalizationLayer(dim=-1) if self.norm else nn.Identity()
        ) for _ in range(self.cb_size)])
        for emb in self.code_embedding_tables:
            nn.init.uniform_(emb.weight)

    def get_embedding(self, i, idx):
        return self.adapt_layers[i](self.code_embedding_tables[i](idx))
    
    def stop_gradient(self):
        for i in range(self.cb_size):
            self.code_embedding_tables[i].weight.requires_grad = False

    def pre_train(self, latent):
        self.adpt = nn.Embedding(latent.shape[0], self.cb_size, device=latent.device)
        self.code_list = list()
        for i in range(self.cb_size):
            kmeans = KMeans(self.mc_size, init='k-means++', n_init=10, max_iter=1000000)
            kmeans.fit(latent.detach().cpu())
            centers = kmeans.cluster_centers_
            centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
            self.code_embedding_tables[i].weight = centers
            self.code_embedding_tables[i] = self.code_embedding_tables[i].to(latent.device)
            codebook = self.adapt_layers[i](self.code_embedding_tables[i].weight)
            distances = torch.cdist(latent, codebook, p=2)
            dis, index = distances.min(dim=1) # index:tensor
            self.code_list.append(index)
            emb = self.get_embedding(i, index)
            # emb_out = latent + (emb-latent).detach()
            emb_out = efficient_rotation_trick_transform(
                    latent / (latent.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    latent
                )
            latent = latent-emb_out

    def train_step(self, latent):
        self.count_collisions()
        embeddings = list()
        embeddings_d = list()
        residual = list()
        self.code_list = list()

        for i in range(self.cb_size):
            codebook = self.adapt_layers[i](self.code_embedding_tables[i].weight)
            distances = torch.cdist(latent, codebook, p=2)
            dis, index = distances.min(dim=1) # index:tensor
            self.code_list.append(index)
            emb = self.get_embedding(i, index)
            # emb_out = latent + (emb-latent).detach()
            emb_out = efficient_rotation_trick_transform(
                    latent / (latent.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    latent
                )
            residual.append(latent)
            embeddings.append(emb)
            embeddings_d.append(emb_out)
            latent = latent-emb_out
        return torch.sum(torch.stack(embeddings_d, dim=0), dim=0), embeddings, residual
    
    def count_collisions(self):
        x = torch.stack(self.code_list, dim=0).t()
        row_keys = [row.numpy().tobytes() for row in x.cpu()]
        counter = Counter(row_keys)
        unique = len(counter)
        return unique

    def get_code_list(self):
        return self.code_list
    
    def set_code_list(self, code_list, device):
        self.code_list = list()
        for i in range(self.cb_size):
            self.code_list.append(code_list[i].to(device=device))
        
    def forward(self, item):
        return torch.stack([self.code_embedding_tables[i](self.code_list[i][item]) for i in range(self.cb_size)], dim=-1)