#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle
import random
import copy

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from framework.modules.utils import get_device
from dataloaders.BaseDataLoader import *

class AmazonDataLoader(BaseDataLoader):
    """
    Amazon, Cen
    """
    device:str

    def __init__(self, data_dir, field, only_inter, task,  device, user_num, item_num, embedding_dim_modal, **params):
        """
        only_inter: True, only intersections
        task: "rank" or "regression"
        """
        self.device = get_device(device)
        self.task = task.lower()
        self.graph_user = torch.load(data_dir + 'graph/' + field + '_user.pth')
        self.graph_item = torch.load(data_dir + 'graph/' + field + '_item.pth')
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim_modal = embedding_dim_modal

        # split data
        train_user = []
        train_item = []
        if task.lower() == "triple":
            self.reviews_triple = {'user':[], 'pos':[], 'neg':[]}
            self.reviews_test = {'user':[], 'item':[], 'label':[]}
            for user in self.graph_user.keys():
                neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                test_sample = self.graph_user[user].pop()
                train_user.extend([user] * len(self.graph_user[user]))
                train_item.extend(self.graph_user[user])
                self.reviews_test['user'].extend([user for _ in range(101)])
                self.reviews_test['item'].extend(neg_items[-100:] + [test_sample])
                self.reviews_test['label'].extend([0.0 for _ in range(100)] + [1.0])
                self.reviews_triple['user'].extend([user for _ in range(len(self.graph_user[user]))])
                self.reviews_triple['pos'].extend(self.graph_user[user])
                self.reviews_triple['neg'].extend(random.choices(neg_items[:-100], k=len(self.graph_user[user])))
            self.reviews_test['user'] = torch.tensor(self.reviews_test['user'], dtype=torch.int64, device=self.device)
            self.reviews_test['item'] = torch.tensor(self.reviews_test['item'], dtype=torch.int64, device=self.device)
            self.reviews_test['label']= torch.tensor(self.reviews_test['label'], dtype=torch.float32, device=self.device)
            self.reviews_triple['user']= torch.tensor(self.reviews_triple['user'], dtype=torch.int64, device=self.device)
            self.reviews_triple['pos']= torch.tensor(self.reviews_triple['pos'], dtype=torch.int64, device=self.device)
            self.reviews_triple['neg']= torch.tensor(self.reviews_triple['neg'], dtype=torch.int64, device=self.device)
        else:
            self.reviews_training = {'user':[], 'item':[], 'label':[]}
            self.reviews_test = {'user':[], 'item':[], 'label':[]}
            if task.lower() == "rank":
                for user in self.graph_user.keys():
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user].pop()
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    self.reviews_test['user'].extend([user for _ in range(101)])
                    self.reviews_test['item'].extend(neg_items[-100:] + [test_sample])
                    self.reviews_test['label'].extend([0.0 for _ in range(100)] + [1.0])
                    self.reviews_training['user'].extend([user for _ in range(len(neg_items[:-100]) + len(self.graph_user[user]))])
                    self.reviews_training['item'].extend(neg_items[:-100] + self.graph_user[user])
                    self.reviews_training['label'].extend([0.0 for _ in range(len(neg_items[:-100]))] + [1.0 for _ in range(len(self.graph_user[user]))])
            else:
                reviews_dict = torch.load(data_dir + '5-core_clip/' + field + '_fl.pth')
                for user in reviews_dict.keys():
                    test_num = round(len(reviews_dict[user]) * 0.2)
                    self.graph_user[user] = self.graph_user[user][:-test_num]
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    for review in reviews_dict[user][:-test_num]:
                        self.reviews_training['user'].append(review['reviewerID'])
                        self.reviews_training['item'].append(review['asin'])
                        self.reviews_training['label'].append(review['overall'])
                    for review in reviews_dict[user][-test_num:]:
                        self.reviews_test['user'].append(review['reviewerID'])
                        self.reviews_test['item'].append(review['asin'])
                        self.reviews_test['label'].append(review['overall'])
            self.reviews_test['user'] = torch.tensor(self.reviews_test['user'], dtype=torch.int64, device=self.device)
            self.reviews_test['item'] = torch.tensor(self.reviews_test['item'], dtype=torch.int64, device=self.device)
            self.reviews_test['label']= torch.tensor(self.reviews_test['label'], dtype=torch.float32, device=self.device)
            self.reviews_training['user']= torch.tensor(self.reviews_training['user'], dtype=torch.int64, device=self.device)
            self.reviews_training['item']= torch.tensor(self.reviews_training['item'], dtype=torch.int64, device=self.device)
            self.reviews_training['label']= torch.tensor(self.reviews_training['label'], dtype=torch.float32, device=self.device)
        self.aj_graph = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                      shape=(user_num, item_num))
        
        if not only_inter:
            self.item_dict = dict()
            item_attr = torch.load(data_dir + 'meta_clip/meta_' + field + '.pth', map_location=self.device)
            for item in item_attr.keys():
                attrs = item_attr[item]
                if 'image' in attrs:
                    img = attrs['image'].to(torch.float32).to(self.device)
                else:
                    img = torch.zeros(1,self.embedding_dim_modal).to(self.device)
                txt = [attrs[attr] for attr in ['title', 'description', 'categories', 'brand', 'feature'] if attr in attrs]
                if len(txt) > 0:
                    txt = torch.cat(txt).to(torch.float32).to(self.device)
                else:
                    txt = torch.zeros(1,self.embedding_dim_modal).to(self.device)
                self.item_dict[item] = [img,txt]

    def get_dataloader(self, batch_size):
        if self.task == "triple":
            dataset = TensorDataset(self.reviews_triple['user'], self.reviews_triple['pos'], self.reviews_triple['neg'])
        else:
            dataset = TensorDataset(self.reviews_training['user'], self.reviews_training['item'], self.reviews_training['label'])
        dataloader = DataLoader(dataset, batch_size, True)
        return dataloader
    
    def get_testdata(self, batch_size=101):
        dataset = TensorDataset(self.reviews_test['user'], self.reviews_test['item'], self.reviews_test['label'])
        dataloader = DataLoader(dataset, batch_size, False)
        return dataloader
    
    def get_item_information(self, item):
        return self.item_dict[item]
    
    def get_user_information(self, user):
        return None
    
    def get_item_mean_feature(self):
        items = []
        for item in range(self.item_num):
            if item in self.item_dict.keys():
                items.append(torch.mean(torch.cat([self.item_dict[item][0], self.item_dict[item][1]], dim=0), dim=0))
            else:
                items.append(torch.zeros(self.embedding_dim_modal).to(self.device))   
        return torch.stack(items, dim=0)
    
    def get_user_mean_feature(self):
        None

    def get_item_mge_feature(self):
        items = {i: list() for i in range(2)}
        for item in range(self.item_num):
            if item in self.item_dict.keys():
                items[0].append(torch.mean(self.item_dict[item][0], dim=0))
                items[1].append(torch.mean(self.item_dict[item][1], dim=0))
            else:
                items[0].append(torch.zeros(self.embedding_dim_modal).to(self.device))   
                items[1].append(torch.zeros(self.embedding_dim_modal).to(self.device))   
        return [torch.stack(items[0], dim=0), torch.stack(items[1], dim=0)]
    
    def get_aj_graph(self):
        return self.aj_graph
    
class AmazonDataLoaderFL(BaseDataLoaderFL):
    """
    Amazon, FL
    """
    device:str

    def __init__(self, data_dir, meta_dir, field, only_inter, task, device, user_num, item_num,  **params):
        """
        only_inter: True, only intersections
        task: "rank" or "regression"
        """
        self.device = get_device(device)
        self.task = task.lower()
        self.graph_user = torch.load(data_dir + 'graph/' + field + '_user.pth')
        self.graph_item = torch.load(data_dir + 'graph/' + field + '_item.pth')
        self.user_num = user_num
        self.item_num = item_num

        # splite data
        train_user = []
        train_item = []
        if task.lower() == "triple":
            self.reviews_triple = {}
            self.reviews_test = {}
            if self.item_num <= 101:
                for user in self.graph_user.keys():
                    self.reviews_triple[user] = {}
                    self.reviews_test[user] = {}
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user].pop()
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    neg_len = len(neg_items)
                    self.reviews_test[user]['user'] = torch.tensor([user for _ in range(neg_len + 1)], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['item']= torch.tensor(neg_items + [test_sample], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['label']= torch.tensor([0.0 for _ in range(neg_len)] + [1.0], dtype=torch.float32, device=self.device)
                    self.reviews_triple[user]['user']= torch.tensor([user for _ in range(len(self.graph_user[user]))], dtype=torch.int64, device=self.device)
                    self.reviews_triple[user]['pos']= torch.tensor(self.graph_user[user], dtype=torch.int64, device=self.device)
                    self.reviews_triple[user]['neg']= torch.tensor(random.choices(neg_items, k=len(self.graph_user[user])), dtype=torch.int64, device=self.device)
            else:
                for user in self.graph_user.keys():
                    self.reviews_triple[user] = {}
                    self.reviews_test[user] = {}
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user].pop()
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    self.reviews_test[user]['user'] = torch.tensor([user for _ in range(101)], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['item']= torch.tensor(neg_items[-100:] + [test_sample], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['label']= torch.tensor([0.0 for _ in range(100)] + [1.0], dtype=torch.float32, device=self.device)
                    self.reviews_triple[user]['user']= torch.tensor([user for _ in range(len(self.graph_user[user]))], dtype=torch.int64, device=self.device)
                    self.reviews_triple[user]['pos']= torch.tensor(self.graph_user[user], dtype=torch.int64, device=self.device)
                    self.reviews_triple[user]['neg']= torch.tensor(random.choices(neg_items[:-100], k=len(self.graph_user[user])), dtype=torch.int64, device=self.device)
        else:
            self.reviews_training = {}
            self.reviews_test = {}
            if task.lower() == "rank":
                for user in self.graph_user.keys():
                    self.reviews_training[user] = {}
                    self.reviews_test[user] = {}
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user].pop()
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    self.reviews_test[user]['user'] = torch.tensor([user for _ in range(101)], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['item']= torch.tensor(neg_items[-100:] + [test_sample], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['label']= torch.tensor([0.0 for _ in range(100)] + [1.0], dtype=torch.float32, device=self.device)
                    self.reviews_training[user]['user']= torch.tensor([user for _ in range(len(neg_items[:-100]) + (len(self.graph_user[user])))], dtype=torch.int64, device=self.device)
                    self.reviews_training[user]['item']= torch.tensor(neg_items[:-100] + self.graph_user[user], dtype=torch.int64, device=self.device)
                    self.reviews_training[user]['label']= torch.tensor([0.0 for _ in range(len(neg_items[:-100]))] + [1.0 for _ in range(len(self.graph_user[user]))], dtype=torch.float32, device=self.device)
            else:
                reviews_dict = torch.load(data_dir + '5-core_clip/' + field + '_fl.pth')
                for user in reviews_dict.keys():
                    test_num = round(len(reviews_dict[user]) * 0.2)
                    self.graph_user[user] = self.graph_user[user][:-test_num]
                    train_user.extend([user] * len(self.graph_user[user]))
                    train_item.extend(self.graph_user[user])
                    self.reviews_training[user] = {'user':[], 'item':[], 'label':[]}
                    self.reviews_test[user] = {'user':[], 'item':[], 'label':[]}
                    for review in reviews_dict[user][:-test_num]:
                        self.reviews_training[user]['user'].append(review['reviewerID'])
                        self.reviews_training[user]['item'].append(review['asin'])
                        self.reviews_training[user]['label'].append(review['overall'])
                    for review in reviews_dict[user][-test_num:]:
                        self.reviews_test[user]['user'].append(review['reviewerID'])
                        self.reviews_test[user]['item'].append(review['asin'])
                        self.reviews_test[user]['label'].append(review['overall'])
                    self.reviews_test[user]['user'] = torch.tensor(self.reviews_test[user]['user'], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['item'] = torch.tensor(self.reviews_test[user]['item'], dtype=torch.int64, device=self.device)
                    self.reviews_test[user]['label']= torch.tensor(self.reviews_test[user]['label'], dtype=torch.float32, device=self.device)
                    self.reviews_training[user]['user']= torch.tensor(self.reviews_training[user]['user'], dtype=torch.int64, device=self.device)
                    self.reviews_training[user]['item']= torch.tensor(self.reviews_training[user]['item'], dtype=torch.int64, device=self.device)
                    self.reviews_training[user]['label']= torch.tensor(self.reviews_training[user]['label'], dtype=torch.float32, device=self.device)
        self.aj_graph = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                      shape=(user_num, item_num))

        if not only_inter:
            self.item_dict = torch.load(meta_dir + field + '.pth', map_location=self.device)
    
    def get_user_topo(self, user):
        return torch.tensor(self.graph_user[user], dtype=torch.int64, device=self.device)

    def get_traindata(self, user):
        if self.task == "triple":
            return self.reviews_triple[user]['user'], self.reviews_triple[user]['pos'], self.reviews_triple[user]['neg']
        else:
            return self.reviews_training[user]['user'], self.reviews_training[user]['item'], self.reviews_training[user]['label']
    
    def get_testdata(self, user):
        return self.reviews_test[user]['user'], self.reviews_test[user]['item'], self.reviews_test[user]['label']
    
    def get_item_information(self, item):
        return self.item_dict[item]
    
    def get_user_information(self, user):
        return None

    def get_item_feature(self):
        return self.item_dict

    def get_aj_graph(self):
        return self.aj_graph
    