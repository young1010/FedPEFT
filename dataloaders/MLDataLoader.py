#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle
import random
import copy
import gc

from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp

from framework.modules.utils import get_device
from dataloaders.BaseDataLoader import *

class MLDataLoader(BaseDataLoader):
    """
    ml-1m, Cen
    """
    device:str

    def __init__(self, data_dir, only_inter, task, device, user_num, item_num, embedding_dim_modal, **params):
        """
        only_inter: True, only intersections
        task: "rank" or "regression"
        """
        self.device = get_device(device)
        self.task = task.lower()
        self.graph_user = torch.load(data_dir + 'graph_user.pth')
        self.graph_item = torch.load(data_dir + 'graph_item.pth')
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim_modal = embedding_dim_modal

        # split data
        train_user = []
        train_item = []
        if task.lower() == "triple":
            self.ratings_triple= {'user':[], 'pos':[], 'neg':[]}
            self.ratings_test = {'user':[], 'item':[], 'label':[]}
            for user in self.graph_user.keys():
                neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                test_sample = self.graph_user[user][-1]
                self.graph_user[user] = self.graph_user[user][:-1]
                train_user.extend([user] * len(self.graph_user[user].tolist()))
                train_item.extend(self.graph_user[user].tolist())
                self.ratings_test['user'].extend([user for _ in range(101)])
                self.ratings_test['item'].extend(neg_items[-100:] + [test_sample])
                self.ratings_test['label'].extend([0.0 for _ in range(100)] + [1.0])
                self.ratings_triple['user'].extend([user for _ in range(len(self.graph_user[user]))])
                self.ratings_triple['pos'].extend(self.graph_user[user].tolist())
                self.ratings_triple['neg'].extend(random.choices(neg_items[:-100], k=len(self.graph_user[user])))
            self.ratings_test['user'] = torch.tensor(self.ratings_test['user'], dtype=torch.int64, device=self.device)
            self.ratings_test['item'] = torch.tensor(self.ratings_test['item'], dtype=torch.int64, device=self.device)
            self.ratings_test['label']= torch.tensor(self.ratings_test['label'], dtype=torch.float32, device=self.device)
            self.ratings_triple['user']= torch.tensor(self.ratings_triple['user'], dtype=torch.int64, device=self.device)
            self.ratings_triple['pos']= torch.tensor(self.ratings_triple['pos'], dtype=torch.int64, device=self.device)
            self.ratings_triple['neg']= torch.tensor(self.ratings_triple['neg'], dtype=torch.int64, device=self.device)
        else:
            self.ratings_training = {'user':[], 'item':[], 'label':[]}
            self.ratings_test = {'user':[], 'item':[], 'label':[]}
            if task.lower() == "rank":
                for user in self.graph_user.keys():
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user][-1]
                    self.graph_user[user] = self.graph_user[user][:-1]
                    train_user.extend([user] * len(self.graph_user[user].tolist()))
                    train_item.extend(self.graph_user[user].tolist())
                    self.ratings_test['user'].extend([user for _ in range(101)])
                    self.ratings_test['item'].extend(neg_items[-100:] + [test_sample])
                    self.ratings_test['label'].extend([0.0 for _ in range(100)] + [1.0])
                    self.ratings_training['user'].extend([user for _ in range(len(neg_items[:-100]) + (len(self.graph_user[user])))])
                    self.ratings_training['item'].extend(neg_items[:-100] + self.graph_user[user].tolist())
                    self.ratings_training['label'].extend([0.0 for _ in range(len(neg_items[:-100]))] + [1.0 for _ in range(len(self.graph_user[user]))])
            else:
                ratings_dict = torch.load(data_dir + 'ratings_fl.pth')
                for user in ratings_dict.keys():
                    test_num = round(len(ratings_dict[user]) * 0.2)
                    self.graph_user[user] = self.graph_user[user][:-test_num]
                    train_user.extend([user] * len(self.graph_user[user].tolist()))
                    train_item.extend(self.graph_user[user].tolist())
                    self.ratings_training['user'].extend(ratings_dict[user][:-test_num,0].tolist())
                    self.ratings_training['item'].extend(ratings_dict[user][:-test_num,1].tolist())
                    self.ratings_training['label'].extend(ratings_dict[user][:-test_num,2].tolist())
                    self.ratings_test['user'].extend(ratings_dict[user][-test_num:,0].tolist())
                    self.ratings_test['item'].extend(ratings_dict[user][-test_num:,1].tolist())
                    self.ratings_test['label'].extend(ratings_dict[user][-test_num:,2].tolist())
            self.ratings_test['user'] = torch.tensor(self.ratings_test['user'], dtype=torch.int64, device=self.device)
            self.ratings_test['item'] = torch.tensor(self.ratings_test['item'], dtype=torch.int64, device=self.device)
            self.ratings_test['label']= torch.tensor(self.ratings_test['label'], dtype=torch.float32, device=self.device)
            self.ratings_training['user']= torch.tensor(self.ratings_training['user'], dtype=torch.int64, device=self.device)
            self.ratings_training['item']= torch.tensor(self.ratings_training['item'], dtype=torch.int64, device=self.device)
            self.ratings_training['label']= torch.tensor(self.ratings_training['label'], dtype=torch.float32, device=self.device)
        self.aj_graph = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                      shape=(user_num, item_num))

        if not only_inter:
            self.item_dict = dict()
            item_attr = torch.load(data_dir + 'items.pth', map_location=self.device)
            for item in item_attr.keys():
                attrs = item_attr[item]    
                temp = torch.cat([attrs['Title']] + attrs['Genres'], dim=0).detach()
                self.item_dict[item] = [temp.to(torch.float32).to(self.device)]
                del attrs['Title'], attrs['Genres'], attrs, temp
            self.user_dict = dict()
            user_attr = torch.load(data_dir + 'users.pth', map_location=self.device)
            for user in user_attr.keys():
                attrs = user_attr[user]       
                self.user_dict[user] = [torch.cat([attrs['Gender'],attrs['Age'],attrs['Occupation']], dim=0).to(torch.float32).to(self.device)]

    def get_dataloader(self, batch_size):
        if self.task == "triple":
            dataset = TensorDataset(self.ratings_triple['user'], self.ratings_triple['pos'], self.ratings_triple['neg'])
        else:
            dataset = TensorDataset(self.ratings_training['user'], self.ratings_training['item'], self.ratings_training['label'])
        dataloader = DataLoader(dataset, batch_size, True)
        return dataloader
    
    def get_testdata(self, batch_size=101):
        dataset = TensorDataset(self.ratings_test['user'], self.ratings_test['item'], self.ratings_test['label'])
        dataloader = DataLoader(dataset, batch_size, False)
        return dataloader
    
    def get_item_information(self, item):
        return self.item_dict[item]
    
    def get_user_information(self, user):
        return self.user_dict[user]
    
    def get_item_mean_feature(self):
        items = []
        for item in range(self.item_num):
            if item in self.item_dict.keys():
                items.append(torch.mean(self.item_dict[item][0], dim=0))
            else:
                items.append(torch.zeros(self.embedding_dim_modal).to(self.device))   
        return torch.stack(items, dim=0)
    
    def get_user_mean_feature(self):
        users = []
        for user in range(self.user_num):
            if user in self.user_dict.keys():
                users.append(torch.mean(self.user_dict[user][0], dim=0))
            else:
                users.append(torch.zeros(self.embedding_dim_modal).to(self.device))   
        return torch.stack(users, dim=0)
    
    def get_item_mge_feature(self):
        return [self.get_item_mean_feature()]
    
    
    def get_aj_graph(self):
        return self.aj_graph

    
class MLDataLoaderFL(BaseDataLoaderFL):
    """
    ml-1m, FL
    """
    device:str

    def __init__(self, data_dir, only_inter, task, device, user_num, item_num, **params):
        """
        only_inter: True, only intersections
        task: "rank" or "regression"
        """
        self.device = get_device(device)
        self.task = task.lower()
        self.graph_user = torch.load(data_dir + 'graph_user.pth')
        self.graph_item = torch.load(data_dir + 'graph_item.pth')
        self.user_num = user_num
        self.item_num = item_num

        # splite data
        self.ratings_training = {}
        self.ratings_test = {}
        train_user = []
        train_item = []
        if task.lower() == "triple":
            self.ratings_triple = {}
            self.ratings_test = {}
            for user in self.graph_user.keys():
                self.ratings_triple[user] = {}
                self.ratings_test[user] = {}
                neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                test_sample = self.graph_user[user][-1]
                self.graph_user[user] = self.graph_user[user][:-1]
                train_user.extend([user] * len(self.graph_user[user].tolist()))
                train_item.extend(self.graph_user[user].tolist())
                self.ratings_test[user]['user'] = torch.tensor([user for _ in range(101)], dtype=torch.int64, device=self.device)
                self.ratings_test[user]['item']= torch.tensor(neg_items[-100:] + [test_sample], dtype=torch.int64, device=self.device)
                self.ratings_test[user]['label']= torch.tensor([0.0 for _ in range(100)] + [1.0], dtype=torch.float32, device=self.device)
                self.ratings_triple[user]['user']= torch.tensor([user for _ in range(len(self.graph_user[user]))], dtype=torch.int64, device=self.device)
                self.ratings_triple[user]['pos']= torch.tensor(self.graph_user[user].tolist(), dtype=torch.int64, device=self.device)
                self.ratings_triple[user]['neg']= torch.tensor(random.choices(neg_items[:-100], k=len(self.graph_user[user])), dtype=torch.int64, device=self.device)
        else:
            self.ratings_training = {}
            self.ratings_test = {}
            if task.lower() == "rank":
                for user in self.graph_user.keys():
                    self.ratings_training[user] = {}
                    self.ratings_test[user] = {}
                    neg_items = self.get_negative_samples(range(len(self.graph_item)), self.graph_user[user], 4*(len(self.graph_user[user])-1) + 100)
                    test_sample = self.graph_user[user][-1]
                    self.graph_user[user] = self.graph_user[user][:-1]
                    train_user.extend([user] * len(self.graph_user[user].tolist()))
                    train_item.extend(self.graph_user[user].tolist())
                    self.ratings_test[user]['user'] = torch.tensor([user for _ in range(101)], dtype=torch.int64, device=self.device)
                    self.ratings_test[user]['item']= torch.tensor(neg_items[-100:] + [test_sample], dtype=torch.int64, device=self.device)
                    self.ratings_test[user]['label']= torch.tensor([0.0 for _ in range(100)] + [1.0], dtype=torch.float32, device=self.device)
                    self.ratings_training[user]['user']= torch.tensor([user for _ in range(len(neg_items[:-100]) + (len(self.graph_user[user])))], dtype=torch.int64, device=self.device)
                    self.ratings_training[user]['item']= torch.tensor(neg_items[:-100] + self.graph_user[user].tolist(), dtype=torch.int64, device=self.device)
                    self.ratings_training[user]['label']= torch.tensor([0.0 for _ in range(len(neg_items[:-100]))] + [1.0 for _ in range(len(self.graph_user[user]))], dtype=torch.float32, device=self.device)
            else:
                ratings_dict = torch.load(data_dir + 'ratings_fl.pth')
                for user in ratings_dict.keys():
                    test_num = round(len(ratings_dict[user]) * 0.2)
                    self.ratings_training[user] = {}
                    self.ratings_test[user] = {}
                    self.graph_user[user] = self.graph_user[user][:-test_num]
                    train_user.extend([user] * len(self.graph_user[user].tolist()))
                    train_item.extend(self.graph_user[user].tolist())
                    self.ratings_training[user]['user'] = torch.tensor(ratings_dict[user][:-test_num,0].tolist(), dtype=torch.int64, device=self.device)
                    self.ratings_training[user]['item'] = torch.tensor(ratings_dict[user][:-test_num,1].tolist(), dtype=torch.int64, device=self.device)
                    self.ratings_training[user]['label'] = torch.tensor(ratings_dict[user][:-test_num,2].tolist(), dtype=torch.float32, device=self.device)
                    self.ratings_test[user]['user'] = torch.tensor(ratings_dict[user][-test_num:,0].tolist(), dtype=torch.int64, device=self.device)
                    self.ratings_test[user]['item'] = torch.tensor(ratings_dict[user][-test_num:,1].tolist(), dtype=torch.int64, device=self.device)
                    self.ratings_test[user]['label'] = torch.tensor(ratings_dict[user][-test_num:,2].tolist(), dtype=torch.float32, device=self.device)
        self.aj_graph = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                      shape=(user_num, item_num))
        
        if not only_inter:
            self.item_dict = torch.load(data_dir + 'items.pth', map_location=self.device)
            self.user_dict = torch.load(data_dir + 'users.pth', map_location=self.device)

    def get_user_topo(self, user):
        return torch.tensor(self.graph_user[user], dtype=torch.int64, device=self.device)
    
    def get_user_information(self, user):
        return self.user_dict[user]
    
    def get_item_feature(self):
        return self.item_dict
    
    def get_user_feature(self):
        return self.user_dict

    def get_traindata(self, user):
        if self.task.lower() == "triple":
            return self.ratings_triple[user]['user'], self.ratings_triple[user]['pos'], self.ratings_triple[user]['neg']
        else:
            return self.ratings_training[user]['user'], self.ratings_training[user]['item'], self.ratings_training[user]['label']
    
    def get_testdata(self, user):
        return self.ratings_test[user]['user'], self.ratings_test[user]['item'], self.ratings_test[user]['label']
    
    def get_aj_graph(self):
        return self.aj_graph
    
