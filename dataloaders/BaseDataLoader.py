import torch
import pickle
import random
import copy
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp

from framework.modules.utils import get_device

class BaseDataLoader:
    def __init__(self):
        return NotImplementedError
    
    def get_dataloader(self, batch_size):
        return NotImplementedError
    
    def get_testdata(self, batch_size):
        return NotImplementedError
    
    def get_item_information(self, item):
        return NotImplementedError
    
    def get_user_information(self, user):
        return NotImplementedError
    
    def get_item_mean_feature(self):
        return NotImplementedError
    
    def get_user_mean_feature(self):
        return NotImplementedError
    
    def get_negative_samples(self, item_list, pos_list, neg_num):
        samples = list(set(item_list) - set(pos_list))
        if neg_num > len(samples):
            return samples
        return random.sample(samples, neg_num)
    
class BaseDataLoaderFL:
    def __init__(self):
        return NotImplementedError
    
    def get_traindata(self, user):
        return NotImplementedError
    
    def get_testdata(self, user):
        return NotImplementedError
    
    def get_item_information(self, item):
        return NotImplementedError
    
    def get_user_information(self, user):
        return NotImplementedError
    
    def get_item_feature(self):
        return NotImplementedError
    
    def get_user_feature(self):
        return NotImplementedError
    
    def get_negative_samples(self, item_list, pos_list, neg_num):
        samples = list(set(item_list) - set(pos_list))
        if neg_num > len(samples):
            return samples
        return random.sample(samples, neg_num)