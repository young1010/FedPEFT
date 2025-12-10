from collections import OrderedDict
import random

class ServerBase:
    def __init__(self, model):
        self.model = model
              
    def aggregation(self):
        raise NotImplementedError

    def distribute_model(self):
        return self.model.state_dict()
    
    def select_clients(self, user_num, client_num):
        return random.sample(range(user_num), client_num)

    def validation(self):
        raise NotImplementedError
    
    def count_parameters(self):
        return self.model.count_parameters()