import torch.nn as nn
import copy

class ClientBase:
    def __init__(self, client_id, model:nn.Module):
        self.client_id = client_id
        self.model = model

    def load_client(self, client_id):
        self.client_id = client_id
        
    def model(self):
        return self.model

    def load_model(self, model):
        self.model.load_weights(model)
        self.model.to(self.model.device)

    def upload_model(self):
        return self.model.state_dict()
    
    def local_train(self):
        raise NotImplementedError
    
    def local_data_num(self):
        return self.__local_data_num
    
    def update_client_params(self, global_model, global_c):
        for client_m, latest_global_m, global_m in zip(self.model.parameters(), self.client_global_model.parameters(),
                                                       global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()
        
        self.global_c = global_c
    
    def update_c(self, num_batches):
        delta_c = copy.deepcopy(self.client_c)
        temp_client_c = copy.deepcopy(self.client_c)
        for ci, c, global_m, client_m in zip(self.client_c, self.global_c, self.client_global_model.parameters(),
                                             self.model.parameters()):
            ci.data = ci.data - c.data + (global_m.data - client_m.data) / (num_batches * self.learn_rate)
        
        for d_c, temp_c, ci_start in zip(delta_c, temp_client_c, self.client_c):
            d_c.data = ci_start.data - temp_c.data
        
        return delta_c
