"""
Zhang C, Long G, Zhou T, et al. Dual personalization on federated recommendation
[C]//Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence. 2023: 4558-4566.

@inproceedings{DBLP:conf/ijcai/ZhangL0YZZY23,
  author       = {Chunxu Zhang and
                  Guodong Long and
                  Tianyi Zhou and
                  Peng Yan and
                  Zijian Zhang and
                  Chengqi Zhang and
                  Bo Yang},
  title        = {Dual Personalization on Federated Recommendation},
  booktitle    = {Proceedings of the Thirty-Second International Joint Conference on
                  Artificial Intelligence, {IJCAI} 2023, 19th-25th August 2023, Macao,
                  SAR, China},
  pages        = {4558--4566},
  publisher    = {ijcai.org},
  year         = {2023},
  url          = {https://doi.org/10.24963/ijcai.2023/507},
  doi          = {10.24963/IJCAI.2023/507},
  timestamp    = {Tue, 15 Oct 2024 16:43:28 +0200},
  biburl       = {https://dblp.org/rec/conf/ijcai/ZhangL0YZZY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

https://github.com/Zhangcx19/IJCAI-23-PFedRec
"""

from collections import OrderedDict
import copy
import numpy as np
import torch.nn as nn
import torch
import logging
from framework.fed.client import ClientBase
from framework.fed.server import ServerBase
from framework.modules.models import BaseModel, AE, PQ_VAE, RPQ_VAE
from framework.modules.layers import MLP_Block, PQ, RPQ
from dataloaders.BaseDataLoader import *
from framework.utils import calculate_model_size
from thop import profile

class model(BaseModel):
    def __init__(self, 
                item_num, 
                embedding_dim, 
                hidden_activations, 
                hidden_units, 
                latent_dim,
                learning_rate, 
                optimizer,
                loss_fn,
                task,
                device=-1, embedding_regularizer=None, net_regularizer=None, metrics=None, *args, **kwargs):
        super(__class__, self).__init__(device, embedding_regularizer, net_regularizer, metrics)

        self.embedding_item = nn.Sequential(OrderedDict([('emb', nn.Embedding(item_num, latent_dim)), 
                                                      ('linear', nn.Linear(latent_dim, embedding_dim, bias=False)),
                                                      ]))
        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim,)
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.task = task
        self.mlp = MLP_Block(input_dim = embedding_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=.5,
                             )
        self.output_activation= nn.Sigmoid()
        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self, ):
        nn.init.normal_(self.embedding_item.emb.weight, std=0.1)
        nn.init.zeros_(self.embedding_item.linear.weight)
        nn.init.normal_(self.embedding_p.weight, std=0.1)

    def emb_item(self, item_id):
        return self.embedding_p(item_id) + self.embedding_item(item_id)

    def emb_item_c(self, item_id):
        return self.embedding_item(item_id)

    def forward(self, users, items):
        output = self.mlp(self.emb_item(items))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def forward_c(self, users, items):
        output = self.mlp(self.emb_item_c(items))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def forward_pre(self, users, items):
        output = self.mlp(self.embedding_p(items))
        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output
    
    def train_step(self, users, items, labels):
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(users, items)
        loss =  self.loss_fn(pred, labels, ) + self.add_regularization()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_step_triple(self, user, pos, neg):
        self.train()
        self.optimizer.zero_grad()
        pred_pos = self.forward(user, pos,)
        pred_neg = self.forward(user, neg,)
        loss =  self.loss_fn(pred_pos, pred_neg,) + self.add_regularization_triple(self.emb_item(pos), self.emb_item(neg),)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_step_triple_c(self, user, pos, neg):
        self.train()
        self.optimizer.zero_grad()
        pred_pos = self.forward_c(user, pos,)
        pred_neg = self.forward_c(user, neg,)
        loss =  self.loss_fn(pred_pos, pred_neg,) + self.add_regularization_triple(self.emb_item_c(pos), self.emb_item_c(neg),)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_step_triple_pre(self, user, pos, neg):
        self.train()
        self.optimizer.zero_grad()
        pred_pos = self.forward_pre(user, pos,)
        pred_neg = self.forward_pre(user, neg,)
        loss =  self.loss_fn(pred_pos, pred_neg,) + self.add_regularization_triple(self.embedding_item.weight[pos], self.embedding_item.weight[neg],)
        loss.backward()
        self.optimizer.step()
        return loss
    
class Client(ClientBase):
    model:model 
    def __init__(self, client_id, model, task,):
        super().__init__(client_id, model)
        self.task = task.lower()

    def local_train(self, user, local_epoch, dataload, pre_train=False, compressed=False):
        self.model.train()
        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)
            for _ in range(local_epoch):
                if compressed:
                    loss = self.model.train_step_triple_c(users, pos, neg)
                elif pre_train:
                    loss = self.model.train_step_triple_pre(users, pos, neg)
                else:
                    loss = self.model.train_step_triple(users, pos, neg)
        else:
            users, items, labels = dataload.get_traindata(user)
            self.__local_data_num = labels.size(0)
            for _ in range(local_epoch):
                loss = self.model.train_step(users, items, labels)
        # logging.info("Client {} for user {}, train loss: {:.6f}".format(self.client_id, user, loss))
        return loss
    
    def local_data_num(self):
        return self.__local_data_num
    
    def storage_score_function(self):
        return self.model.mlp.state_dict()
    
    def load_score_function(self, mlp):
        self.model.mlp.load_state_dict(mlp)
        self.model.to(self.model.device)

class Server(ServerBase):
    model:model
    def __init__(self, model, ):
        super().__init__(model)
        self.models = {}
        self.global_model = copy.deepcopy(self.model.mlp.state_dict())

    def count_parameters(self):
        # flops, params = profile(self.model, inputs=(torch.tensor(0, dtype=torch.int64, device=self.model.device),
        #                                             torch.tensor(1, dtype=torch.int64, device=self.model.device)))
        # logging.info("FLOPs: {:.8f} MFLOPs".format(flops/ 1e6))
        # logging.info("Param: {:.8f} M".format(params/ 1e6))
        self.model.eval()
        base_model_dict = copy.deepcopy(self.model.state_dict())
        model_size = 0.
        for name in base_model_dict.keys():
            if "embedding" in name:
                continue
            else:
                _, param_size = calculate_model_size(base_model_dict[name])
                logging.info("Model {} size: {:.8f}MB".format(name, param_size))
                model_size += param_size
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Model all size: {:.8f}MB".format(model_size))
        
    def aggregation(self, user_list, model_list, mlp_list, num_list, loss_list):
        self.model.eval()
        for i, user in enumerate(user_list):
            self.models[user] = mlp_list[i]
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())
        for name in base_model_dict.keys():
            base_model_dict[name] = sum([model[name] * num for model, num in zip(model_list, num_list)]) / data_num
        self.model.load_weights(copy.deepcopy(base_model_dict))
        self.global_model = copy.deepcopy(self.model.mlp.state_dict())
        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))

    def distribute_mlp(self, user):
        if user in self.models:
            return self.models[user]
        else:
            return self.global_model

    def get_client_model(self, user):
        if user in self.models:
            self.model.mlp.load_state_dict(self.models[user])
        else:
            self.model.mlp.load_state_dict(self.global_model)
        self.model.to(self.model.device)

    def evaluate(self, dataload, user_list):
        self.model.eval()
        y_pred = []
        y_true = []
        group_id = []
        for user in user_list:
            self.get_client_model(user)
            users, items, labels = dataload.get_testdata(user)
            y_pred.extend(self.model.forward(users, items).data.cpu().numpy().reshape(-1))
            y_true.extend(labels.data.cpu().numpy().reshape(-1))
            group_id.extend(users.data.cpu().numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        val_logs = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

class PFedRec_Lora:
    def __init__(self,
                dataload:BaseDataLoaderFL,
                clients_num_per_turn, 
                local_epoch, 
                train_turn,
                user_num,
                item_num,
                embedding_dim,
                hidden_activations, 
                hidden_units, 
                latent_dim,
                embedding_regularizer, 
                net_regularizer, 
                learning_rate,
                optimizer, 
                loss_fn,
                device,
                metrics,
                task,
                *args, **kwargs
                ):
        self.clients_num_per_turn = clients_num_per_turn
        self.task = task.lower()
        self.local_epoch =  local_epoch
        self.train_turn = train_turn
        self.device = device
        self.user_num = user_num
        self.item_num = item_num
        self.dataload = dataload
        self.server_model = model(user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            hidden_activations=hidden_activations,
            hidden_units=hidden_units,
            latent_dim=latent_dim,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,)
            
        self.server_model.reset_parameters()

        self.server = Server(self.server_model)
        self.client = Client(client_id=0, model=model(user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            hidden_activations=hidden_activations,
            hidden_units=hidden_units,
            latent_dim=latent_dim,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,),  task=task.lower()) 

        self.g_model = AE(hidden_units = kwargs["g_hidden_units"],
                hidden_activations = kwargs["g_hidden_activations"],
                embedding_dim = kwargs["sen_embedding_dim"], 
                embedding_dim_latent = embedding_dim,
                device = device, 
                embedding_regularizer=0., 
                net_regularizer=1e-2, 
                learning_rate=1e-4,
                optimizer=optimizer,
                loss_fn = "mse_loss",)

        self.pre_epoch = kwargs["pre_epoch"]
        self.compressed = kwargs.get("compressed", False)
    
    def fit(self):
        self.server.count_parameters()
        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for turn in range(self.pre_epoch):
                loss = self.g_model.train_step(item_feature)
                logging.info("loss: {} for iter: {}".format(loss, turn))
            latent = self.g_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())
            self.server.global_model = self.server.model.embedding_p.state_dict()
            
        for turn in range(self.train_turn):
            logging.info("********* Train Turn {} *********".format(turn))
            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)
            client_model = []
            client_local_data_num = []
            client_mlp = []
            losses = []
            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model())
                self.client.load_score_function(self.server.distribute_mlp(user))
                loss = self.client.local_train(user, self.local_epoch, self.dataload, turn < 20, self.compressed)
                losses.append(loss)
                client_model.append(self.client.upload_model())
                client_mlp.append(self.client.storage_score_function())
                client_local_data_num.append(self.client.local_data_num())
            self.server.aggregation(select_users, client_model, client_mlp, client_local_data_num, losses)
            torch.cuda.empty_cache()
                
        logging.info("********* Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results