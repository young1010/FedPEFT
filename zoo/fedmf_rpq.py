"""

"""

import copy
import numpy as np
import torch.nn as nn
import torch
import logging
from dataloaders.BaseDataLoader import *
from framework.fed.client import ClientBase
from framework.fed.server import ServerBase
from framework.modules.models import BaseModel, AE, PQ_VAE, RPQ_VAE
from framework.modules.layers import MLP_Block, PQ, RPQ
from framework.utils import calculate_model_size
from thop import profile

class model(BaseModel):
    def __init__(self, 
                 user_num, 
                 item_num, 
                 embedding_dim, 
                 mc_size,
                 cb_size,
                 task,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer,
                 loss_fn,
                 metrics,
                 *args, **kwargs):
        super(__class__, self).__init__(device=device,
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  metrics=metrics)
        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_item = RPQ(latent_size=embedding_dim, device = device, mc_size=mc_size, cb_size=cb_size,)
        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.task = task
        self.output_activation= nn.Sigmoid()
        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self, ):
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_p.weight)
        for emb in self.embedding_item.code_embedding_tables:
            # nn.init.zeros_(emb.weight)
            nn.init.xavier_uniform_(emb.weight)

    def emb_item(self, item_id):
        return self.embedding_p(item_id) + torch.sum(self.embedding_item(item_id), dim=-1)

    def emb_item_c(self, item_id):
        return torch.sum(self.embedding_item(item_id), dim=-1)

    def forward(self, user_id, item_id):
        output = (self.embedding_user(user_id)*self.emb_item(item_id)).sum(1)

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def forward_c(self, user_id, item_id):
        output = (self.embedding_user(user_id)*self.emb_item_c(item_id)).sum(1)

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output
    
    def forward_pre(self, user_id, item_id):
        output = (self.embedding_user(user_id)*self.embedding_p(item_id)).sum(-1)

        if self.task != "triple":
            output = self.output_activation(output)
            if self.task == "regression":
                output = output * 4.0 + 1.0
            return output
        return output

    def train_step(self, users, items, label):
        self.train()
        self.embedding_p.requires_grad_ = False
        self.optimizer.zero_grad()
        pred = self.forward(users, items).squeeze()
        loss = self.loss_fn(pred, label, reduction='mean') + self.add_regularization()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_step_triple(self, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = False
        self.optimizer.zero_grad()
        pred_pos = self.forward(users, pos)
        pred_neg = self.forward(users, neg)
        if len(users) <=0:
            loss = self.loss_fn(pred_pos, pred_neg, )
        else:
            loss = self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(self.embedding_user.weight[users[0]], 
                                                                                self.emb_item(pos), self.emb_item(neg),
                                                                                   )
        loss.backward()
        self.optimizer.step()
        return loss

    def train_step_triple_c(self, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = False
        self.optimizer.zero_grad()
        pred_pos = self.forward_c(users, pos)
        pred_neg = self.forward_c(users, neg)
        if len(users) <=0:
            loss = self.loss_fn(pred_pos, pred_neg, )
        else:
            loss = self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(self.embedding_user.weight[users[0]], 
                                                                                self.emb_item_c(pos), self.emb_item_c(neg),
                                                                                   )
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_step_triple_pre(self, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = True
        self.optimizer.zero_grad()
        pred_pos = self.forward_pre(users, pos)
        pred_neg = self.forward_pre(users, neg)
        if len(users) <=0:
            loss = self.loss_fn(pred_pos, pred_neg, )
        else:
            loss = self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(self.embedding_user.weight[users[0]], 
                                                                                self.embedding_p(pos), self.embedding_p(neg),
                                                                                   )
        loss.backward()
        self.optimizer.step()
        return loss

class Client(ClientBase):
    model:model
    def __init__(self, client_id, model, task,):
        super().__init__(client_id, model)
        self.task = task.lower()

    def load_model(self, model):
        self.model.load_weights(model[0])
        self.model.embedding_item.code_list = model[1]
        self.model.to(self.model.device)

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

class Server(ServerBase):
    model:model
    def __init__(self, model, ):
        super().__init__(model)
        self.models = {}
        self.global_model = copy.deepcopy(self.model.embedding_p.state_dict())
    
    def count_parameters(self):
        # flops, params = profile(self.model, inputs=(torch.tensor(0, dtype=torch.int64, device=self.model.device),
        #                                             torch.tensor(1, dtype=torch.int64, device=self.model.device)))
        # logging.info("FLOPs: {:.8f} MFLOPs".format(flops/ 1e6))
        # logging.info("Param: {:.8f} M".format(params/ 1e6))
        self.model.eval()
        base_model_dict = copy.deepcopy(self.model.state_dict())
        model_size = 0.
        for name in base_model_dict.keys():
            if "embedding_user" in name:
                continue
            else:
                _, param_size = calculate_model_size(base_model_dict[name])
                logging.info("Model {} size: {:.8f}MB".format(name, param_size))
                model_size += param_size
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Model all size: {:.8f}MB".format(model_size))
    
    def distribute_model(self, user):
        return (super().distribute_model(), self.model.embedding_item.get_code_list())
    
    def aggregation(self, user_list, model_list, num_list, loss_list, cdp=None, ldp=None):
        self.model.eval()
        # for i, user in enumerate(user_list):
        #     self.models[user] = model_list[i]["embedding_p.weight"]
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())
        for name in base_model_dict.keys():
            if "embedding_user" in name:
                for model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = model[name].data[user]
            else:
                base_model_dict[name] = sum([model[name] * num for model, num in zip(model_list, num_list)]) / data_num
                if cdp is not None and cdp > 0.:
                    base_model_dict[name] += torch.normal(0, cdp, size=base_model_dict[name].size()).to(self.model.device)
                elif ldp is not None and ldp > 0.:
                    noise_list = [torch.normal(0, ldp, size=base_model_dict[name].size()).to(self.model.device) for _ in range(len(user_list))]
                    base_model_dict[name] += torch.mean(torch.stack(noise_list), dim=0)
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))

    def get_client_model(self, user):
        if user in self.models:
            self.model.embedding_p.load_state_dict({"weight":self.models[user]})
        else:
            self.model.embedding_p.load_state_dict(self.global_model)
        self.model.to(self.model.device)

    def evaluate(self, dataload, user_list):
        self.model.eval()
        y_pred = []
        y_true = []
        group_id = []
        for user in user_list:
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
    
class FedMF_RPQ:
    def __init__(self, 
                 dataload:BaseDataLoaderFL,
                 clients_num_per_turn, 
                 local_epoch, 
                 train_turn,
                 user_num,
                 item_num,
                 embedding_dim,
                 mc_size,
                 cb_size,
                 g_hidden_units,
                 g_hidden_activations,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer, 
                 loss_fn,
                 metrics,
                 task,
                 *args, **kwargs
                 ):
        server_model =  model(
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            mc_size=mc_size,
            cb_size=cb_size,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
        )
        server_model.reset_parameters()
        self.server = Server(server_model)
        self.client = Client(client_id=0, model=model(
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            mc_size=mc_size,
            cb_size=cb_size,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
        ), task=task.lower()) 

        self.g_model = RPQ_VAE(mc_size=mc_size, cb_size=cb_size,
                hidden_units = g_hidden_units,
                hidden_activations = g_hidden_activations,
                embedding_dim = kwargs["sen_embedding_dim"], 
                embedding_dim_latent = embedding_dim,
                device = device, 
                embedding_regularizer=0., 
                net_regularizer=1e-2, 
                learning_rate=1e-4,
                optimizer=optimizer,
                loss_fn = "mse_loss",)
        self.p_model = AE(hidden_units = g_hidden_units,
                hidden_activations = g_hidden_activations,
                embedding_dim = kwargs["sen_embedding_dim"], 
                embedding_dim_latent = embedding_dim,
                device = device, 
                embedding_regularizer=0., 
                net_regularizer=1e-2, 
                learning_rate=1e-4,
                optimizer=optimizer,
                loss_fn = "mse_loss",)

        self.clients_num_per_turn = clients_num_per_turn
        self.local_epoch =  local_epoch
        self.train_turn = train_turn
        self.user_num = user_num
        self.task = task.lower()
        self.device = device
        self.dataload = dataload
        self.model_dir = kwargs["model_dir"]
        self.pre_epoch = kwargs["pre_epoch"]
        self.compressed = kwargs.get("compressed", False)

    def fit(self,):
        self.server.count_parameters()
        item_feature = self.dataload.get_item_feature()
        self.g_model.pre_train(item_feature, self.pre_epoch, self.model_dir)
        for turn in range(self.pre_epoch):
            loss, latent = self.g_model.train_step(item_feature)
            logging.info("loss: {} for iter: {}".format(loss, turn))
        self.server.model.embedding_item.set_code_list(self.g_model.get_code_table().get_code_list(), self.server.model.device)

        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for turn in range(self.pre_epoch):
                loss = self.p_model.train_step(item_feature)
                logging.info("loss: {} for iter: {}".format(loss, turn))
            latent = self.p_model.get_latent(item_feature)
            self.server.model.embedding_p.weight.data = copy.deepcopy(latent.detach())

        for turn in range(self.train_turn):
            logging.info("********* Train Turn {} *********".format(turn))
            select_users = self.server.select_clients(self.user_num, self.clients_num_per_turn)
            client_model = []
            client_local_data_num = []
            losses = []
            for user in select_users:
                self.client.load_client(user)
                self.client.load_model(self.server.distribute_model(user))
                loss = self.client.local_train(user, self.local_epoch, self.dataload, turn < 20, self.compressed)
                losses.append(loss)
                client_model.append(self.client.upload_model())
                client_local_data_num.append(self.client.local_data_num())
            self.server.aggregation(select_users, client_model, client_local_data_num, losses, )

        logging.info("********* Test *********")
        results = self.server.evaluate(self.dataload, range(self.user_num))
        return results

