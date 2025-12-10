"""
Wu C, Wu F, Lyu L, et al. A federated graph neural network framework for privacy-preserving personalization
[J]. Nature Communications, 2022, 13(1): 3091.

TY  - JOUR
AU  - Wu, Chuhan
AU  - Wu, Fangzhao
AU  - Lyu, Lingjuan
AU  - Qi, Tao
AU  - Huang, Yongfeng
AU  - Xie, Xing
PY  - 2022
DA  - 2022/06/02
TI  - A federated graph neural network framework for privacy-preserving personalization
JO  - Nature Communications
SP  - 3091
VL  - 13
IS  - 1
AB  - Graph neural network (GNN) is effective in modeling high-order interactions and has been widely used in various personalized applications such as recommendation. However, mainstream personalization methods rely on centralized GNN learning on global graphs, which have considerable privacy risks due to the privacy-sensitive nature of user data. Here, we present a federated GNN framework named FedPerGNN for both effective and privacy-preserving personalization. Through a privacy-preserving model update method, we can collaboratively train GNN models based on decentralized graphs inferred from local data. To further exploit graph information beyond local interactions, we introduce a privacy-preserving graph expansion protocol to incorporate high-order information under privacy protection. Experimental results on six datasets for personalization in different scenarios show that FedPerGNN achieves 4.0% ~ 9.6% lower errors than the state-of-the-art federated personalization methods under good privacy protection. FedPerGNN provides a promising direction to mining decentralized graph data in a privacy-preserving manner for responsible and intelligent personalization.
SN  - 2041-1723
UR  - https://doi.org/10.1038/s41467-022-30714-9
DO  - 10.1038/s41467-022-30714-9
ID  - Wu2022
ER  - 

@article{wu2022federated,
  title={A federated graph neural network framework for privacy-preserving personalization},
  author={Wu, Chuhan and Wu, Fangzhao and Lyu, Lingjuan and Qi, Tao and Huang, Yongfeng and Xie, Xing},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={3091},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

https://github.com/wuch15/FedPerGNN
"""
import copy
import numpy as np
import torch.nn as nn
import torch
import logging
from framework.fed.client import ClientBase
from framework.fed.server import ServerBase
from framework.modules.models import BaseModel, AE, PQ_VAE, RPQ_VAE
from framework.modules.layers import MLP_Block, PQ, RPQ, SENet_Block, HashEmb
from dataloaders.BaseDataLoader import *
from framework.modules.layers import MLP_Block
import scipy.sparse as sp
from framework.utils import calculate_model_size
from thop import profile
    
class model(BaseModel):
    def __init__(self, 
                user_num, 
                item_num, 
                embedding_dim, 
                layer_num,
                mc_size,
                cb_size,
                hash_senet,
                learning_rate, 
                optimizer,
                loss_fn,
                task,
                light=True,
                device=-1, embedding_regularizer=None, net_regularizer=None, metrics=None, *args, **kwargs):
        super(__class__, self).__init__(device, embedding_regularizer, net_regularizer, metrics)

        self.embedding_user = nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_dim)
        self.embedding_item = HashEmb(embedding_dim_latent=embedding_dim, device = device, mc_size=mc_size, cb_size=cb_size,)
        self.hash_senet = hash_senet
        if hash_senet:
            self.senet = SENet_Block(input_dim=cb_size, reduction_ratio=16, activation="ReLU")
        self.embedding_p = nn.Embedding(num_embeddings=item_num, embedding_dim=embedding_dim)
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num
        self.task = task
        if task == "regression":
            self.predictor = MLP_Block(input_dim = embedding_dim*2,output_dim=1, output_activation="sigmoid")
            
        self.reset_parameters()
        self.__init_weight()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def __init_weight(self, ):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_p.weight, std=0.1)
        nn.init.normal_(self.embedding_item.code_embedding_tables.weight, std=0.1)

    def emb_item(self, item_id):
        if self.hash_senet:
            return self.embedding_p(item_id) + torch.sum(self.senet(self.embedding_item(item_id)), dim=-1)
        return self.embedding_p(item_id) + torch.mean(self.embedding_item(item_id), dim=-1)

    def emb_item_c(self, item_id):
        if self.hash_senet:
            return torch.sum(self.senet(self.embedding_item(item_id)), dim=-1)
        return torch.mean(self.embedding_item(item_id), dim=-1)

    def propagate(self, graph):
        users_emb = self.embedding_user.weight
        items_emb = self.emb_item(torch.tensor(range(self.item_num), device=self.device))
        all_emb = torch.cat([users_emb, items_emb])
        for _ in range(self.layer_num):
            all_emb = torch.sparse.mm(graph, all_emb)
        users, items = torch.split(all_emb, [self.user_num, self.item_num])
        return users, items

    def propagate_c(self, graph):
        users_emb = self.embedding_user.weight
        items_emb = self.emb_item_c(torch.tensor(range(self.item_num), device=self.device))
        all_emb = torch.cat([users_emb, items_emb])
        for _ in range(self.layer_num):
            all_emb = torch.sparse.mm(graph, all_emb)
        users, items = torch.split(all_emb, [self.user_num, self.item_num])
        return users, items
        
    def propagate_pre(self, graph):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_p.weight
        all_emb = torch.cat([users_emb, items_emb])
        for _ in range(self.layer_num):
            all_emb = torch.sparse.mm(graph, all_emb)
        users, items = torch.split(all_emb, [self.user_num, self.item_num])
        return users, items
    
    def forward(self, all_users, users, items):
        user_agg = all_users[users]
        items_emb = self.emb_item(items)
        return self.predict(user_agg, items_emb)

    def forward_c(self, all_users, users, items):
        user_agg = all_users[users]
        items_emb = self.emb_item_c(items)
        return self.predict(user_agg, items_emb)
    
    def forward_pre(self, all_users, users, items):
        user_agg = all_users[users]
        items_emb = self.embedding_p(items)
        return self.predict(user_agg, items_emb)
    
    def train_step(self, graph, users, items, labels):
        self.train()
        self.optimizer.zero_grad()
        all_users, _ = self.propagate(graph)
        pred = self.forward(all_users, users, items)
        loss =  self.loss_fn(pred, labels, ) + self.add_regularization()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_step_triple(self, graph, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = False
        self.optimizer.zero_grad()
        all_users, all_items = self.propagate(graph)
        pred_pos = self.forward(all_users, users, pos)
        pred_neg = self.forward(all_users, users, neg)
        loss =  self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(all_users[users], all_items[pos], all_items[neg])
        loss.backward()
        self.optimizer.step()
        return loss

    def train_step_triple_c(self, graph, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = False
        self.optimizer.zero_grad()
        all_users, all_items = self.propagate_c(graph)
        pred_pos = self.forward_c(all_users, users, pos)
        pred_neg = self.forward_c(all_users, users, neg)
        loss =  self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(all_users[users], all_items[pos], all_items[neg])
        loss.backward()
        self.optimizer.step()
        return loss

    def train_step_triple_pre(self, graph, users, pos, neg):
        self.train()
        self.embedding_p.requires_grad_ = True
        self.optimizer.zero_grad()
        all_users, all_items = self.propagate_pre(graph)
        pred_pos = self.forward_pre(all_users, users, pos)
        pred_neg = self.forward_pre(all_users, users, neg)
        loss =  self.loss_fn(pred_pos, pred_neg, ) + self.add_regularization_triple(all_users[users], all_items[pos], all_items[neg])
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, users_emb, items_emb):
        if self.task == "regression":
            pred = self.predictor(torch.cat([users_emb, items_emb], dim=-1)).squeeze(1)
            gamma = pred * 4.0 + 1.0
        else:
            inner_pro = torch.mul(users_emb, items_emb)
            gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
        
class Client(ClientBase):
    model:model 
    def __init__(self, client_id, model, task,):
        super().__init__(client_id, model)
        self.task = task.lower()

    def load_model(self, model):
        self.model.load_weights(model[0])
        self.model.embedding_item.code_list = model[1]
        self.model.to(self.model.device)

    def local_train(self, graph, user, local_epoch, dataload, pre_train=False, compressed=False):
        self.model.train()
        if self.task == "triple":
            users, pos, neg = dataload.get_traindata(user)
            self.__local_data_num = users.size(0)
            for _ in range(local_epoch):
                if compressed:
                    loss = self.model.train_step_triple_c(graph, users, pos, neg)
                elif pre_train:
                    loss = self.model.train_step_triple_pre(graph, users, pos, neg)
                else:
                    loss = self.model.train_step_triple(graph, users, pos, neg)
        else:
            users, items, labels = dataload.get_traindata(user)
            self.__local_data_num = labels.size(0)
            for _ in range(local_epoch):
                loss = self.model.train_step(graph, users, items, labels)
        # logging.info("Client {} for user {}, train loss: {:.6f}".format(self.client_id, user, loss))
        return loss
    
    def local_data_num(self):
        return self.__local_data_num
    
class Server(ServerBase):
    model:model

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
    
    def distribute_model(self):
        return (super().distribute_model(), self.model.embedding_item.get_code_list())

    def aggregation(self, user_list, model_list, num_list, loss_list):
        self.model.eval()
        data_num = sum(num_list)
        base_model_dict = copy.deepcopy(self.model.state_dict())
        for name in base_model_dict.keys():
            if "embedding_user" in name:
                for model, user in zip(model_list, user_list):
                    base_model_dict[name].data[user] = model[name].data[user]
            else:
                base_model_dict[name] = sum([model[name] * num for model, num in zip(model_list, num_list)]) / data_num
        self.model.load_weights(copy.deepcopy(base_model_dict))
        logging.info("Clients average loss: {}".format(torch.mean(torch.tensor(loss_list))))
        
    def evaluate(self, graph, dataload, user_list):
        self.model.eval()
        all_users, _ = self.model.propagate(graph)
        y_pred = []
        y_true = []
        group_id = []
        for user in user_list:
            users, items, labels = dataload.get_testdata(user)
            y_pred.extend(self.model.forward(all_users, users, items).data.cpu().numpy().reshape(-1))
            y_true.extend(labels.data.cpu().numpy().reshape(-1))
            group_id.extend(users.data.cpu().numpy().reshape(-1))
        y_pred = np.array(y_pred, np.float64)
        y_true = np.array(y_true, np.float64)
        group_id = np.array(group_id) if len(group_id) > 0 else None
        val_logs = self.model.evaluate_metrics(y_true, y_pred, self.model.metrcis, group_id)
        logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
        return val_logs

class FedPerGNN_Hash:
    def __init__(self,
                dataload:BaseDataLoaderFL,
                clients_num_per_turn, 
                local_epoch, 
                train_turn,
                user_num,
                item_num,
                embedding_dim,
                layer_num,
                light,
                mc_size,
                cb_size,
                hash_senet,
                g_hidden_units,
                g_hidden_activations,
                embedding_regularizer, 
                net_regularizer, 
                learning_rate,
                optimizer, 
                loss_fn,
                device,
                metrics,
                path,
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
            layer_num=layer_num,
            light=light,
            mc_size=mc_size,
            cb_size=cb_size,
            hash_senet=hash_senet,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,)
            
        self.graph = self.getSpareseGraph(path, light)
        self.server_model.reset_parameters()

        self.server = Server(self.server_model)
        self.client = Client(client_id=0, model=model(user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            layer_num=layer_num,
            light=light,
            mc_size=mc_size,
            cb_size=cb_size,
            hash_senet=hash_senet,
            task=task.lower(),
            device=device,
            embedding_regularizer=embedding_regularizer, 
            net_regularizer=net_regularizer, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,),  task=task.lower()) 
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
        
        self.model_dir = kwargs["model_dir"]
        self.pre_epoch = kwargs["pre_epoch"]
        self.compressed = kwargs.get("compressed", False)
        self.mc_size = mc_size

    def getSpareseGraph(self, path, light):
        """
        https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/dataloader.py#L332
        """
        
        try:
            if light:
                pre_adj_mat = sp.load_npz(path + 'light_graph.npz')
            else:
                pre_adj_mat = sp.load_npz(path + 'gcn_graph.npz')
            norm_adj = pre_adj_mat
        except :
            adj_mat = sp.dok_matrix((self.user_num +  self.item_num, self.user_num + self.item_num), dtype=np.int64)
            adj_mat = adj_mat.tolil()
            R = self.dataload.get_aj_graph()
            adj_mat[:self.user_num, self.user_num:] = R
            adj_mat[self.user_num:, :self.user_num] = R.T
            adj_mat = adj_mat.todok()
            if not light:
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            if light:
                sp.save_npz(path + 'light_graph.npz', norm_adj)
            else:
                sp.save_npz(path + 'gcn_graph.npz', norm_adj)

        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)
    
    def fit(self):
        self.server.count_parameters()
        item_feature = self.dataload.get_item_feature()
        self.server.model.embedding_item.pre_train(item_feature)
        
        if not self.compressed:
            item_feature = self.dataload.get_item_feature()
            for turn in range(self.pre_epoch):
                loss = self.p_model.train_step(item_feature)
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
                self.client.load_model(self.server.distribute_model())
                loss = self.client.local_train(self.graph, user, self.local_epoch, self.dataload, turn < 20, self.compressed)
                losses.append(loss)
                client_model.append(self.client.upload_model())
                client_local_data_num.append(self.client.local_data_num())
            self.server.aggregation(select_users, client_model, client_local_data_num, losses,)
            torch.cuda.empty_cache()
                
        logging.info("********* Test *********")
        results = self.server.evaluate(self.graph, self.dataload, range(self.user_num))
        return results