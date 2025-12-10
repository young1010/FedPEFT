import os
import random
import torch.nn as nn
import torch
import logging
import pickle
from framework.metrics import evaluate_metrics
from framework.modules.utils import get_device, get_optimizer, get_loss, get_regularizer
from framework.modules.layers import MLP_Block, PQ, RPQ

class BaseModel(nn.Module):
    def __init__(self, 
                 device=-1, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 metrics=None,
                 *args, **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(device)
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self.metrcis=metrics

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss = loss
        self.loss_fn = get_loss(loss)

    def add_regularization(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        if self._embedding_regularizer:
                            for emb_p, emb_lambda in emb_reg:
                                reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                    else:
                        if self._net_regularizer:
                            for net_p, net_lambda in net_reg:
                                reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term
    
    def add_regularization_triple(self, *embs):
        reg_loss = 0
        if self._embedding_regularizer:
            for emb in embs:
                reg_loss += torch.norm(emb, p=2)
            reg_loss /= embs[-1].shape[0]
        return reg_loss * self._embedding_regularizer

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)
        
    def model_to_device(self):
        self.to(device=self.device)
    
    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint:str):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint:str):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def load_weights(self, state_dict:dict):
        self.to(self.device)
        self.load_state_dict(state_dict, strict=False)
    
    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

    def grad_false(self):
        for name, param in self.named_parameters(): 
            if "attack" not in name:
                param.requires_grad = False

class AE(BaseModel):
    def __init__(self, 
                 hidden_units,
                 hidden_activations,
                 embedding_dim, 
                 embedding_dim_latent,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer,
                 loss_fn,
                 *args, **kwargs):
        super(__class__, self).__init__(device=device,
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer)
        self.encoder = MLP_Block(input_dim = embedding_dim,
                             output_dim= embedding_dim_latent,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             )
        self.decoder = MLP_Block(input_dim = embedding_dim_latent,
                        output_dim= embedding_dim,
                        hidden_units=hidden_units[-1::-1],
                        hidden_activations=hidden_activations,
                        )
        self.reset_parameters()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def forward(self, input):
        latent = self.encoder(input)
        output = self.decoder(latent)
        return output
    
    def get_latent(self, input):
        self.eval()
        latent = self.encoder(input)
        return latent.detach()
    
    def train_step(self, input):
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(input)
        loss = self.loss_fn(pred, input, reduction='mean') + self.add_regularization()
        loss.backward()
        self.optimizer.step()
        return loss
    
class PQ_VAE(BaseModel):
    def __init__(self, 
                 mc_size,
                 hidden_units,
                 hidden_activations,
                 embedding_dim, 
                 embedding_dim_latent,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer,
                 loss_fn,
                 *args, **kwargs):
        super(__class__, self).__init__(device=device,
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer)
        self.encoder = PQ(embedding_dim=embedding_dim, embedding_dim_latent=embedding_dim_latent, mc_size=mc_size, device=device)
        self.decoder = MLP_Block(input_dim = embedding_dim_latent,
                        output_dim= embedding_dim,
                        hidden_units=hidden_units[-1::-1],
                        hidden_activations=hidden_activations,
                        )
        self.reset_parameters()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def forward(self, input):
        latent, code_list = self.PQ.forward_pre(self.encoder(input))
        self.code_list = code_list
        output = self.decoder(latent)
        return output
    
    def get_code_list(self):
        self.eval()
        return self.PQ, self.code_list
    
    def train(self, input):
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(input)
        loss = self.loss_fn(pred, input, reduction='sum') + self.add_regularization()
        loss.backward()
        self.optimizer.step()
        return loss

class RPQ_VAE(BaseModel):
    def __init__(self, 
                 mc_size,
                 cb_size,
                 hidden_units,
                 hidden_activations,
                 embedding_dim, 
                 embedding_dim_latent,
                 device, 
                 embedding_regularizer, 
                 net_regularizer, 
                 learning_rate,
                 optimizer,
                 loss_fn,
                 *args, **kwargs):
        super(__class__, self).__init__(device=device,
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer)
        self.encoder = MLP_Block(input_dim = embedding_dim,
                             output_dim= embedding_dim_latent,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             batch_norm=True,
                             )
        self.RPQ = RPQ(latent_size=embedding_dim_latent, device=device, mc_size=mc_size, cb_size=cb_size, adapt=True, norm=True)
        self.decoder = MLP_Block(input_dim = embedding_dim_latent,
                        output_dim= embedding_dim,
                        hidden_units=hidden_units[-1::-1],
                        hidden_activations=hidden_activations,
                        batch_norm=True,
                        )
        self.mc_size=mc_size
        self.reset_parameters()
        self.compile(optimizer=optimizer, loss=loss_fn, lr=learning_rate)
        self.model_to_device()

    def forward(self, input):
        latent = self.RPQ(self.encoder(input))
        output = self.decoder(latent)
        return latent, output
    
    def get_code_table(self):
        return self.RPQ
    
    def train_step(self, input):
        self.train()
        self.optimizer.zero_grad()
        latent = self.encoder(input)
        latent_generated, embeddings, residual = self.RPQ.train_step(latent)
        output = self.decoder(latent_generated)
        loss = self.loss_fn(output, input, reduction='mean') 
        for emb, res in zip(embeddings, residual):
            loss += self.loss_fn(emb, res.detach(), reduction='mean')
            loss += self.loss_fn(res, emb.detach(), reduction='mean')* 0.25
        loss.backward()
        self.optimizer.step()
        return loss, latent
    
    def pre_train(self, input, epoch, model_dir):
        latent = self.encoder(input)
        self.RPQ.pre_train(latent)

    def save_model(self, epoch, model_dir):
        with open(model_dir + str(self.mc_size) + "_" + str(epoch) + ".pkl", 'wb') as f:
            pickle.dump(self.RPQ.code_list, f)

    def load_model(self, epoch, model_dir):
        with open(model_dir + str(self.mc_size) + "_" + str(epoch) + ".pkl", 'rb') as f:
            loaded_list = pickle.load(f,)
        for i, data in enumerate(loaded_list):
            loaded_list[i] = data.to(self.device)
        self.RPQ.code_list = loaded_list
        # self.RPQ.count_collisions()

        # model_path = model_dir + str(self.mc_size) + "_" + str(epoch) + ".pth"
        # if os.path.exists(model_path):
        #     print("Loading pre Training Model...")
        #     state_dict = torch.load(model_path, map_location=self.device)
        #     self.load_weights(state_dict)
        #     with open(model_dir + str(self.mc_size) + "_" + str(epoch) + ".pkl", 'rb') as f:
        #         loaded_list = pickle.load(f)
        #     self.RPQ.code_list = loaded_list
        # else:
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        #     print("Start Pre Training...")
        #     self.train()
        #     for turn in range(epoch):
        #         self.optimizer.zero_grad()
        #         latent = self.encoder(input)
        #         output = self.decoder(latent)
        #         loss = self.loss_fn(output, input, reduction='mean')
        #         loss.backward()
        #         self.optimizer.step()
        #         logging.info("loss: {} for iter: {}".format(loss, turn))

        #     latent = self.encoder(input)
        #     self.RPQ.pre_train(latent)
            
        #     torch.save(self.state_dict(), model_path)  
        #     with open(model_dir + str(self.mc_size) + "_" + str(epoch) + ".pkl", 'wb') as f:
        #         pickle.dump(self.RPQ.code_list, f)

