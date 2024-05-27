import numpy as np
import torch
import torch.nn as nn
from .data import PositionalEncoding
from tqdm.auto import tqdm
class Decoder(nn.Module):
    def __init__(self,embedding_dim,hidden_size):
        super(Decoder,self).__init__()
        self.change_final_shape=False
        self.value_cache=[]
        self.key_cache=[]
        self.W_value=nn.Parameter(torch.rand(embedding_dim,hidden_size))
        self.W_Key=nn.Parameter(torch.rand(embedding_dim,hidden_size))
        self.W_Query=nn.Parameter(torch.rand(embedding_dim,hidden_size)) 
        self.softmax=nn.Softmax(dim=1)
        if (hidden_size!=embedding_dim):
            self.change_final_shape=True
            self.linear=nn.Linear(hidden_size,embedding_dim)
    def forward(self,x):
        """
            x : input embeding + positional_encoding : (batch_size,embeding_dim)
        """
        # calculate the value
        value=torch.matmul(x,self.W_value) # (batch_size,hidden_size)
        # claculate the key
        key=torch.matmul(x,self.W_Key) # (batch_size,hidden_size)
        # calculate the query
        query=torch.matmul(x,self.W_Query) # (batch_size,hidden_size)
        #add the key and value in cache
        self.value_cache.append(value)
        self.key_cache.append(key)
        #calculate the result of key*query for every past word + current word
        softmax_in=torch.sum(key*query,dim=1)
        softmax_in=torch.unsqueeze(softmax_in,dim=-1)
        for k in reversed(self.key_cache[:-1]):
            softmax_in=torch.concat((torch.unsqueeze(torch.sum(k*query,dim=-1),dim=-1)
                                     ,softmax_in),dim=1) #each have the shape (batch_Size,1)
        # softmax_in :(batch_size,seq_length)
        # caclulate the softmax  
        
        softmax_out=self.softmax(softmax_in)
        attentions=torch.unsqueeze(softmax_out,dim=-1)  # (batch_size,seq_length,1)
        # calculate attention efects on values
        res=attentions[:,-1]*self.value_cache[-1]# (batch_size,hidden_size) or (batch_size,embedding_dim) if hidden_size = vocab size
        for i,v in enumerate(self.value_cache[:-1]):
            res+=v*attentions[:,i]
        res=res/len(attentions) # can remove it

        if(self.change_final_shape==True):
            res=self.linear(res) #(batch_size,embedding_dim)
        
        # add res to positional embeding tokens
        decoder_output = res+ x
        
        return decoder_output #(batch_size,embedding_dim)
    def clear_cache(self):
        del self.value_cache[:]
        del self.key_cache[:]

class TransformerModel(nn.Module):
    def __init__(self,embedding_dim,hidden_size,vocab_size,device="cpu"):
        super(TransformerModel,self).__init__()
        self.embedding=nn.Embedding(embedding_dim=embedding_dim,num_embeddings=vocab_size).to(device)
        self.decoder=Decoder(embedding_dim,hidden_size).to(device)
        self.linear=nn.Linear(embedding_dim,vocab_size).to(device)
        self.softmax=nn.Softmax(dim=-1).to(device)
        self.device=device
        self.embedding_dim=embedding_dim

    def Compile(self,metric,optimizer="adam",lr=0.001,lr_scheduler=None,loss_fn=nn.CrossEntropyLoss()):
        self.metric=metric
        self.has_lr_scheduler=False
        if(optimizer=="adam"):
            self.Optimizer=torch.optim.Adam(lr=lr,params=self.parameters())
        else:
            self.Optimizer=torch.optim.SGD(lr=lr,params=self.parameters())
        if(lr_scheduler!=None):
            self.has_lr_scheduler=True
            self.lr_scheduler=lr_scheduler
        self.loss_fn=loss_fn
    def predict(self,y_pred):
        return torch.argmax(y_pred,dim=-1).detach()
    def send_prompt(self,prompt):
        pass
    def step_optimizer(self):
        self.Optimizer.step()
        self.Optimizer.zero_grad()
    def step_scheduler(self):
        if(self.has_lr_scheduler==False):
            return
        self.lr_scheduler.step()
    def Train(self,trainLoader,validLoader,epochs=10):

        """
            trainLoader : returns x and y with shape (batch_size,embedding_dim)
        """
        with tqdm(total=epochs) as main_pbar:
            for ep in tqdm(range(epochs)):
                with tqdm(total=len(trainLoader)) as pbar:
                    for x_batch,y_batch in trainLoader:
                        x_batch=x_batch.to(self.device)
                        y_batch=y_batch.to(self.device)
                        seq_len=x_batch.shape[-1]
                        batch_size=x_batch.shape[0]
                        embedded_input=self.embedding(x_batch).reshape(seq_len,-1,self.embedding_dim)
                        position_encodings=PositionalEncoding(seq_len=seq_len,
                                                            dim=self.embedding_dim,
                                                            batch_size=batch_size)
                        # print(x_batch.shape)
                        # print(position_encodings.shape)
                        # print(embedded_input.shape)
                        encoded_input=embedded_input+position_encodings # (seq_len,batch_size,embedding_dim)
                        loss=0
                        acc=0
                        for i in range(seq_len):
                            token=encoded_input[i] # (batch_size,embedding_dim)
                            logits=self.decoder(token) # (batch_size,embedding_dim)
                            logits=self.linear(token) # (batch_size,vocab_size)
                            y_pred=self.softmax(logits) # (batch_size,vocab_size)
                            loss+=self.loss_fn(input=y_pred,target=y_batch[:,i].reshape(-1))
                            y_hat=self.predict(y_pred)
                            acc+=self.metric(y_true=y_batch[:,i].reshape(-1).detach().cpu().numpy(),
                                            y_pred=y_hat.cpu().numpy())
                        loss=loss/batch_size
                        acc=acc/seq_len
                        loss.backward()
                        self.step_optimizer()

                        pbar.set_postfix({"Epoch":ep,"TrainLoss":loss.detach().item(),"TrainAcc":acc,
                                    "lr":self.Optimizer.param_groups[0]['lr']})
                        pbar.update(1)
                valid_loss,acc_valid=self.Evaluation(validLoader)
                main_pbar.set_postfix({"Epoch":ep,"ValidLoss":valid_loss.detach().item(),"ValidAcc":acc_valid})
                main_pbar.update(1)
                self.step_scheduler()
        
    @torch.no_grad()
    def Evaluation(self,dataloader):
        for x_batch,y_batch in tqdm(dataloader):
            x_batch=x_batch.to(self.device)
            y_batch=y_batch.to(self.device)
            seq_len=x_batch.shape[-1]
            batch_size=x_batch.shape[0]
            embedded_imput=self.embedding(x_batch).reshape(self.embedding_dim,-1,seq_len)
            position_encodings=position_encodings(seq_len=seq_len,
                                                dim=self.embedding_dim,
                                                batch_size=batch_size)
            encoded_imput=embedded_imput+position_encodings # (seq_len,batch_size,embedding_dim)
            loss=0
            acc=0
            for i in range(seq_len):
                token=encoded_imput[i] # (batch_size,embedding_dim)
                logits=self.decoder(token) # (batch_size,embedding_dim)
                logits=self.linear(token) # (batch_size,vocab_size)
                y_pred=self.softmax(logits) # (batch_size,vocab_size)
                loss+=self.loss_fn(input=y_pred,target=y_batch[:,i].reshape(-1))
                y_hat=self.predict(y_pred)
                acc+=self.metric(y_true=y_batch[:,i].reshape[-1].detach().cpu(),
                                y_pred=y_hat.cpu())
        return loss/batch_size , acc/batch_size