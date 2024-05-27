import pandas as pd
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from itertools import zip_longest
import gc
import pickle
import json
import numpy as np
class processs:
    def __init__(self,tokenizer=word_tokenize,dir_path="./data/full/",max_size=None):
        self.train_path=os.path.join(dir_path,"train.csv")
        self.valid_path=os.path.join(dir_path,"valid.csv")
        self.test_path=os.path.join(dir_path,"test.csv")
        self.train=pd.read_csv(self.train_path)["texts"].tolist()
        self.test=pd.read_csv(self.test_path)["texts"].tolist()
        self.valid=pd.read_csv(self.valid_path)["texts"].tolist()
        self.tokenizer=tokenizer
        self.PAD="__PAD__"
        self.EOS="__EOS__"
        self.SOS="__SOS__"
        self.UNK="__UNK__"
        self.max_size=max_size
    def index_token(self):
        self.Index={}
        for i,t in enumerate(self.vocab):
            self.Index[t]=i
    def fit_transform(self):
        if(self.max_size==None):
            self.max_size=float('+inf')
        m=float("-inf")
        self.vocab=[self.PAD,self.SOS,self.EOS,self.UNK]
        temp=[]
        for i in tqdm(range(len(self.train))):
            tokenized=self.tokenizer(self.train[i])
            if(len(tokenized)+2>self.max_size):
                continue
            self.train[i]=[self.SOS]+tokenized+[self.EOS]
            temp+=tokenized
            m=max(m,len(tokenized)+2)
        self.max_size=m
        print("max size : ",self.max_size)
        self.vocab+=list(set(temp))
        self.index_token()

    def transform(self):
        for i in tqdm(range(len(self.valid))):
            tokenized=self.tokenizer(self.valid[i])
            self.valid[i]=[self.SOS]+tokenized+[self.EOS]
        
        for i in tqdm(range(len(self.test))):
            tokenized=self.tokenizer(self.test[i])
            self.test[i]=[self.SOS]+tokenized+[self.EOS]

    def token_To_index(self):
        for i in tqdm(range(len(self.train))):
            for j in range(len(self.train[i])):
                self.train[i][j]=self.Index.get(self.train[i][j],self.Index[self.UNK])

        for i in tqdm(range(len(self.valid))):
            for j in range(len(self.valid[i])):
                self.valid[i][j]=self.Index.get(self.valid[i][j],self.Index[self.UNK])

        for i in tqdm(range(len(self.test))):
            for j in range(len(self.test[i])):
                self.test[i][j]=self.Index.get(self.test[i][j],self.Index[self.UNK])
    def standard_pipeline(self,out_dir="./data/tokenized/"):
        print("createing vocab and transfroming train")
        self.fit_transform()
        print("transforming valid and test")
        self.transform()
        print("turning token to index")
        self.token_To_index()
        print(f"saving the data in {out_dir}")
        self.save_data(out_dir)
        print("clear memory")
        self.clear_memory()
    def save_data(self,out_dir="./data/tokenized/"):
        with open(os.path.join(out_dir,"train"), "wb") as fp:
            pickle.dump(self.train, fp)
        with open(os.path.join(out_dir,"test"), "wb") as fp:
            pickle.dump(self.test, fp)
        with open(os.path.join(out_dir,"valid"), "wb") as fp:
            pickle.dump(self.valid, fp)
        with open(os.path.join(out_dir,"vocab"), "wb") as fp:
            pickle.dump(self.vocab, fp)
        with open(os.path.join(out_dir,"index.json"), "w") as fp: 
            json.dump(self.Index, fp)
        with open(os.path.join(out_dir,"PAD"), "wb") as fp:
            pickle.dump(self.PAD, fp)
        with open(os.path.join(out_dir,"EOS"), "wb") as fp:
            pickle.dump(self.EOS, fp)
        with open(os.path.join(out_dir,"SOS"), "wb") as fp:
            pickle.dump(self.SOS, fp)
        with open(os.path.join(out_dir,"UKN"), "wb") as fp:
            pickle.dump(self.UNK, fp)
        with open(os.path.join(out_dir,"max_size"), "wb") as fp:
            pickle.dump(self.max_size, fp)
    def clear_memory(self):
        del self.train
        del self.valid
        del self.test
        del self.Index
        gc.collect()
class dataset(Dataset):
    def __init__(self,data,is_test=False):
        super(dataset,self).__init__()
        self.data=data
        self.test=is_test
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if(self.test==False):
            return torch.tensor(self.data[idx])[:-1],torch.tensor(self.data[idx])[1:]
        return torch.tensor(self.data[idx])[:-1]
def collect_fn(batch,pad=0):
    x,y=zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=pad)
    y = pad_sequence(y, batch_first=True, padding_value=pad)
    return x,y

def read_tokenized_data(path="./data/tokenized/"):
    train = pd.read_pickle(os.path.join(path,"train"))
    valid = pd.read_pickle(os.path.join(path,"valid"))
    test = pd.read_pickle(os.path.join(path,"test"))
    vocab = pd.read_pickle(os.path.join(path,"vocab"))
    max_size = pd.read_pickle(os.path.join(path,"max_size"))
    PAD = pd.read_pickle(os.path.join(path,"PAD"))
    EOS = pd.read_pickle(os.path.join(path,"EOS"))
    SOS = pd.read_pickle(os.path.join(path,"SOS"))
    UKN = pd.read_pickle(os.path.join(path,"UKN"))
    with open(os.path.join(path,"index.json"),"r") as f:
        Index = json.load(f)
    return train,valid,test,vocab,Index,PAD,EOS,SOS,UKN,max_size

def PositionalEncoding(seq_len,dim,batch_size,n=10000):
    sin_dim=(dim-(dim//2))
    cos_dim=(dim//2)
    sin_ind=np.arange(sin_dim)
    cos_ind=np.arange(cos_dim)
    posE=torch.zeros(seq_len,dim)
    for i in range(seq_len):
        sin_denom=np.power(n,(2*sin_ind)/dim)
        cos_denom=np.power(n,(2*cos_ind)/dim)
        sin=np.sin(i/sin_denom)
        cos=np.cos(i/cos_denom)
        posE[i]=torch.tensor([item for pair in zip_longest(sin, cos) for item in pair if item is not None])
    return posE.unsqueeze(1).repeat(1,batch_size, 1)

