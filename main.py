import pandas as pd
from libraries import read_tokenized_data , dataset ,collect_fn ,TransformerModel
from sklearn.metrics import accuracy_score
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='main model runner')
def str2bool(v):
    return v.lower()=="true"
parser.add_argument('--tokenized_path', default="./data/tokenized", type=str, help='the path to tokenized data .default : ./data/tokenized')
parser.add_argument('--batch_size', default=32, type=int, help='the batch size for dataloader .default : 32')
parser.add_argument('--embedding_dim', default=100, type=int, help='the size of the embedding vector .default : 100')
parser.add_argument('--hidden_size', default=256, type=int, help='the number of hidden layers .default : 256')
parser.add_argument('--device', default="cuda", type=str, help='the device .default : cuda')
parser.add_argument('--epochs', default=10, type=int, help='the number of epochs .default : 10')
parser.add_argument('--lr', default=0.001, type=float, help='the learning rate of train function .default : 0.001')
parser.add_argument('--optimizer', default="adam", type=str, help='the optimizer .default : adam')
args = parser.parse_args()

tokenized_path=args.tokenized_path
batch_size=args.batch_size
train,valid,test,vocab,Index,PAD,EOS,SOS,UKN,max_size = read_tokenized_data(tokenized_path)

#Create Datasets

train_ds=dataset(train)
valid_ds=dataset(valid)
# test_ds=dataset(test,is_test=True)

#Create Dataloaders

cl=partial(collect_fn, pad=Index[PAD])
trainLoader=DataLoader(train_ds,batch_size=batch_size,collate_fn=cl)
ValidLoader=DataLoader(valid_ds,batch_size=batch_size,collate_fn=cl)
# testLoader=DataLoader(test_ds,batch_size=batch_size,collate_fn=cl)


embedding_dim=args.embedding_dim
hidden_size=args.hidden_size
vocab_size=len(vocab)
device=args.device
epochs=args.epochs
lr=args.lr
# create Transformer
model=TransformerModel(embedding_dim=embedding_dim,
                 hidden_size=hidden_size,
                 vocab_size=vocab_size,
                 device=device)

# compile model
metric=accuracy_score
optimizer=args.optimizer
loss_fn=nn.CrossEntropyLoss()
model.Compile(metric=metric,
              optimizer=optimizer,
              loss_fn=loss_fn,
              lr=lr
              )

#start train function

model.Train(trainLoader=trainLoader,
            validLoader=ValidLoader,
            epochs=epochs)
