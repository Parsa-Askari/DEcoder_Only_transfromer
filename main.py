import pandas as pd
from libraries import read_tokenized_data , dataset ,collect_fn ,TransformerModel
from sklearn.metrics import accuracy_score
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader

tokenized_path="./data/tokenized/"
batch_size=32
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


embedding_dim=100
hidden_size=256
vocab_size=len(vocab)
device="cpu"
epochs=1
# create Transformer
model=TransformerModel(embedding_dim=embedding_dim,
                 hidden_size=hidden_size,
                 vocab_size=vocab_size)

# compile model
metric=accuracy_score
optimizer="adam"
loss_fn=nn.CrossEntropyLoss()
model.Compile(metric=metric,
              optimizer=optimizer,
              loss_fn=loss_fn
              )

#start train function

model.Train(trainLoader=trainLoader,
            validLoader=ValidLoader,
            epochs=1)



