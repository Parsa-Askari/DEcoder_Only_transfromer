import numpy as np
import pandas as pd
import regex as re
import argparse
from tqdm import tqdm
import os


def cleaner(text):
    text=text.lower()
    text=re.sub(r"\S*@\S*\s?"," ",text)
    text=re.sub(r",+"," _comma_ ",text)
    text=re.sub(r"\.+"," . ",text)
    text=re.sub(r"\$"," ",text)
    text=re.sub(r"#"," ",text)
    text=re.sub(r"="," ",text)
    text=re.sub(r"@"," ",text)
    text=re.sub(r"\d"," ",text)
    text=re.sub(r"~"," ",text)
    text=re.sub(r"\'"," ",text)
    text=re.sub(r">"," ",text)
    text=re.sub(r"<"," ",text)
    text=re.sub(r"\+"," ",text)
    text=re.sub(r"\-"," ",text)
    text=re.sub(r"\/"," ",text)
    text=re.sub(r"\*"," ",text)
    text=re.sub(r"\""," ",text)

    pattern = r"_(.+?)_"
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace(f"_{match}_", f" _{match}_ ")
    return text

# empatheticdialogues

class PreprocessEmpathy:
    def __init__(self,path,output_path="./data/empatheticdialogues/processed/"):
        self.train_data=[]
        self.test_data=[]
        self.valid_data=[]
        self.train_path=path+"train.csv"
        self.valid_path=path+"valid.csv"
        self.test_path=path+"test.csv"
        self.output_path=output_path
        with open(self.train_path , "r") as f:
            self.train_data=f.readlines()[1:]
        with open(self.valid_path , "r") as f:
            self.valid_data=f.readlines()[1:]
        with open(self.test_path , "r") as f:
            self.test_data=f.readlines()[1:]
    def select_columns(self):
        #split columns
        self.train_data=[x.split(',')[5] for x in tqdm(self.train_data)]
        self.test_data=[x.split(',')[5] for x in tqdm(self.test_data)]
        self.valid_data=[x.split(',')[5] for x in tqdm(self.valid_data)]
    def clean_data(self):
        self.train_data=[cleaner(x) for x in tqdm(self.train_data)]
        self.test_data=[cleaner(x) for x in tqdm(self.test_data)]
        self.valid_data=[cleaner(x) for x in tqdm(self.valid_data)]
    def transform(self):
        print("Selecting columns")
        self.select_columns()
        print("Cleaning data")
        self.clean_data()
        print(f"Saving the data to {self.output_path}")
        self.train_data=pd.DataFrame({"texts":self.train_data})
        self.test_data=pd.DataFrame({"texts":self.test_data})
        self.valid_data=pd.DataFrame({"texts":self.valid_data})
        
        self.train_data.to_csv(self.output_path+"train.csv",index=False)
        self.valid_data.to_csv(self.output_path+"valid.csv",index=False)
        self.test_data.to_csv(self.output_path+"test.csv",index=False)

# dailydialog
class PreprocessDailydialog:
    def __init__(self,path,output_path="./data/dailydialog/processed/"):
        self.train_data=[]
        self.test_data=[]
        self.valid_data=[]
        self.train_path=path+"train.txt"
        self.valid_path=path+"valid.txt"
        self.test_path=path+"test.txt"
        self.output_path=output_path
        with open(self.train_path , "r") as f:
            self.train_data=f.readlines()
        with open(self.valid_path , "r") as f:
            self.valid_data=f.readlines()
        with open(self.test_path , "r") as f:
            self.test_data=f.readlines()
    def split_data(self):
        temp=[]
        for x in tqdm(self.train_data):
            temp+=x.split("__eou__")[:-1]
        self.train_data=temp
        
        temp=[]
        for x in tqdm(self.test_data):
            temp+=x.split("__eou__")[:-1]
        self.test_data=temp

        temp=[]
        for x in tqdm(self.valid_data):
            temp+=x.split("__eou__")[:-1]
        self.valid_data=temp

    def clean_data(self):
        self.train_data=[cleaner(x) for x in tqdm(self.train_data)]
        self.test_data=[cleaner(x) for x in tqdm(self.test_data)]
        self.valid_data=[cleaner(x) for x in tqdm(self.valid_data)]
    
    def transform(self):
        print("Selecting columns")
        self.split_data()
        print("Cleaning data")
        self.clean_data()
        print(f"Saving the data to {self.output_path}")
        self.train_data=pd.DataFrame({"texts":self.train_data})
        self.test_data=pd.DataFrame({"texts":self.test_data})
        self.valid_data=pd.DataFrame({"texts":self.valid_data})
        
        self.train_data.to_csv(self.output_path+"train.csv",index=False)
        self.valid_data.to_csv(self.output_path+"valid.csv",index=False)
        self.test_data.to_csv(self.output_path+"test.csv",index=False)




