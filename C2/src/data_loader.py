import re
import os
import glob
import numpy as np
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import torch
LOGGER = logging.getLogger(__name__)



def erase_and_mask(s, erase_len=5):
    if len(s) <= erase_len: return s
    if len(s) < 20: return s 
    ind = np.random.randint(len(s)-erase_len)
    left, right = s.split(s[ind:ind+erase_len], 1)
    return " ".join([left, "[MASK]", right])
    

class C2_Dataset(Dataset):
    def __init__(self, path, l1, l2, l1_voc, l2_voc, tokenizer, random_erase=0, template = 0): 
        with open(path, 'r') as f:
            lines = f.readlines()

        self.str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish","bg":"bulgarian","ca":"catalan","hu":"hungarian","eu":"basque","et":"estonian","he":"hebrew"}
        self.l1 = l1
        self.l2 = l2
        self.template = template
        self.my_template = "the word '{}' in {}."
        self.l1_voc = np.load(l1_voc, allow_pickle=True).item()
        self.l2_voc = np.load(l2_voc, allow_pickle=True).item()

        self.query_ids = []
        self.query_names = []
        self.idxs = []
        for i,line in enumerate(lines):
            line = line.rstrip("\n")
            query_id, name1, name2 = line.split("|+|")
            self.query_ids.append(query_id)
            name1_, name2_ = name1.split(), name2.split()
            if bool(self.template):
                name1_ = [self.my_template.format(w,self.str2lang[self.l1]) for w in name1_]
                name2_ = [self.my_template.format(w,self.str2lang[self.l2]) for w in name2_]
            
            self.query_names.append((name1_, name2_))
            self.idxs.append(i)

        self.tokenizer = tokenizer
        self.query_id_2_index_id = {k: v for v, k in enumerate(list(set(self.query_ids)))}
        self.random_erase = random_erase
    
    def __getitem__(self, query_idx):

        query_name1 = self.query_names[query_idx][0]
        query_name2 = self.query_names[query_idx][1]

        idx = self.idxs[query_idx]
        if self.random_erase != 0:
            query_name2 = erase_and_mask(query_name2, erase_len=int(self.random_erase))
        query_id = self.query_ids[query_idx]
        query_id = int(self.query_id_2_index_id[query_id])

        return query_name1, query_name2, query_id


    def __len__(self):
        return len(self.query_names)




