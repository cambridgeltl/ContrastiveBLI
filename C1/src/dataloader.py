import copy
import json
import operator
import pickle as pkl
import numpy as np
from collections import OrderedDict
import time
import torch
from torch.autograd import Variable
from util import *
random = np.random



def next_batch_joint_sup_confuse(confuse_src, confuse_tgt, features_l1, features_l2, batch_size, args):
    sup_index = np.arange(len(confuse_src)) 
    sup_index = torch.tensor(sup_index)
    src_input = torch.index_select(confuse_src,0,sup_index).reshape(-1) 
    trg_input = torch.index_select(confuse_tgt,0,sup_index).reshape(-1)


    src_input = torch.index_select(features_l1,0,src_input) 
    trg_input = torch.index_select(features_l2,0,trg_input)

    src_input = src_input.reshape(-1,args.num_sample+1,args.D_emb)
    trg_input = trg_input.reshape(-1,args.num_sample+1,args.D_emb)

    return src_input,trg_input

