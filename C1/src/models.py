import time
import operator
import math
import sys
import os
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *


class C1_Model(torch.nn.Module):
    def __init__(self, args):
        super(C1_Model, self).__init__()
        self.beholder  = Beholder(args)
        self.D_emb = args.D_emb
        self.sup_batch_size = args.mini_batch_size
        self.norm_input = args.norm_input
        self.s2m_mapping = nn.Linear(args.D_emb, args.D_emb, bias=False)
        self.t2m_mapping = nn.Linear(args.D_emb, args.D_emb, bias=False)
        self.resnet = args.resnet

    def forward(self, sup_batch, mode="train"):
        loss_repel = self.forward_sup(sup_batch,mode)
        return loss_repel


    def forward_sup(self, wv_batch, mode="train"):
        src_input,tgt_input = wv_batch[0].data, wv_batch[1].data
        
        sup_batch_size = src_input.size(0)
        conf_size = src_input.size(1)

        src_input_ = self.beholder(src_input)
        tgt_input_ = self.beholder(tgt_input)

        if self.resnet:
            src_mid = src_input_ + self.s2m_mapping(src_input_)
            tgt_mid = tgt_input_ + self.t2m_mapping(tgt_input_)
        else:
            src_mid = self.s2m_mapping(src_input_)
            tgt_mid = self.t2m_mapping(tgt_input_)
 
        if self.norm_input:
            tgt_mid = tgt_mid / (torch.norm(tgt_mid, dim=-1, keepdim=True).detach() + 1e-9)
            src_mid = src_mid / (torch.norm(src_mid, dim=-1, keepdim=True).detach() + 1e-9)

 
        src_mid_anchor = src_mid[:,0,:].unsqueeze(1)
        tgt_mid_anchor = tgt_mid[:,0,:].unsqueeze(1)

        if mode == "train":

            output_1 = src_mid
            output_2 = tgt_mid            

            compare_1 = tgt_mid_anchor
            sum_1 = torch.sum(output_1 * compare_1, dim=-1)


            compare_2 = src_mid_anchor
            sum_2 = torch.sum(output_2 * compare_2, dim=-1)

            sum_3 = torch.cat([sum_2,sum_1[:,1:]],dim=-1)


            sum_3 = torch.exp(sum_3)
            combined_loss = -torch.mean(torch.log(sum_3[:,0]/torch.sum(sum_3,dim=1))) 
        else:
            combined_loss = 0

        return combined_loss

    def eval_src2mid(self, input_batch, mode="eval"):

        src_input = input_batch.data
        if self.resnet:
            mid_output = src_input + self.s2m_mapping(src_input)
        else:
            mid_output = self.s2m_mapping(src_input)
        return mid_output
    def eval_tgt2mid(self, input_batch, mode="eval"):

        tgt_input = input_batch.data
        if self.resnet:
            mid_output = tgt_input + self.t2m_mapping(tgt_input)
        else:
            mid_output = self.t2m_mapping(tgt_input)
        return mid_output



class Beholder(torch.nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, features):
        h = self.drop(features)
        return h

