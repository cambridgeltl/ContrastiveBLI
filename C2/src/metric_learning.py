import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
LOGGER = logging.getLogger(__name__)



class mBERT_pairwise(nn.Module):
    def __init__(self, mbert_model, agg_mode):
        super(mBERT_pairwise, self).__init__()
        self.encoder = mbert_model
        self.agg_mode = agg_mode
    def forward(self,query_toks1, query_toks2):
        outputs1 = self.encoder(**query_toks1, return_dict=True, output_hidden_states=False)
        outputs2 = self.encoder(**query_toks2, return_dict=True, output_hidden_states=False)

        last_hidden_state1 = outputs1.last_hidden_state
        last_hidden_state2 = outputs2.last_hidden_state

        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  
            query_embed2 = last_hidden_state2[:,0]  
        elif self.agg_mode == "mean_pool":
            query_embed1 = last_hidden_state1.mean(1)  
            query_embed2 = last_hidden_state2.mean(1)

        return query_embed1, query_embed2

class C2_Metric_Learning(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, agg_mode="cls", num_neg = 10, batch_size=50, infoNCE_tau=0.1):

        LOGGER.info("C2_Metric_Learning: learning_rate={} weight_decay={} use_cuda={} infoNCE_tau={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,infoNCE_tau,agg_mode
        ))
        super(C2_Metric_Learning, self).__init__()
        self.model_pairwise = mBERT_pairwise(encoder, agg_mode)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.model_pairwise.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.infoNCE_tau = infoNCE_tau
        self.num_neg = num_neg
        self.batch_size = batch_size
    
    @autocast() # comment this line when using mt5 ! 
    def forward(self, query_toks1, query_toks2):

        query_embed1, query_embed2 = self.model_pairwise(query_toks1, query_toks2)
        query_embed1 = query_embed1 / (torch.norm(query_embed1 , dim=1, keepdim=True) + 1e-9 )
        query_embed2 = query_embed2 / (torch.norm(query_embed2 , dim=1, keepdim=True) + 1e-9 )
        neg_size = self.num_neg
        batch_size = self.batch_size 
        emb_size = query_embed1.size(-1)
 
        query_embed1 = query_embed1.contiguous().view(batch_size, neg_size + 1, emb_size)
        query_embed2 = query_embed2.contiguous().view(batch_size, neg_size + 1, emb_size)

        src_mid_anchor = query_embed1[:,0,:].unsqueeze(1)
        trg_mid_anchor = query_embed2[:,0,:].unsqueeze(1)
        output_1 = query_embed1
        output_2 = query_embed2
        compare_1 = trg_mid_anchor
        sum_1 = torch.sum(output_1 * compare_1, dim=-1)
        compare_2 = src_mid_anchor
        sum_2 = torch.sum(output_2 * compare_2, dim=-1)
        sum_3 = torch.cat([sum_2,sum_1[:,1:]],dim=-1) / self.infoNCE_tau
        sum_3 = torch.exp(sum_3)
        combined_loss = -torch.mean(torch.log(sum_3[:,0]/torch.sum(sum_3,dim=1))) 

        return combined_loss

