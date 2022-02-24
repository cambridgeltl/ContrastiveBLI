import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
import torch
from metric_learning import *

from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

LOGGER = logging.getLogger()


class Model_Wrapper(object):
    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, output_dir, context=False):
        self.encoder.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):

        #if (path == "google/mt5-small") or (path == "google/mt5-base"):
        #    self.tokenizer = AutoTokenizer.from_pretrained(path,
        #            use_fast=True, do_lower_case=lowercase)
        #    self.encoder = AutoModel.from_pretrained(path)#.encoder
        #else: # read "xlm-mlm-100-1280" and "bert-base-multilingual-uncased"
        #    self.tokenizer = AutoTokenizer.from_pretrained(path,
        #            use_fast=True, do_lower_case=lowercase)
        #    self.encoder = AutoModel.from_pretrained(path)

        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)

        if use_cuda:
            self.encoder = self.encoder.cuda()
        return self.encoder, self.tokenizer
    

