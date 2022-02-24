import time
import random
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F


def forward_joint(sup_batch, model, loss_dict_, args, mode):
 
    loss = model(sup_batch, mode)

    if mode == "train":
        loss_dict_["loss"].update(loss.data)
    return loss


