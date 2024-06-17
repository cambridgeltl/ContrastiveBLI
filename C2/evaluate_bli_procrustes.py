import pandas as pd
import random
import time
import torch
import string
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy import spatial
import math  
from scipy.stats.stats import pearsonr,spearmanr
import io
import collections
import sys
import io
import subprocess as commands
import codecs
import copy
import argparse


def procrustes(X_src, Y_tgt):
    X_dim = X_src.shape[1]
    Y_dim = Y_tgt.shape[1]
    U, s, V = np.linalg.svd(np.dot(X_src.T, Y_tgt))
    if X_dim == Y_dim:
        return np.dot(U, V)
    elif X_dim < Y_dim:
        zeros_pad = np.zeros([X_dim,Y_dim-X_dim])
        U = np.concatenate((U,zeros_pad),axis=1)
        return np.dot(U, V)
    else:
        zeros_pad = np.zeros([X_dim-Y_dim, Y_dim])
        V = np.concatenate((V,zeros_pad),axis=0)
        return np.dot(U, V)

def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i

def load_lexicon_s2t(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        word_src, word_tgt = word_src.lower(), word_tgt.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))

def load_lexicon_t2s(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_tgt, word_src = line.split()
        word_tgt, word_src = word_tgt.lower(), word_src.lower()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def compute_nn_accuracy_torch(x_src, x_tgt, lexicon, bsz=256, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src_ = x_src / (torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
    x_tgt_ = x_tgt / (torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = torch.matmul(x_tgt_, x_src_[idx_src[i:e]].T)
        pred = torch.argmax(scores,dim=0)
        pred = pred.cpu().numpy()
        for j in range(i, e):
            if pred[j - i] in lexicon[idx_src[j]]:
                acc += 1.0
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=256):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    x_src_ =x_src /(torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
    x_tgt_ =x_tgt /(torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
    sr = x_src_[idx_src]
    sc = torch.zeros(sr.size(0),x_tgt_.size(0)).cuda()
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        sc_ = torch.matmul(x_tgt_, sr[i:e].T)
        sc[i:e] = sc_.T
    similarities = 2 * sc
    sc2 = torch.zeros(x_tgt_.size(0)).cuda()
    for i in range(0, x_tgt_.size(0), bsz):
        j = min(i + bsz, x_tgt_.size(0))
        sc_batch = torch.matmul(x_tgt_[i:j,:], x_src_.T)
        dotprod = torch.topk(sc_batch,k=k,dim=1,sorted=False).values
        sc2[i:j] = torch.mean(dotprod, dim=1)
    similarities -= sc2.unsqueeze(0)

    nn = torch.argmax(similarities, dim=1).cpu().tolist()
    correct = 0.0
    for k in range(0, len(lexicon)):
        if nn[k] in lexicon[idx_src[k]]:
            correct += 1.0
    return correct / lexicon_size

def eval_BLI(train_data_l1, train_data_l2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s):

    train_data_l1_translation = train_data_l1.cuda()
    train_data_l2_translation = train_data_l2.cuda()
    acc_s2t = compute_nn_accuracy_torch(train_data_l1_translation, train_data_l2_translation, src2tgt, lexicon_size=-1) 
    cslsacc_s2t = compute_csls_accuracy(train_data_l1_translation, train_data_l2_translation, src2tgt, lexicon_size=-1)  

    acc_t2s = compute_nn_accuracy_torch(train_data_l2_translation, train_data_l1_translation, tgt2src, lexicon_size=-1) 
    cslsacc_t2s = compute_csls_accuracy(train_data_l2_translation, train_data_l1_translation, tgt2src, lexicon_size=-1)

    BLI_accuracy_l12l2 = (acc_s2t, cslsacc_s2t)
    BLI_accuracy_l22l1 = (acc_t2s, cslsacc_t2s)
    return (BLI_accuracy_l12l2, BLI_accuracy_l22l1) 


def SAVE_DATA(args, train_data_l1, train_data_l2, voc_l1, voc_l2):
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)
    train_data_l1_translation = train_data_l1 
    train_data_l2_translation = train_data_l2 

    np.save(args.root + "{}2{}_{}_voc.npy".format(args.l1,args.l2,args.l2) , voc_l2)
    np.save(args.root + "{}2{}_{}_voc.npy".format(args.l1,args.l2,args.l1), voc_l1)
    torch.save(train_data_l1_translation,args.root + "{}2{}_{}_emb.pt".format(args.l1,args.l2,args.l1)) #aligned l1 WEs
    torch.save(train_data_l2_translation,args.root + "{}2{}_{}_emb.pt".format(args.l1,args.l2,args.l2)) #aligned l2 WEs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C2 EVALUATION')

    parser.add_argument("--l1", type=str, default=" ",
                    help="l1")
    parser.add_argument("--l2", type=str, default=" ",
                    help="l2")
    parser.add_argument("--train_size", type=str, default="5k",
                    help="training dictionary size")
    parser.add_argument("--root", type=str, default="./",
                    help="save root")
    parser.add_argument("--model_name", type=str, default="./",
                    help="model name")
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean_pool}")
    parser.add_argument('--template', default=0, type=int)
    parser.add_argument('--max_length', default=5, type=int)
    parser.add_argument('--l1_voc', type=str, required=True,
                        help='Directory of L1 Vocabulary')
    parser.add_argument('--l1_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L1')
    parser.add_argument('--l2_voc', type=str, required=True,
                        help='Directory of L2 Vocabulary')
    parser.add_argument('--l2_emb', type=str, required=True,
                        help='Directory of Aligned Static Embeddings for L2')
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train dict directory")
    parser.add_argument("--test_dict_dir", type=str, default="./",
                    help="test dict directory")
    parser.add_argument('--lambda_', type=float, default=0.2,
                        help='lambda_',)
    parser.add_argument("--origin_model_name", type=str, default="./",
                    help="model name")

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    print("Evaluate BLI")
    str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish","bg":"bulgarian","ca":"catalan","hu":"hungarian","eu":"basque","et":"estonian","he":"hebrew"}
    my_template = "the word '{}' in {}."

    l1_voc = args.l1_voc
    l1_emb = args.l1_emb
    l2_voc = args.l2_voc
    l2_emb = args.l2_emb
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir
                     
    train_path = args.root + "/{}2{}_train.txt".format(args.l1,args.l2) 
    model_name = args.model_name
    print(model_name)
    agg_mode = args.agg_mode
    if agg_mode != "cls":
        agg_mode = "mean-tok"
    device = 0
    bsz = 128
    maxlen = args.max_length
    procrustes_flag = "train"
    norm_mbert = True


    l1_voc = np.load(l1_voc, allow_pickle=True).item()
    l2_voc = np.load(l2_voc, allow_pickle=True).item()
    l1_emb = torch.load(l1_emb)
    l2_emb = torch.load(l2_emb)

    feature_size = l1_emb.size(1) 
    print("feature_size: ",feature_size)

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    words_src = list(l1_voc.keys())
    words_tgt = list(l2_voc.keys())


    words_l1 = []
    words_l2 = []
    ids_l1 = []
    ids_l2 = []

    with open(train_path,"r") as f:
        for line in f.readlines():
            _, l1_words, l2_words = line.split("|+|")
            l1_word = l1_words.split()[0]
            l2_word = l2_words.split()[0]
            words_l1.append(l1_word)
            words_l2.append(l2_word)
            ids_l1.append(l1_voc[l1_word])
            ids_l2.append(l2_voc[l2_word])



    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)


    if (args.origin_model_name == "google/mt5-small") or (args.origin_model_name == "google/mt5-base"):
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True, do_lower_case=True)
        model = AutoModel.from_pretrained(model_name).get_encoder().cuda(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True, do_lower_case=True)
        model = AutoModel.from_pretrained(model_name).cuda(device)


    if bool(args.template):
        words_src_template = [my_template.format(w,str2lang[args.l1]) for w in words_src]
        words_tgt_template = [my_template.format(w,str2lang[args.l2]) for w in words_tgt]
    else:
        words_src_template, words_tgt_template = words_src, words_tgt

    string_features1, string_features2 = [], []
    for i in tqdm(np.arange(0, len(words_src), bsz)):
        toks1 = tokenizer.batch_encode_plus(words_src_template[i:i+bsz], max_length = maxlen,
                                    truncation = True, padding="max_length", return_tensors="pt")                           
        toks1_cuda = {k:v.cuda(device) for k,v in toks1.items()}
        with torch.no_grad():
            outputs_1 = model(**toks1_cuda, output_hidden_states=True)
        last_hidden_state1 = outputs_1.last_hidden_state
        if agg_mode == "mean-tok":
            np_feature_mean_tok1 = last_hidden_state1.detach().cpu().mean(1) 
        elif agg_mode == "cls":
            np_feature_mean_tok1 = last_hidden_state1.detach().cpu()[:,0,:] 
        string_features1.append(np_feature_mean_tok1)

    string_features1 = torch.cat(string_features1, dim=0)

    for i in tqdm(np.arange(0, len(words_tgt), bsz)):
        toks2 = tokenizer.batch_encode_plus(words_tgt_template[i:i+bsz], max_length = maxlen,
                                    truncation = True, padding="max_length", return_tensors="pt")                           
        toks2_cuda = {k:v.cuda(device) for k,v in toks2.items()}
        with torch.no_grad():
            outputs_2 = model(**toks2_cuda, output_hidden_states=True)
        last_hidden_state2 = outputs_2.last_hidden_state
        if agg_mode == "mean-tok":
            np_feature_mean_tok2 = last_hidden_state2.detach().cpu().mean(1)
        elif agg_mode == "cls":
            np_feature_mean_tok2 = last_hidden_state2.detach().cpu()[:,0,:] 
        string_features2.append(np_feature_mean_tok2)

    string_features2 = torch.cat(string_features2, dim=0)

    if norm_mbert:
        string_features1 = string_features1 / (torch.norm(string_features1, dim=1, keepdim=True) + 1e-9 )
        string_features2 = string_features2 / (torch.norm(string_features2, dim=1, keepdim=True) + 1e-9 )




    if procrustes_flag == "frequent":
        ft_features = torch.cat([l1_emb[:5000],l2_emb[:5000]],dim=0).numpy()
        mbert_features = torch.cat([string_features1[:5000],string_features2[:5000]],dim=0).numpy()
    elif procrustes_flag == "train":
        print("train vectors procrustes")
        ft_embeddings1 = torch.index_select(l1_emb,0,torch.tensor(ids_l1))
        ft_embeddings2 = torch.index_select(l2_emb,0,torch.tensor(ids_l2))
        ft_features = torch.cat([ft_embeddings1, ft_embeddings2],dim = 0).numpy()

        mbert_embeddings1 = torch.index_select(string_features1,0,torch.tensor(ids_l1))
        mbert_embeddings2 = torch.index_select(string_features2,0,torch.tensor(ids_l2))
        mbert_features = torch.cat([mbert_embeddings1, mbert_embeddings2],dim = 0).numpy()    


    linear_model = procrustes(ft_features,  mbert_features)


    if args.origin_model_name == "xlm-mlm-100-1280":
        linear = torch.nn.Linear(feature_size,1280,bias=False)
    elif args.origin_model_name == "google/mt5-small":
        linear = torch.nn.Linear(feature_size,512,bias=False)
    else: # for "bert-base-multilingual-uncased" and "google/mt5-base"
        linear = torch.nn.Linear(feature_size,768,bias=False)

    linear_model = torch.tensor(linear_model,dtype=torch.float32)
    linear.weight.data = linear_model.t()
    linear.weight.requires_grad=False

    l1_emb = linear(l1_emb)
    l2_emb = linear(l2_emb)

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    mix_1 = (1.0 - args.lambda_) * l1_emb + args.lambda_ * string_features1
    mix_2 = (1.0 - args.lambda_) * l2_emb + args.lambda_ * string_features2

    mix_1 = mix_1 / (torch.norm(mix_1, dim=1, keepdim=True) + 1e-9 )
    mix_2 = mix_2 / (torch.norm(mix_2, dim=1, keepdim=True) + 1e-9 )
    accuracy_BLI = eval_BLI(mix_1, mix_2, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s)
    print("C2: ", "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1], " {} <---> {}".format(args.l1,args.l2))

