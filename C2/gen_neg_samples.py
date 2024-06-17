import sys
import io
import subprocess as commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import collections

def getknn(src_, tgt_, tgt_ids, k=10, bsz=1024):
    #rn = 30
    src_ = src_.cuda()
    tgt_ = tgt_.cuda()
    src = src_ / (torch.norm(src_, dim=1, keepdim=True) + 1e-9)
    tgt = tgt_ / (torch.norm(tgt_, dim=1, keepdim=True) + 1e-9)
    num_imgs = len(src)
    confuse_output_indices = []
    confuse_output_indices_long = []
    for batch_idx in range( int( math.ceil( float(num_imgs) / bsz ) ) ):
        start_idx = batch_idx * bsz
        end_idx = min( num_imgs, (batch_idx + 1) * bsz )
        length = end_idx - start_idx
        prod_batch = torch.matmul(src[start_idx:end_idx, :], tgt.T)
        dotprod = torch.topk(prod_batch,k=k+1,dim=1,sorted=True,largest=True).indices
        confuse_output_indices_long += dotprod.cpu().tolist()


    for i in range(len(confuse_output_indices_long)):
        confuse_output_i = confuse_output_indices_long[i]
        if tgt_ids[i] in confuse_output_i:
            confuse_output_i_new = confuse_output_i.copy()
            confuse_output_i_new.remove(tgt_ids[i])
            confuse_output_indices.append(confuse_output_i_new)
        else:
            confuse_output_indices.append(confuse_output_i[:-1])
    return confuse_output_indices

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

def SAVE_DATA(args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup, voc_l1, voc_l2):
    batch_size = args.eval_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)
    train_data_l1_translation = train_data_l1 
    train_data_l2_translation = train_data_l2 

    sup_data_l1_translation = torch.index_select(train_data_l1_translation,0,torch.tensor(l1_idx_sup))
    sup_data_l2_translation = torch.index_select(train_data_l2_translation,0,torch.tensor(l2_idx_sup))

    voc_l1_id2word = {v:k for k,v in voc_l1.items()} 
    voc_l2_id2word = {v:k for k,v in voc_l2.items()} 

    neg_sample = args.num_neg
    neg_max = args.neg_max
    confuse_tgt = getknn(sup_data_l1_translation, train_data_l2_translation[:neg_max], l2_idx_sup, k = neg_sample, bsz=1024) #5000*k
    confuse_src = getknn(sup_data_l2_translation, train_data_l1_translation[:neg_max], l1_idx_sup, k = neg_sample, bsz=1024)  #5000*k
    with open(args.root + "{}2{}_train.txt".format(args.l1, args.l2)  ,"w") as f:
        for i in range(len(l1_idx_sup)):
            l1_word = l1_idx_sup[i]
            l2_word = l2_idx_sup[i]
            l2_conf = confuse_tgt[i]
            l1_conf = confuse_src[i]
            l1_words = [l1_word] + l1_conf
            l2_words = [l2_word] + l2_conf
            l1_words = [voc_l1_id2word[idx] for idx in l1_words]
            l2_words = [voc_l2_id2word[idx] for idx in l2_words]
            l1_words = " ".join(l1_words)
            l2_words = " ".join(l2_words)
            line = str(i)+"|+|"+l1_words+"|+|"+l2_words
            f.write(line+"\n")

def high_conf_pairs(args, train_data_l1, train_data_l2, l1_idx_sup, l2_idx_sup):
    batch_size = args.eval_batch_size
    num_imgs_l1 = len(train_data_l1)
    num_imgs_l2 = len(train_data_l2)


    train_data_l1_translation = train_data_l1.cuda()
    train_data_l2_translation = train_data_l2.cuda()

    l1_idx_aug, l2_idx_aug = generate_new_dictionary_bidirectional(args, train_data_l1_translation, train_data_l2_translation, l1_idx_sup, l2_idx_sup)


    return l1_idx_aug, l2_idx_aug

def get_nn_avg_dist(emb, query, knn):
    bs = 1024
    all_distances = []
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb.T)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances

def generate_new_dictionary_bidirectional(args, emb1_, emb2_, l1_idx_sup, l2_idx_sup):

    emb1 = emb1_ / (torch.norm(emb1_, dim=1, keepdim=True) + 1e-9) #.cuda()
    emb2 = emb2_ / (torch.norm(emb2_, dim=1, keepdim=True) + 1e-9)#.cuda()
    bs = 128
    all_scores_S2T = []
    all_targets_S2T = []
    all_scores_T2S = []
    all_targets_T2S = []
    n_src = args.dico_max_rank
    knn = 10

    average_dist1 = get_nn_avg_dist(emb2, emb1, knn) 
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn) 
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    ## emb1 to emb2
    for i in range(0, n_src, bs):
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist2[None, :])
        best_scores, best_targets = scores.topk(1, dim=1, largest=True, sorted=True)

        all_scores_S2T.append(best_scores.cpu())
        all_targets_S2T.append(best_targets.cpu())

    all_scores_S2T = torch.cat(all_scores_S2T, 0).squeeze(1).tolist()
    all_targets_S2T = torch.cat(all_targets_S2T, 0).squeeze(1).tolist()

    pairs_S2T = [(i, all_targets_S2T[i], all_scores_S2T[i]) for i in range(len(all_scores_S2T))]

    # emb2 to emb1
    for i in range(0, n_src, bs):
        scores = emb1.mm(emb2[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[None, :])
        best_scores, best_targets = scores.topk(1, dim=1, largest=True, sorted=True)

        all_scores_T2S.append(best_scores.cpu())
        all_targets_T2S.append(best_targets.cpu())

    all_scores_T2S = torch.cat(all_scores_T2S, 0).squeeze(1).tolist()
    all_targets_T2S = torch.cat(all_targets_T2S, 0).squeeze(1).tolist()

    pairs_T2S = [(all_targets_T2S[i], i, all_scores_T2S[i]) for i in range(len(all_scores_T2S))]

    pairs_S2T = sorted(pairs_S2T,key=lambda x:x[-1],reverse=True)[:args.num_aug]
    pairs_T2S = sorted(pairs_T2S,key=lambda x:x[-1],reverse=True)[:args.num_aug]

    final_pairs = set()

    S_set = set(l1_idx_sup)
    T_Set = set(l2_idx_sup)

    for i in range(len(pairs_S2T )):

        if (pairs_S2T[i][0] not in S_set) and (pairs_S2T[i][1] not in T_Set) and (len(final_pairs) < args.num_aug_total):
            final_pairs.add((pairs_S2T[i][0], pairs_S2T[i][1]))

        if (pairs_T2S[i][0] not in S_set) and (pairs_T2S[i][1] not in T_Set) and (len(final_pairs) < args.num_aug_total):
            final_pairs.add((pairs_T2S[i][0], pairs_T2S[i][1]))



    final_pairs = list(final_pairs)

    if len(final_pairs) > 0:
        final_s_aug = [a for (a,b) in final_pairs]
        final_t_aug = [b for (a,b) in final_pairs]
    else:
        final_s_aug, final_t_aug = [], []


    return final_s_aug, final_t_aug    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C2 GEN NEG SAMPLES')

    parser.add_argument("--l1", type=str, default=" ",
                    help="l1")
    parser.add_argument("--l2", type=str, default=" ",
                    help="l2")
    parser.add_argument("--num_iter", type=int, default=1,
                    help="num of iterations")
    parser.add_argument("--train_size", type=str, default="5k",
                    help="train dict size")
    parser.add_argument("--root", type=str, default="./",
                    help="save root")
    parser.add_argument("--dico_max_rank", type=int, default=20000,
                    help="dico max rank")
    parser.add_argument("--num_aug", type=int, default=6000,
                    help="num_aug")
    parser.add_argument("--num_neg", type=int, default=10,
                    help="num_neg")
    parser.add_argument("--num_aug_total", type=int, default=4000,
                    help="num_aug_total")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                    help="Batch size For Validation")
    parser.add_argument("--neg_max", type=int, default=150000,
                    help="neg_max")
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

    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    print("Generate Neg Samples")
    sys.stdout.flush()
    l1_voc = args.l1_voc
    l1_emb = args.l1_emb
    l2_voc = args.l2_voc
    l2_emb = args.l2_emb
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir


    l1_voc = np.load(l1_voc, allow_pickle=True).item()
    l2_voc = np.load(l2_voc, allow_pickle=True).item()
    l1_emb = torch.load(l1_emb)
    l2_emb = torch.load(l2_emb)

    l1_emb = l1_emb / (torch.norm(l1_emb, dim=1, keepdim=True) + 1e-9 )
    l2_emb = l2_emb / (torch.norm(l2_emb, dim=1, keepdim=True) + 1e-9 )

    words_src = list(l1_voc.keys())
    words_tgt = list(l2_voc.keys())

    src2tgt, lexicon_size_s2t = load_lexicon_s2t(DIR_TEST_DICT, words_src, words_tgt)
    tgt2src, lexicon_size_t2s = load_lexicon_t2s(DIR_TEST_DICT, words_tgt, words_src)
    print("lexicon_size_s2t, lexicon_size_t2s", lexicon_size_s2t, lexicon_size_t2s)
    if l1_emb.size(1) < 768:
        accuracy_BLI = eval_BLI(l1_emb, l2_emb, src2tgt, lexicon_size_s2t, tgt2src, lexicon_size_t2s)
        print("C1: ", "BLI Accuracy L1 to L2: ", accuracy_BLI[0], "BLI Accuracy L2 to L1: ", accuracy_BLI[1])
        sys.stdout.flush()


    #Load Train
    if args.train_size == "0k":
        l1_idx_sup = []
        l2_idx_sup = []   
    else:
        file = open(DIR_TRAIN_DICT,'r')
        l1_dic = []
        l2_dic = []
        for line in file.readlines():
            pair = line[:-1].split('\t')
            l1_dic.append(pair[0].lower())
            l2_dic.append(pair[1].lower())
        file.close()
        l1_idx_sup = []
        l2_idx_sup = []
        for i in range(len(l1_dic)):
            l1_tok = l1_voc.get(l1_dic[i])
            l2_tok = l2_voc.get(l2_dic[i])
            if (l1_tok is not None) and (l2_tok is not None):
                l1_idx_sup.append(l1_tok)
                l2_idx_sup.append(l2_tok)
    
        print("Sup Set Size: ", len(l1_idx_sup), len(l2_idx_sup))

    #Find High Conf Pairs 

    if args.train_size == "0k" or args.train_size == "1k":
        with torch.no_grad():
            l1_idx_aug, l2_idx_aug = high_conf_pairs(args, l1_emb, l2_emb, l1_idx_sup, l2_idx_sup)
            print("augment ", len(l1_idx_aug), " training pairs")
            sys.stdout.flush()
    else:  # "5k" setup
        l1_idx_aug, l2_idx_aug = [], []
    

    SAVE_DATA(args, l1_emb, l2_emb, l1_idx_sup+l1_idx_aug, l2_idx_sup+l2_idx_aug, l1_voc, l2_voc)
    print("positive-negative pairs for contrastive tuning saved")


