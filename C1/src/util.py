import pprint
import codecs
import os
import sys
import time
import pickle as pkl
import numpy as np
from collections import OrderedDict
import io
import torch
from torch.autograd import Variable
import collections


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

    emb1 = emb1_ / (torch.norm(emb1_, dim=1, keepdim=True) + 1e-9) 
    emb2 = emb2_ / (torch.norm(emb2_, dim=1, keepdim=True) + 1e-9)
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
    
    pairs_all = pairs_S2T + pairs_T2S

    final_pairs = set()

    S_set = set(l1_idx_sup)
    T_Set = set(l2_idx_sup)

    for i in range(len(pairs_all)):
        if (pairs_all[i][0] not in S_set) and (pairs_all[i][1] not in T_Set):
            final_pairs.add((pairs_all[i][0], pairs_all[i][1]))

    final_pairs = list(final_pairs)

    if len(final_pairs) > 0:    
        final_s_aug = [a for (a,b) in final_pairs]
        final_t_aug = [b for (a,b) in final_pairs]
    else:
        final_s_aug, final_t_aug = [], []


    return final_s_aug, final_t_aug



def whitening_transformation(m):
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    return vt.T.dot(np.diag(1/s)).dot(vt)

def AdvancedMapping(xw,zw):
    assert xw.shape[1] == zw.shape[1]
    assert xw.shape[0] == zw.shape[0]
    WX_FINAL = np.empty([xw.shape[1],xw.shape[1]],dtype=np.float64)
    WZ_FINAL = np.empty([xw.shape[1],xw.shape[1]],dtype=np.float64)
    wx1 = whitening_transformation(xw)
    wz1 = whitening_transformation(zw)

    WX_FINAL[:] = wx1
    WZ_FINAL[:] = wz1
    xw = xw.dot(wx1)
    zw = zw.dot(wz1)

    wx2, s, wz2_t = np.linalg.svd(xw.T.dot(zw))
    wz2 = wz2_t.T

    WX_FINAL = np.dot(WX_FINAL,wx2)
    WZ_FINAL = np.dot(WZ_FINAL,wz2)

    xw = xw.dot(wx2)
    zw = zw.dot(wz2)

    WX_FINAL *= s**0.5
    WZ_FINAL *= s**0.5

    WX_FINAL = np.dot(WX_FINAL, wx2.T.dot(np.linalg.inv(wx1)).dot(wx2))
    WZ_FINAL = np.dot(WZ_FINAL, wz2.T.dot(np.linalg.inv(wz1)).dot(wz2))
    return WX_FINAL.T,WZ_FINAL.T


def cal_new_loss(current_dict, pretrained_dict, num_iter = 1):
    diff_loss = 0
    num_of_para = 0
    for it in range(num_iter):
        for item in current_dict.keys():
            if item in pretrained_dict:
                diff = pretrained_dict[item] - current_dict[item]
                k = 1
                for i in diff.size():
                    k *= i
                num_of_para += k
                prod_ = diff * diff 
                diff_loss += torch.sum(prod_)
    return diff_loss


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = np.zeros([n, d])
    y = np.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y

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


def load_pairs(filename, idx_src, idx_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    pairs = []
    tot = 0
    for line in f:
        if len(line.rstrip().split(' ')) > 1:
            a, b = line.rstrip().split(' ')
        else:
            a, b = line.rstrip().split('\t')
        tot += 1
        a,b = a.lower(),b.lower()
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if verbose:
        coverage = (1.0 * len(pairs)) / tot
        print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
    return pairs


def compute_nn_accuracy_torch(x_src, x_tgt, lexicon, args, model, bsz=1024, lexicon_size=-1):
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


def compute_csls_accuracy(x_src, x_tgt, lexicon, args, model, lexicon_size=-1, k=10, bsz=1024):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src_ =x_src /(torch.norm(x_src, dim=1, keepdim=True) + 1e-9)
    x_tgt_ =x_tgt /(torch.norm(x_tgt, dim=1, keepdim=True) + 1e-9)
    sr = x_src_[idx_src]
    sc = torch.zeros(sr.size(0),x_tgt_.size(0)).to(args.device)
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        sc_ = torch.matmul(x_tgt_, sr[i:e].T)
        sc[i:e] = sc_.T
    similarities = 2 * sc
    sc2 = torch.zeros(x_tgt_.size(0)).to(args.device)
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



def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

def load_embs(path, topk = None, dimension = None):
  print(topk)
  print("Loading embeddings")
  vocab_dict = {}
  embeddings = []
  with codecs.open(path, encoding = 'utf8', errors = 'replace') as f:
      line = f.readline().strip().split()
      cntr = 1
      if len(line) == 2:
        vocab_size = int(line[0])
        if not dimension: 
          dimension = int(line[1])
      else: 
        if not dimension or (dimension and len(line[1:]) == dimension):
          vocab_dict[line[0].strip()] = len(vocab_dict)
          embeddings.append(np.array(line[1:], dtype=np.float32))
        if not dimension:
          dimension = len(line) - 1
      print("Vector dimensions: " + str(dimension))
      while line: 
        line = f.readline().strip().split() 
        if (not line):
          print("Loaded " + str(cntr) + " vectors.") 
          break

        if line[0].strip() == "":
          continue
        
        cntr += 1
        if cntr % 20000 == 0:
          print(cntr)

        if len(line[1:]) == dimension:
          if (line[0].strip().lower() not in vocab_dict): 
              vocab_dict[line[0].strip().lower()] = len(vocab_dict) 
              embeddings.append(np.array(line[1:], dtype=np.float32))
        else: 
          print("Error in the embeddings file, line " + str(cntr) + 
                             ": unexpected vector length (expected " + str(dimension) + 
                             " got " + str(len(np.array(line[1:]))) + " for word '" + line[0] + "'")
        
        if (topk and cntr >= topk): 
          print("Loaded " + str(cntr) + " vectors.") 
          break           

  embeddings = np.array(embeddings, dtype=np.float32)
  print(len(vocab_dict), str(embeddings.shape))
  return vocab_dict, embeddings

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_log_loss_dict_():
    return {"loss":AverageMeter()}

def get_avg_from_loss_dict_(log_loss_dict):
    res = {} 
    for k, v in log_loss_dict.items():
        res[k] = v.avg
    return res


def print_loss_(epoch, avg_loss_dict, mode="train"):
    prt_msg = "epoch{:5d} {}".format(epoch, mode)
    prt_msg += "|loss"
    prt_msg += " {:.8f}".format(avg_loss_dict["loss"])
    prt_msg += "|"
    return prt_msg


