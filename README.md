# Improving Word Translation via Two-Stage Contrastive Learning

This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Fangyu Liu, Nigel Collier, Anna Korhonen, and Ivan Vulić. 2022. *Improving Word Translation via Two-Stage Contrastive Learning*. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022). [[arXiv]](https://arxiv.org/abs/2203.08307)


**ContrastiveBLI** addresses the problem of Bilingual Lexicon Induction (BLI) / Word Translation. Our method consists of two consecutive but independent stages, i.e., C1 and C2: both stages rely on contrastive learning, and each stage can learn its own cross-lingual word embeddings (CLWEs). **Stage C1** uses static word embeddings only (e.g., fastText). <ins>As an independent model, C1 can be evaluated separately and thus can serve as a strong fastText-based baseline for BLI tasks.</ins> **Stage C2** leverages both C1-aligned CLWEs, and a pretrained multilingual LM such as mBERT / XLM / mT5, used as **Bi-encoder / Siamese-encoder**, to further improve the BLI performance. Of course, C2 is compatible with other standard BLI methods: you could instead use, say, VecMap or RCSLS to derive CLWEs which can then replace C1-aligned CLWEs in C2!    

<p align="center">
  <img width="650" src="model.png">
</p>

Our code is tested on both supervised (e.g., with 5k seed translation pairs) and semi-supervised/weakly supervised (e.g., with 1k seed translation pairs) BLI setups. You could specify the seed dictionary size in [./C1/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C1/run_all.py) and [./C2/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C2/run_all.py).

In addition to BLI, we encourage interested readers to try C1-aligned 300-dim (fastText) CLWEs and C2-aligned 768-dim (fastText + mBERT) CLWEs on different downstream cross-lingual transfer learning tasks.

## Follow-up Work:

**Update**: please see our follow-up work [BLICEr](https://github.com/cambridgeltl/BLICEr) (Findings of EMNLP 2022) where we further improve BLI with post-hoc reranking, applicable to any precalculated CLWE space. Specifically, [BLICEr](https://github.com/cambridgeltl/BLICEr) first retrieve contrastive word pairs (positive and negative) and then use the word pairs to fine-tune multilingual LMs into [Cross-encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html#bi-encoder-vs-cross-encoder) that can refine the cross-lingual similarity scores predicted by the CLWEs.

## Dependencies:

- PyTorch >= 1.7.0
- Transformers >= 4.4.2

## Get Data and Set Input/Output Directories:
The data for our main experiments are obtained from the [XLING](https://github.com/codogogo/xling-eval), and we include a simple script to download its BLI dictionaries and preprocessed word embeddings.
```bash
sh get_data.sh
```
We also evaluate BLI for <ins>lower-resource languages</ins> with [PanLex-BLI](https://github.com/cambridgeltl/panlex-bli) data. The code for deriving the monolingual word embeddings is available at [./Get_PanLex_Data](https://github.com/cambridgeltl/ContrastiveBLI/tree/main/Get_PanLex_Data).

Please make sure that the input and output directories are correctly set before running the code. See [./SetDirectories.md](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/SetDirectories.md) for details. 

## Run the Code:
Stage C1 (Training and Evaluation over 28 language pairs in both directions):
```bash
cd C1
sh run_all.sh
```

Stage C2 (Training and Evaluation over 28 language pairs in both directions):
```bash
cd C2
sh run_all.sh
```
Since our method is symmetric, it is not needed to train separate models for source->target translation and target->source translation. Each of Stage C1 and Stage C2 will output 4 scores in a single run, for each language pair. I.e., for a language pair (L1, L2), you will get the following 4 scores together: 
- P@1 via NN retrieval for L1->L2; 
- P@1 via CSLS retrieval for L1->L2; 
- P@1 via NN retrieval for L2->L1; 
- P@1 via CSLS retrieval for L2->L1.


## Environment Setup:

Here is our software environment that we use for our main experiments. Please feel free to skip this part if you would like to adopt different settings. If the software environment changes, the experimental results can slightly fluctuate, but it will not influence the overall robustness. 

Our original implementation uses Nvidia Driver 465.19.01 and depends on the Nvidia official docker image pytorch:20.10-py3 [LINK](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-10.html#rel_20-10) which specifies the following: Ubuntu 18.04, Python 3.6.10, Cuda 11.1.0, cuDNN 8.0.4, NCCL 2.7.8, PyTorch 1.7.0 and TensorRT 7.2.1.

Step 1. Build a docker container:
```bash
sudo docker pull nvcr.io/nvidia/pytorch:20.10-py3
sudo nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -e LANG=en_SG.utf8 -e LANGUAGE=en_SG:en -e LC_ALL=en_SG.utf8 -v [Storage Mapping] -it --ipc host --name [Container Name] [Image ID]
```

Step 2. Install dependancies:
```bash
sudo nvidia-docker start [Container Name]
sudo nvidia-docker attach [Container Name]
git clone https://github.com/cambridgeltl/ContrastiveBLI.git
cd ContrastiveBLI 
sh setup.sh
```

## Tips on Hyper-parameter Search:

When running experiments on a different dataset, on different language pairs or having different BLI settings such as seed dictionary sizes, word embeddings (WEs) or pretrained LMs, doing hyper-parameter search in both Stage C1 and Stage C2 is necessary, whenever a dev set is available.
 
- In C1, you might use valid_every = 10 to track the BLI performance when doing hyper-parameter search. If the BLI accuracy score (on your dev set) in each training epoch obviously drops from some point, then reduce num_games, lr, or gamma in [./C1/src/main.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C1/src/main.py); otherwise you may increase them.

- In C1, when having a different seed dictionary size (other than 1k and 5k), we would recommend to also tune num_sl, num_aug, and dico_max_rank (in [./C1/src/main.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C1/src/main.py)) on your dev set. Besides, you may need to modify sup_batch_size and mini_batch_size.

- lambda_ in [./C2/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C2/run_all.py) is possibly sensitive to typologically distant languages (especially lower-resource languages). We recommend to tune lambda_ in these cases.

## Sample Code A. Encode Words with mBERT(tuned):
Here is a simple example to encode words with mBERT. We uploaded to [huggingface.co/models](https://huggingface.co/models) two mBERT models tuned with BLI-oriented loss for the language pair DE-TR:  [cambridgeltl/c2_mbert_de2tr_5k](https://huggingface.co/cambridgeltl/c2_mbert_de2tr_5k) and [cambridgeltl/c2_mbert_de2tr_1k](https://huggingface.co/cambridgeltl/c2_mbert_de2tr_1k).

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "cambridgeltl/c2_mbert_de2tr_5k" # "cambridgeltl/c2_mbert_de2tr_1k"
maxlen = 6

tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True, do_lower_case=True)
model = AutoModel.from_pretrained(model_name)

words = ["durch","benutzen","tarafından","kullanım"]
toks = tokenizer.batch_encode_plus(words, max_length = maxlen, truncation = True, padding="max_length", return_tensors="pt")      
outputs = model(**toks, output_hidden_states=True).last_hidden_state[:,0,:] 
mbert_features = outputs / (torch.norm(outputs, dim=1, keepdim=True) + 1e-9 )
```

## Sample Code B. Mix C1 & mBERT Features: 
Suppose we have C1-aligned CLWEs and mbert(tuned) features for a source langugae and a target language respectively. In Stage C2, we comebine them as follows.
```python
lambda = 0.2
# c1_features_source and c1_features_target, of size (n, 768), are C1-aligned CLWEs already mapped from the original 300-dim space (fastText) to a 768-dim space (mBERT) via Procrustes, normalised.
c2_features_source = (1.0 - lambda) * c1_features_source  + lambda * mbert_features_source 
c2_features_target = (1.0 - lambda) * c1_features_target  + lambda * mbert_features_target

# then normalise them
c2_features_source = c2_features_source / (torch.norm(c2_features_source, dim=1, keepdim=True) + 1e-9 )
c2_features_target = c2_features_target / (torch.norm(c2_features_target, dim=1, keepdim=True) + 1e-9 )
```
## Sample Code C. Word Translation:
Here is a simple implementation of source->target word translation via NN retrieval (for CSLS retrieval, see [./C1/util.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C1/src/util.py)). Note that Stage C1 can be evaluated independently.
```python
# Stage C1: c1_features_source and c1_features_target are of size (n, 300) before Procrustes mapping, normalised. 
sims_source_to_target = c1_features_source @ c1_features_target.T
target_predict = torch.argmax(sims_source_to_target, dim=1)
 
# Stage C2: c2_features_source and c2_features_target are of size (n, 768), normalised.
sims_source_to_target = c2_features_source @ c2_features_target.T
target_predict = torch.argmax(sims_source_to_target, dim=1)
```

## Baseline Methods:

The four baselines covered in our experiments are [RCSLS](https://github.com/facebookresearch/fastText/tree/main/alignment), [VecMap](https://github.com/artetxem/vecmap), [LNMap](https://github.com/taasnim/lnmap) and [FIPP](https://github.com/vinsachi/FIPPCLE). When running these baseline models, we followed their own original settings and hyperparamter values suggested in their respective repositories: e.g., using each method's own word embedding preprocessing method, self-learning algorithm (each of VecMap, LNMap and FIPP has its own version of self-learning), hyperparameter values and other settings recommended respectively for supervised (5k) and semi-supervised (1k) tasks (e.g., VecMap and FIPP recommend to switch off self-learning in supervised, i.e., 5k, setups). We verified that these suggested settings yield (near-)optimal BLI performance.  

## Known Issues:

It is reported that T5/mT5 produces "NaN" outputs under mixed-precision or fp16 mode when using some Transformer versions ([ISSUE](https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139)). Our code also suffers from this issue. When running Stage C2 with mT5, we recommend to switch off torch.cuda.amp (Automatic Mixed Precision) by commenting out line 54 in [./C2/src/metric_learning.py](https://github.com/cambridgeltl/ContrastiveBLI/blob/main/C2/src/metric_learning.py).

## Acknowledgements:

Part of our code is adapted from the following GitHub repos: [XLING](https://github.com/codogogo/xling-eval), [RCSLS](https://github.com/facebookresearch/fastText/tree/main/alignment), [VecMap](https://github.com/artetxem/vecmap), [Mirror-BERT](https://github.com/cambridgeltl/mirror-bert) and [ECNMT](https://github.com/cambridgeltl/ECNMT). 

## Citation:
Please cite our paper if you find **ContrastiveBLI** useful. If you like our work, please ⭐ this repo.
```bibtex
@inproceedings{li-etal-2022-improving,
    title     = {Improving Word Translation via Two-Stage Contrastive Learning},
    author    = {Li, Yaoyiran and Liu, Fangyu and Collier, Nigel and Korhonen, Anna and Vulić, Ivan},
    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},    
    year      = {2022}
}
```

Follow-up Work:
```bibtex
@inproceedings{li-etal-2022-bilingual,
    title     = {Improving Bilingual Lexicon Induction with Cross-Encoder Reranking},
    author    = {Li, Yaoyiran and Liu, Fangyu and Vulić, Ivan and Korhonen, Anna},
    booktitle = {In Findings of the Association for Computational Linguistics: EMNLP 2022},    
    year      = {2022}
}
```
