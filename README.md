# Improving Word Translation via Two-Stage Contrastive Learning

This repository is the official PyTorch implementation of the following paper: 

Yaoyiran Li, Fangyu Liu, Nigel Collier, Anna Korhonen and Ivan Vulić. 2022. *Improving Word Translation via Two-Stage Contrastive Learning*. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022). [LINK](https://openreview.net/pdf?id=ycgOlOnbbMq)

![C2](model.png "C2")

The work addresses the problem of Bilingual Lexicon Induction (BLI) / Word Translation. Our method consists of two stages: C1 and C2. Stage C1 uses static word embeddings (e.g. fastText) only. As an independent model, C1 can be evaluated separately and thus serves as a strong fastText-based baseline for BLI tasks. Stage C2 leverages both C1-aligned cross-lingual word embeddings (CLWEs) and pretrained multilingual LMs such as mBERT/XLM/mT5 to further improve the BLI performance. Of course, C2 is compatible with other CLWE models: you could instead use, say, VecMap or RCSLS to derive CLWEs which can then replace C1-aligned CLWEs in C2!    

## Dependencies:

- PyTorch 1.7.0
- Transformers 4.4.2

## Get Data:
Our data are obtained from the [XLING repo](https://github.com/codogogo/xling-eval), and we include some simple codes to download its BLI dictionaries and preprocessed word embeddings.
```bash
sh get_data.sh
```


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

When running experiments on a different dataset, on different language pairs or having different BLI settings such as dictionary sizes, word embeddings (WEs) or pretrained LMs, doing hyper-parameter search in both Stage C1 and Stage C2 is necessary, whenever a dev set is available.
 
1. In C1, you might use valid_every = 10 to track the BLI performance when doing hyper-parameter search. If the BLI accuracy score (on your dev set) in each training epoch obviously drops from some point, then reduce num_games, lr, or gamma in C1/src/main.py; otherwise you may increase them.

2. In C1, when having a different train_size (other than 1k and 5k), we would recommend to tune num_sl, num_aug, and dico_max_rank (in C1/src/main.py) on your dev set. Besides, you may need to modify sup_batch_size and mini_batch_size according to your train_size.

3. lambda_ in C2/run_all.py is possibly sensitive to typologically remote language pairs, especially for those including lower-resource languages. We recommend to tune lambda_ in these cases.

## Known Issues:

It is reported that T5/mT5 produce "nan" outputs under mixed-precision or fp16 mode when using some Transformer versions ([ISSUE](https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139)). Our code also suffers from this issue. When running C2 with mT5, we recommend to switch off amp by commenting line 54 in ./C2/src/metric_learning.py.

## Acknowledgements:

Part of our code is adapted from the following GitHub repos: [XLING](https://github.com/codogogo/xling-eval), [RCSLS](https://github.com/facebookresearch/fastText/tree/main/alignment), [VecMap](https://github.com/artetxem/vecmap), [Mirror-BERT](https://github.com/cambridgeltl/mirror-bert) and [ECNMT](https://github.com/cambridgeltl/ECNMT). 

## If you find our paper and resources useful, please kindly cite our paper:

    @inproceedings{YL:BLI2022,
      author    = {Yaoyiran Li and Fangyu Liu and Nigel Collier and Anna Korhonen and Ivan Vulić},
      title     = {Improving Word Translation via Two-Stage Contrastive Learning},
      year      = {2022},
      booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
    }
