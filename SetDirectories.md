If you also use XLING data and save the dataset in the same directories as we did, you could directly run the code without modifying anything; just pay a little attention to (c) and (d) below. When using customised data, you would also need to modify the input/output directories specified in (a) and (b) below. 

(a) In [./C1/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/C1/run_all.py), set the directories for (1) pretrained static word embeddings for source and target languages respectively; (2) test dictionary; and (3) training dictionary.

(b) In [./C2/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/C2/run_all.py), set the directories for (1) saving hard negative pairs extracted from C1; (2) saving fine-tuned multilingual LMs; (3) pretrained multilingual LMs; (3) some hyper parameter values for C2; (4) pretrained static word embeddings for source and target languages; (5) test dictionary; and (6) training dictionary.

(c) To run supervised (5k seed pairs) and semi-supervised (1k seed pairs) BLI tasks, please set the size of the training dictionary to '5k' or '1k' respectively in both [./C1/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/C1/run_all.py) and [./C2/run_all.py](https://github.com/cambridgeltl/ContrastiveBLI/C2/run_all.py).

(d) To save C1-aligned embeddings for C2's use, please set ```args.save_aligned_we = True``` in [./C1/src/main.py](https://github.com/cambridgeltl/ContrastiveBLI/C1/src/main.py) .
