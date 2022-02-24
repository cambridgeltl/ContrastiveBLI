import os
import sys

lang_pairs = [('de', 'fi'),
 ('de', 'fr'),
 ('de', 'hr'),
 ('de', 'it'),
 ('de', 'ru'),
 ('de', 'tr'),
 ('en', 'de'),
 ('en', 'fi'),
 ('en', 'fr'),
 ('en', 'hr'),
 ('en', 'it'),
 ('en', 'ru'),
 ('en', 'tr'),
 ('fi', 'fr'),
 ('fi', 'hr'),
 ('fi', 'it'),
 ('fi', 'ru'),
 ('hr', 'fr'),
 ('hr', 'it'),
 ('hr', 'ru'),
 ('it', 'fr'),
 ('ru', 'fr'),
 ('ru', 'it'),
 ('tr', 'fi'),
 ('tr', 'fr'),
 ('tr', 'hr'),
 ('tr', 'it'),
 ('tr', 'ru')]


#Hyper-parameters to reproduce our results.
gpuid = '0,1'
train_size = '1k' #or '5k'
DIR_NEW = "./BLI/"
output_root = "/media/data/bert_models_final/"
random_seed = 33
model_dir = "bert-base-multilingual-uncased" # "google/mt5-small", "xlm-mlm-100-1280"
epoch = 5 # 6 for "google/mt5-small" 
train_batch_size = 100 # 50 for "xlm-mlm-100-1280"
learning_rate = 2e-5 # 6e-4 for "google/mt5-small" 
max_length = 6
checkpoint_step = 9999999
infoNCE_tau = 0.1
agg_mode = "cls"
num_neg = 28
num_iter = 1
template = 0
neg_max = 60000 
lambda_ = 0.2

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()


    ROOT_FT = "/media/data/SAVE{}/".format(train_size)
    l1_voc = ROOT_FT + "{}2{}_{}_voc.npy".format(lang1,lang2,lang1)
    l1_emb = ROOT_FT + "{}2{}_{}_emb.pt".format(lang1,lang2,lang1)
    l2_voc = ROOT_FT + "{}2{}_{}_voc.npy".format(lang1,lang2,lang2)
    l2_emb = ROOT_FT + "{}2{}_{}_emb.pt".format(lang1,lang2,lang2)
    DIR_TEST_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1,lang2,lang1,lang2)
    DIR_TRAIN_DICT = "media/data/xling-eval/bli_datasets/{}-{}/yacle.train.freq.{}.{}-{}.tsv".format(lang1,lang2,train_size,lang1,lang2)

    train_dir = DIR_NEW + "{}2{}_train.txt".format(lang1,lang2)

    output_dir = output_root + "mbert_{}2{}_{}".format(lang1,lang2,train_size)
    os.system("rm -f {}/*".format(output_dir))
    os.system("mkdir -p {}".format(output_dir))

    print("GEN NEG SAMPLES")
    sys.stdout.flush()
    os.system('CUDA_VISIBLE_DEVICES={} python gen_neg_samples.py --l1 {} --l2 {} --train_size {} --root {} --num_neg {} --neg_max {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {}'.format(gpuid,lang1, lang2, train_size, DIR_NEW, num_neg, neg_max, l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT))

    print("C2 Contrastive TRAINING")
    sys.stdout.flush() 
    os.system("CUDA_VISIBLE_DEVICES={} python3 train.py --model_dir {} --train_dir {} --output_dir {} --l1 {} --l2 {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --use_cuda --epoch {} --train_batch_size {} --learning_rate {} --max_length {} --checkpoint_step {} --parallel --amp --random_seed {} --infoNCE_tau {} --random_erase 0 --dropout_rate 0.1 --agg_mode {} --num_neg {} --template {} --random_seed {}".format(gpuid, model_dir, train_dir, output_dir, lang1, lang2, l1_voc, l1_emb, l2_voc, l2_emb, epoch, train_batch_size, learning_rate, max_length, checkpoint_step, random_seed, infoNCE_tau, agg_mode, num_neg, template, random_seed))

    print("EVALUATION")
    sys.stdout.flush() 
    os.system("CUDA_VISIBLE_DEVICES={} python evaluate_bli_procrustes.py --l1 {} --l2 {} --train_size {} --root {} --model_name {} --agg_mode {} --template {} --max_length {} --l1_voc {} --l1_emb {} --l2_voc {} --l2_emb {} --train_dict_dir {} --test_dict_dir {} --lambda_ {} --origin_model_name {}".format(gpuid, lang1, lang2, train_size, DIR_NEW, output_dir, agg_mode, template, max_length,l1_voc, l1_emb, l2_voc, l2_emb, DIR_TRAIN_DICT, DIR_TEST_DICT, lambda_, model_dir))

