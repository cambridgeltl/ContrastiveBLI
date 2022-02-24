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

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()

    size_train = "5k" # or "1k"
    ROOT_EMB_SRC = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang1)
    ROOT_EMB_TRG = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang2)
    ROOT_TEST_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1, lang2, lang1, lang2)
    ROOT_TRAIN_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.train.freq.{}.{}-{}.tsv".format(lang1, lang2, size_train , lang1, lang2)
    SAVE_ROOT = "/media/data/SAVE" # save aligend WEs

    os.system('CUDA_VISIBLE_DEVICES=0  python ./src/main.py --l1 {} --l2 {} --self_learning --train_size {} --emb_src_dir {} --tgt_src_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {}'.format(lang1, lang2, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, ROOT_TRAIN_DICT, ROOT_TEST_DICT, SAVE_ROOT))