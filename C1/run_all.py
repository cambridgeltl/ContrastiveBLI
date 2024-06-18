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
 ('tr', 'ru'),
 ('bg', 'ca'),
 ('ca','hu'),
 ('hu','bg'),
 ('ca','bg'),
 ('hu','ca'),
 ('bg','hu')]

XLING = set(["en","de","fr","it","ru","tr","hr","fi"])
PanLex = set(["bg","ca","hu","eu","et","he"])

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()

    size_train = "5k" # "5k" (supervised setup), "1k" (semi-supervised setup), or "0k" (unsupervised setup).  

    if lang1 in XLING:    
        DIR_EMB_SRC = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang1)
        DIR_EMB_TGT = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang2)
        DIR_TEST_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1, lang2, lang1, lang2)
    else:
        DIR_EMB_SRC = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang1)
        DIR_EMB_TGT = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang2)
        DIR_TEST_DICT = "/media/data/panlex-bli/lexicons/all/{}-{}/{}-{}.test.2000.cc.trans".format(lang1, lang2, lang1, lang2)        
    SAVE_DIR = "/media/data/SAVE" # save aligend WEs

    if size_train == "0k":
        # In unsupervised setup, need aligned CLWEs from another unsupervised BLI approach
        if lang1 in XLING:
            aux_emb_src_dir = "/media/data/SAVE0kVecMap/{}-{}.OUTPUT_SRC.tsv".format(lang1, lang2)
            aux_emb_tgt_dir =  "/media/data/SAVE0kVecMap/{}-{}.OUTPUT_TRG.tsv".format(lang1, lang2)
        else:
            aux_emb_src_dir = "/media/data/SAVE0kVecMapP/{}-{}.OUTPUT_SRC.tsv".format(lang1, lang2)
            aux_emb_tgt_dir =  "/media/data/SAVE0kVecMapP/{}-{}.OUTPUT_TRG.tsv".format(lang1, lang2)
        DIR_TRAIN_DICT = "./"
    else:
        aux_emb_src_dir = None # None if not unsupervised setup
        aux_emb_tgt_dir = None 
        DIR_TRAIN_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.train.freq.{}.{}-{}.tsv".format(lang1, lang2, size_train , lang1, lang2)

    os.system('CUDA_VISIBLE_DEVICES=0  python ./src/main.py --l1 {} --l2 {} --self_learning --save_aligned_we --train_size {} --emb_src_dir {} --emb_tgt_dir {} --aux_emb_src_dir {} --aux_emb_tgt_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {}'.format(lang1, lang2, size_train, DIR_EMB_SRC, DIR_EMB_TGT, aux_emb_src_dir, aux_emb_tgt_dir, DIR_TRAIN_DICT, DIR_TEST_DICT, SAVE_DIR))

