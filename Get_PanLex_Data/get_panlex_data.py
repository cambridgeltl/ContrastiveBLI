import fasttext.util
import fasttext
import io
import numpy as np


fasttext.util.download_model('et', if_exists='ignore')
fasttext.util.download_model('hu', if_exists='ignore')
fasttext.util.download_model('ka', if_exists='ignore')
fasttext.util.download_model('bg', if_exists='ignore')
fasttext.util.download_model('ca', if_exists='ignore')
fasttext.util.download_model('he', if_exists='ignore')


root = "/media/data/PanLex_WEs/"

lang_list = ["bg","ca","he","et","hu","ka"]

for lang in lang_list:
    print(lang)
    f_name = "fasttext.cc.{}.300.vocab_200K.vec".format(lang)
    ft = fasttext.load_model('cc.{}.300.bin'.format(lang))
    MAX_LEN = 200000 - 1
    word_list_org = ft.get_words(include_freq=True)[0][:MAX_LEN*2]
    word_list = []
    word_list_old = []
    word_set = set()
    for w in word_list_org:
        w_ = w.lower()
        #if w_ not in word_set:
        if (w_ not in word_set) and (len(w_.split())>0):
            word_set.add(w_)
            word_list.append(w_)
            word_list_old.append(w)
        if len(word_set) == MAX_LEN:
            break

    f = open(root+f_name,"w")
    f.write("{} 300\n".format(MAX_LEN))
    k = 0
    for i,w in enumerate(word_list):
        we = ft.get_word_vector(word_list_old[i])

        we = [str(num) for num in we]

        line = w + " " + " ".join(we) + "\n"
        f.write(line)
        k += 1
        if k % 20000 == 0:
            print(k)
    f.close()
