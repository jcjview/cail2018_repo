import json
import multiprocessing
import re
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from predictor import data
import bz2
from normalize import normalizer
dim = 500000
special_character_removal = re.compile(r'[@#$%^&*,.【】[]{}；‘，。、？!?“”‘’; \\/"\']', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

normalizer = normalizer('word.txt')

word_len=2
# cut = thulac.thulac(seg_only=True)
def seg(text):
    seg_list = jieba.cut(text.strip())
    seg_list = [word for word in seg_list if len(word) >= word_len]
    return " ".join(seg_list)


def text_to_wordlist(text):
    return " ".join(normalizer.seg_one_text(text,2))
    # text = special_character_removal.sub('', text)
    # text = replace_numbers.sub('NUMBERREPLACE', text)
    # return seg(text)


def multi_preprocess(comments=[]):
    pool_size = 6
    print("pool_size", pool_size)
    pool = multiprocessing.Pool(pool_size)
    pool_outputs = pool.map(text_to_wordlist, comments)
    pool.close()
    pool.join()
    print('successful')
    return pool_outputs


def cut_text(alltext):
    comments = multi_preprocess(comments=alltext)
    return comments


def save_seg(textlist, filepath):
    with open(filepath, 'w', encoding="utf-8") as thefile:
        for item in textlist:
            thefile.write("%s\n" % item)


def read_seg(filepath):
    textlist = []
    with open(filepath, encoding="utf-8") as thefile:
        for line in thefile:
            textlist.append(line.strip())


if __name__ == '__main__':

    TRAIN_HDF5='process.h5'
    print('reading...')
    alltext, accu_label, law_label, time_label,time_death,time_life = data.read_trainData('../good/cail2018_big.json.bz2')

    train_data = cut_text(alltext)
    #
    import pandas as pd

    save_seg(train_data, 'train_data_seg.txt')

    df = pd.DataFrame({'text': train_data, 'accu_label': accu_label, 'law_label': law_label, 'time_label': time_label,'time_death':time_death,'time_life':time_life})
    df.to_csv('process_big.csv',compression='bz2')
    print(len(df))
