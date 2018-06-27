import json
import multiprocessing
import re
import jieba
import time
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
# import thulac
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from predictor import data
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
    # return " ".join(normalizer.seg_one_text(text, 2))
    text = special_character_removal.sub('', text)
    text = replace_numbers.sub('NUMBERREPLACE', text)
    return seg(text)


def multi_preprocess(comments=[]):
    pool_size = 8 #multiprocessing.cpu_count() + 1
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


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=True
    )
    tfidf.fit(train_data)

    X = tfidf.fit_transform(train_data)
    word_dict = {}
    name = tfidf.get_feature_names()
    with open('name.txt', 'w') as fw:
        for i, s in enumerate(name):
            s = s.replace("", "_")
            word_dict[i] = s
            fw.write(s)
            fw.write('\n')

    raw_text = []

    for line in X.A:
        s = ""
        for i in line:
            s += " " + word_dict[i]
        raw_text.append(s)
    return raw_text





def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


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

    print('reading...')
    alltext, accu_label, law_label, time_label,time_death,time_life  = data.read_trainData('../data/data_train.json')
    alltext1, accu_label1, law_label1, time_label1,time_death1,time_life1  = data.read_trainData('../data/data_valid.json')
    alltext2, accu_label2, law_label2, time_label2,time_death2,time_life2  = data.read_trainData('../data/data_test.json')
    alltext3, accu_label3, law_label3, time_label3, time_death3, time_life3 = data.read_trainData('../data/all.json')
    alltext4, accu_label4, law_label4, time_label4, time_death4, time_life4 = data.read_trainData('../data/create_data.json')
    print('train',len(alltext))
    print('valid',len(alltext1))
    print('test',len(alltext2))

    alltext += alltext3+alltext4
    accu_label += accu_label3+accu_label4
    law_label += law_label3+law_label4
    time_label += time_label3+time_label4
    time_death += time_death3+time_death4
    time_life += time_life3+time_life4

    alltext+=alltext1+alltext2
    accu_label+=accu_label1+accu_label2
    law_label+=law_label1+law_label2
    time_label+=time_label1+time_label2

    time_death+=time_death1+time_death2
    time_life+=time_life1+time_life2

    train_data = cut_text(alltext)
    import pandas as pd
    df = pd.DataFrame({'text': train_data, 'accu_label': accu_label, 'law_label': law_label, 'time_label': time_label,'time_death':time_death,'time_life':time_life})

    timeStr = time.strftime("%m-%d_%H", time.localtime())

    df.to_csv('process_create_%s.csv.bz2'%timeStr,compression='bz2')
    print(len(df))