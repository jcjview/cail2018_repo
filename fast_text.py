# _*_coding:utf-8 _*_
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
import json
from baseline.predictor import data
import thulac
import jieba
def read_trainData(path):
    fin = open(path, 'r', encoding='utf8')

    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        accu_label.append(data.getlabel(d, 'accu'))
        law_label.append(data.getlabel(d, 'law'))
        time_label.append(data.getlabel(d, 'time'))
        line = fin.readline()
    fin.close()

    return alltext, accu_label, law_label, time_label


def read_stop_words(stops_words_path):
    fin = open(stops_words_path, 'r', encoding='utf8')
    stop_words = []
    line = fin.readline()
    while line:
        stop_words.append(line.strip())
        line = fin.readline()
    fin.close()
    return stop_words


def cut_text(alltext, stop_words):
    count = 0
    cut = thulac.thulac(seg_only=True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 10 == 0:
            print(count)
            return train_text
        cut_list = cut.cut(text, text=True)
        #for word in cut_list:
        #    if word in stop_words:
        #        cut_list.remove(word)
        train_text.append(cut_list)

    return train_text


if __name__ == '__main__':
     print('reading stopwords')
     stop_words = read_stop_words("stopwords.txt")

     print('reading train data')
     alltext, accu_label, law_label, time_label = read_trainData('data_valid.json')

     print('cut text...')
     train_data = cut_text(alltext,stop_words)

     print('prepare train data...')
     f_acc = open("acc_train.txt", 'w')
     f_law = open("acc_law.txt", 'w')
     f_time = open("acc_time.txt", 'w')
     for i,text in enumerate(train_data):
         f_acc.write(text + " __label__" + str(accu_label[i]) + "\n")
         f_law.write(text + " __label__" + str(law_label[i]) + "\n")
         f_time.write(text + " __label__" + str(time_label[i]) + "\n")
     f_acc.close()
     f_law.close()
     f_time.close()

     print('train acc...')
     classifier = fasttext.supervised("acc_train.txt", "acc.model", label_prefix="__label__")
     print('train law...')
     classifier = fasttext.supervised("acc_law.txt", "law.model", label_prefix="__label__")
     print('train time...')
     classifier = fasttext.supervised("acc_time.txt", "time.model", label_prefix="__label__")


#load训练好的模型
#classifier = fasttext.load_model('fasttext.model.bin', label_prefix='__label__')
#测试模型
#result = classifier.test("news_fasttext_test.txt")
#print(result.precision)
#print(result.recall)