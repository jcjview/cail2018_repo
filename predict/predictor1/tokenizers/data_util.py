import json
import random

import numpy as np
import os


def init():
    f = open('../data/law.txt', 'r', encoding='utf8')
    law = {}
    lawname = {}
    line = f.readline()
    while line:
        lawname[len(law)] = line.strip()
        law[line.strip()] = len(law)
        line = f.readline()
    f.close()

    f = open('../data/accu.txt', 'r', encoding='utf8')
    accu = {}
    accuname = {}
    line = f.readline()
    while line:
        accuname[len(accu)] = line.strip()
        accu[line.strip()] = len(accu)
        line = f.readline()
    f.close()

    return law, accu, lawname, accuname


law, accu, lawname, accuname = init()


def getClassNum(kind):
    global law
    global accu

    if kind == 'law':
        return len(law)
    if kind == 'accu':
        return len(accu)


def getName(index, kind):
    global lawname
    global accuname
    if kind == 'law':
        return lawname[index]

    if kind == 'accu':
        return accuname[index]


def gettime(time):
    # 将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return -2
    elif time['life_imprisonment']:
        return -1
    else:
        return v
    '''
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8
    '''


def getlabel(d, kind):
    global law
    global accu

    # print(d)
    if kind == 'law':
        return d['meta']['relevant_articles']
    if kind == 'accu':
        accu = []
        accus = d['meta']['accusation']
        for t in accus:
            t = t.replace("[", "").replace("]", "")
            accu.append(t)
        return accu

    if kind == 'time':
        return gettime(d['meta']['term_of_imprisonment'])

    # return label


def read_stopwrods():
    stops_words_path = "stopwords.txt"
    fin = open(stops_words_path, 'r', encoding='utf8')
    stop_words = []
    line = fin.readline()
    while line:
        stop_words.append(line.strip())
        line = fin.readline()
    fin.close()
    return stop_words


def words_counter(train_data, stop_word):
    words_count = {}

    for cut_words in train_data:
        words_list = cut_words.split(' ')
        for word in words_list:
            if word in stop_word:
                continue

            if word in words_count:
                words_count[word] = words_count[word] + 1
            else:
                temp = 1
                words_count[word] = temp

    return sorted(words_count.items(), key=lambda d: d[1], reverse=True)


def read_trainData(path):
    fin = open(path, 'r', encoding='utf8')

    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        # print(line)
        d = json.loads(line)
        accu_label.append(getlabel(d, 'accu'))
        alltext.append(d['fact'])
        law_label.append(getlabel(d, 'law'))
        time_label.append(getlabel(d, 'time'))
        line = fin.readline()
    fin.close()
    time_label = time_class(time_label)
    return alltext, accu_label, law_label, time_label


def read_trainData_by_fre(path):
    fin = open(path, 'r', encoding='utf8')

    alltext = []
    accu_label = []
    law_label = []
    time_label = []

    alltext_single = []
    accu_label_single = []
    law_label_single = []
    time_label_single = []

    line = fin.readline()
    while line:
        # print(line)
        d = json.loads(line)
        accu_tmp = getlabel(d, 'accu')
        if len(accu_tmp) == 1:
            accu_label_single.append(''.join(getlabel(d, 'accu')))
            alltext_single.append(d['fact'])
            law_label_single.append(getlabel(d, 'law'))
            time_label_single.append(getlabel(d, 'time'))
        else:
            accu_label.append(getlabel(d, 'accu'))
            alltext.append(d['fact'])
            law_label.append(getlabel(d, 'law'))
            time_label.append(getlabel(d, 'time'))
        line = fin.readline()
    fin.close()

    return alltext, accu_label, law_label, time_label,alltext_single,accu_label_single,law_label_single,time_label_single


def read_trainData_by_dir(dir):
    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    for file_name in os.listdir(dir):
        fin = open(os.path.join(dir, file_name), "r", encoding='utf-8')
        line = fin.readline()
        while line:
            # print(line)
            d = json.loads(line)
            alltext.append(d['fact'])
            accu_label.append(getlabel(d, 'accu'))
            law_label.append(getlabel(d, 'law'))
            time_label.append(getlabel(d, 'time'))
            line = fin.readline()
        fin.close()
    time_label = time_class(time_label)
    return alltext, accu_label, law_label, time_label


def time_class(time_label):
    time_label_p = []
    mar = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for time in time_label:
        if time in mar:
            time_label_p.append(time)
        elif 11 <= time <= 15:
            time_label_p.append(12)
        elif 16 <= time <= 21:
            time_label_p.append(18)
        elif 22 <= time <= 27:
            time_label_p.append(24)
        elif 28 <= time <= 33:
            time_label_p.append(30)
        elif 34 <= time <= 39:
            time_label_p.append(36)
        elif 40 <= time <= 45:
            time_label_p.append(42)
        elif 46 <= time <= 51:
            time_label_p.append(48)
        elif 52 <= time <= 57:
            time_label_p.append(54)
        elif 58 <= time <= 63:
            time_label_p.append(60)
        elif 64 <= time <= 69:
            time_label_p.append(66)
        elif 70 <= time <= 75:
            time_label_p.append(72)
        elif 76 <= time <= 81:
            time_label_p.append(78)
        elif 82 <= time <= 87:
            time_label_p.append(84)
        elif 88 <= time <= 93:
            time_label_p.append(90)
        elif 94 <= time <= 99:
            time_label_p.append(96)
        elif 100 <= time <= 105:
            time_label_p.append(102)
        elif 106 <= time <= 111:
            time_label_p.append(108)
        elif 112 <= time <= 117:
            time_label_p.append(114)
        elif 118 <= time <= 123:
            time_label_p.append(120)
        elif 124 <= time <= 129:
            time_label_p.append(126)
        elif 130 <= time <= 135:
            time_label_p.append(132)
        elif 136 <= time <= 141:
            time_label_p.append(138)
        elif 142 <= time <= 147:
            time_label_p.append(144)
        elif 148 <= time <= 153:
            time_label_p.append(150)
        elif 154 <= time <= 159:
            time_label_p.append(156)
        elif 160 <= time <= 165:
            time_label_p.append(162)
        elif 166 <= time <= 171:
            time_label_p.append(168)
        elif 172 <= time <= 177:
            time_label_p.append(174)
        elif 178 <= time <= 183:
            time_label_p.append(180)
        elif 184 <= time <= 189:
            time_label_p.append(186)
        elif 190 <= time <= 195:
            time_label_p.append(192)
        elif 196 <= time <= 201:
            time_label_p.append(198)
        elif 202 <= time <= 207:
            time_label_p.append(204)
        elif 208 <= time <= 213:
            time_label_p.append(210)
        elif 214 <= time <= 219:
            time_label_p.append(216)
        elif 220 <= time <= 225:
            time_label_p.append(222)
        elif 226 <= time <= 231:
            time_label_p.append(228)
        elif 232 <= time <= 237:
            time_label_p.append(234)
        elif 238 <= time <= 243:
            time_label_p.append(240)
        elif 244 <= time <= 249:
            time_label_p.append(246)
        elif 250 <= time <= 255:
            time_label_p.append(252)
        elif 256 <= time <= 261:
            time_label_p.append(258)
        elif 262 <= time <= 267:
            time_label_p.append(264)
        elif 268 <= time <= 273:
            time_label_p.append(270)
        elif 274 <= time <= 279:
            time_label_p.append(276)
        elif 280 <= time <= 285:
            time_label_p.append(282)
        elif 286 <= time <= 291:
            time_label_p.append(288)
        elif 292 <= time <= 297:
            time_label_p.append(294)
        elif 298 <= time:
            time_label_p.append(300)
    return time_label_p


def transform_multilabel(labels, vocab_label2index):
    """
    convert data as indexes using word2index dicts.
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    label_size = len(vocab_label2index)
    Y = []
    for label_list in labels:
        label_list = [vocab_label2index[str(label)] for label in label_list]
        y = transform_multilabel_as_multihot(label_list, label_size)
        if 1 not in y:
            print('########### error data ###########')
        Y.append(y)
    return Y


def transform_multilabel_as_multihot(label_list, label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1
    return result


def big_data_split(path):
    fin = open(path, "r", encoding='utf-8')
    fout_train = open('E:/02-study/cail2018/cail2018_big/big_train.json', "w", encoding='utf-8')
    fout_test = open('E:/02-study/cail2018/cail2018_big/big_test.json', "w", encoding='utf-8')

    line = fin.readline()

    ori_content = []
    while line:
        # print(line)
        # line = '{"fact": "瑞安市人民检察院指控：\r\n2014年1月1日下午，'
        line = line.replace("\\r", '').replace("\\n", '').strip()
        ori_content.append(line)
        line = fin.readline()
    fin.close()

    random.shuffle(ori_content)

    n = 1
    for content in ori_content:
        if n < 50000:
            fout_test.write(content + '\n')
        else:
            fout_train.write(content + '\n')
        n = n + 1

    fout_train.close()
    fout_test.close()


def mkSubFile(lines, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w', encoding='utf-8')
    try:
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()


def splitByLineCount(filename, count):
    fin = open(filename, 'r', encoding='utf-8')
    try:
        buf = []
        sub = 1
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, filename, sub)
                buf = []
        if len(buf) != 0:
            sub = mkSubFile(buf, filename, sub)
    finally:
        fin.close()


def test_data_process(test_path, small_dir, out_path):
    # read test data
    fin = open(test_path, 'r', encoding='utf8')
    test_alltext = []

    line = fin.readline()
    while line:
        # print(line)
        d = json.loads(line)
        test_alltext.append(d['fact'])
        line = fin.readline()
    fin.close()

    # read small
    small_data = []
    for file_name in os.listdir(small_dir):
        fin = open(os.path.join(small_dir, file_name), "r", encoding='utf-8')
        line = fin.readline()
        while line:
            # print(line)
            d = json.loads(line)
            small_data.append(d['fact'])
            line = fin.readline()
    fin.close()

    # check
    fout = open(out_path, 'w', encoding='utf8')
    n = 1
    for test in small_data:
        if n % 20000 == 0:
            print(n)
        if test in test_alltext:
            print(test)
        else:
            fout.write(test)
        n = n + 1
    fout.close()


def data_anal(path):
    # read test data
    fin = open(path, 'r', encoding='utf8')
    fout = open('anal_law_time.txt', 'w', encoding='utf8')
    time_calc = {}
    time_label = []
    line = fin.readline()
    while line:
        # print(line)
        d = json.loads(line)
        time = getlabel(d, 'time')
        time_label.append(time)
        line = fin.readline()
    fin.close()
    temp = time_class(time_label)

    key_zw = {}
    for key in temp:
        key_zw[key] = get_median(temp[key])
    print(key_zw)


def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


if __name__ == '__main__':
    # big_data_split('E:/02-study/cail2018/cail2018_big/big_train.json')
    # splitByLineCount('E:/02-study/cail2018/cail2018_big/big_train.json', 170000)
    # test_data_process('E:/02-study/cail2018/cail2018_big/cail2018_big.json', 'E:/02-study/cail2018/small', 'E:/02-study/cail2018/cail2018_big/big_split/big_test-pre.json')
    data_anal('E:/02-study/cail2018/cail2018_big/cail2018_big_0612/cail2018_big/cail2018_big.json')
