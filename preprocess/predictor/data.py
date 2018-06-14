import bz2
import json


def init():
    f = open('law.txt', 'r', encoding='utf8')
    law = {}
    lawname = {}
    line = f.readline()
    while line:
        lawname[len(law)] = line.strip()
        law[line.strip()] = len(law)
        line = f.readline()
    f.close()

    f = open('accu.txt', 'r', encoding='utf8')
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

    # if time['death_penalty']:
    #     return -1
    # if time['life_imprisonment']:
    #     return -2
    return v

def getDeath(time):
    if time['death_penalty']:
        return 1
    return 0

def getLife(time):
    if time['life_imprisonment']:
        return 1
    return 0

def getlabel(d, kind):
    global law
    global accu
    # print(d)
    label = []
    if kind == 'law':
        for t in d['meta']['relevant_articles']:
            label.append(str(law[str(t)]))
        return ";".join(label)
    if kind == 'accu':
        for t in d['meta']['accusation']:
            t = t.replace("[", "")
            t = t.replace("]", "")
            label.append(str(accu[str(t)]))
        return ";".join(label)
    if kind == 'time':
        return gettime(d['meta']['term_of_imprisonment'])
    if kind == 'death':
        return getDeath(d['meta']['term_of_imprisonment'])
    if kind == 'life':
        return getLife(d['meta']['term_of_imprisonment'])

def read_trainData(path):
    with bz2.BZ2File(path, "r") as fin:
        alltext = []
        accu_label = []
        law_label = []
        time_label = []
        time_death=[]
        time_life=[]
        line = fin.readline()
        while line:
            d = json.loads(line)
            alltext.append(d['fact'])
            accu_label.append(getlabel(d, 'accu'))
            law_label.append(getlabel(d, 'law'))
            time_label.append(getlabel(d, 'time'))
            time_death.append(getlabel(d, 'death'))
            time_life.append(getlabel(d, 'life'))
            line = fin.readline()

    return alltext, accu_label, law_label, time_label,time_death,time_life