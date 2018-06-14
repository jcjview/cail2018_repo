import re
import json
import jieba

jieba.setLogLevel('WARN')

class normalizer:
    def __init__(self, stopword_filepath):
        '''
        :param stopword_filepath: 停用词表路径
        :return:
        '''
        self.raw_json_list = []
        self.cut_list = []

        self.data_out = None
        self.stopword_filepath = stopword_filepath
        self.pre_sub_pattern = [
            r'([\d一二三四五六七八九十零]+年|[\d一二三四五六七八九十零]+月|[\d一二三四五六七八九十零]+(日|号)|[\d一二三四五六七八九十零]+时|\d+分)+',

        ]
        self.pre_sub_replacer = [
            'DATETIME',

        ]
        self.post_sub_pattern = [
            r'^(.*?)某$',
            r'^(.*?)某(\d+|甲|乙|丙|丁)$',
            r'^(.+)某(.+)$',
        ]
        self.post_sub_replacer = [
            'PERSONNAME',
            '',
            '',
        ]
        # self.read_data()
        # self.replace_word('fact')
        # self.stopwordlist = None
        self.stopwordlist = self.read_data_to_list(self.stopword_filepath)
        # self.segmenter('fact', 2)

    def read_data(self):
        with open(self.input_filepath, 'r', encoding='utf8') as fin:
            line = fin.readline()
            while line:
                line = line.strip()
                if line != "":
                    self.raw_json_list.append(json.loads(line))
                line = fin.readline()

    def read_data_to_list(self, filepath):
        rslt_list = []

        with open(filepath, 'r', encoding='utf8') as fin:
            line = fin.readline()
            while line:
                line = line.strip()
                if line != "":
                    rslt_list.append(line)
                line = fin.readline()

        return rslt_list

    def replace_word(self, key):
        for one_json in self.raw_json_list:
            for i in range(len(self.pre_sub_pattern)):
                one_json[key] = re.sub(self.pre_sub_pattern[i], self.pre_sub_replacer[i], one_json[key])
        print("ok.")



    def seg_one_text(self, one_text, filterd_word_len):
        rslt_list = []
        text = one_text
        text=text.replace("\r","")
        text=text.replace("\n","")
        for i in range(len(self.pre_sub_pattern)):
            text = re.sub(self.pre_sub_pattern[i], self.pre_sub_replacer[i], text)
        tmp_cut_list = [word for word in jieba.lcut(text) if len(word) >= filterd_word_len]
        for word in tmp_cut_list:
            if word in self.stopwordlist:
                continue
            if (re.match(r'^\d+(\.)?\d+$', word) != None):
                number = float(word)
                if (number < 1000 and number >= 0):
                    rslt_list.append('m0')
                elif (number < 10000 and number >= 1000):
                    rslt_list.append('m' + str((int)(number / 1000)))
                elif (number < 100000 and number >= 10000):
                    rslt_list.append('mm' + str((int)(number / 10000)))
                elif (number < 1000000 and number >= 100000):
                    rslt_list.append('mmm' + str((int)(number / 100000)))
                elif (number < 10000000 and number >= 1000000):
                    rslt_list.append('mmmm' + str((int)(number / 1000000)))
                elif (number < 100000000 and number >= 10000000):
                    rslt_list.append('mmmmm' + str((int)(number / 10000000)))
                elif (number >= 100000000):
                    rslt_list.append('mmmmmm')
                continue
            elif (re.match(r'^[\d\.%a-zA-Z]+$', word) != None):
                continue
            elif (re.match(r'^(.*?)(县|市|乡|镇|州|村|区)$', word) != None):
                rslt_list.append('PLACEAERA')
                continue
            for i in range(len(self.post_sub_pattern)):
                word = re.sub(self.post_sub_pattern[i], self.post_sub_replacer[i], word)
                if word == '':
                    break
            if word != '':
                rslt_list.append(word)

        return rslt_list

    def seg_batch_text(self, text_list, filterd_word_len):
        rslt_batch_list = []

        for one_text in text_list:
            rslt_batch_list.append(self.seg_one_text(one_text, filterd_word_len))

        return rslt_batch_list