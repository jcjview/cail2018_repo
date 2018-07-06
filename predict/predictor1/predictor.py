import pickle

from sklearn.externals import joblib
from predictor.tokenizers.norm_tokenizer import NormalTokenizer
from predictor.tokenizers.dnn_model import seg,load_keras_model,MAX_TEXT_LENGTH,transform,returnOneHot
from keras.preprocessing.sequence import pad_sequences

import time

from sklearn.preprocessing import MultiLabelBinarizer


class Predictor(object):
    def __init__(self):
        print('start init')
        start_time = time.time()
        self.tfidf = joblib.load('predictor/model/tfidf.model')
        self.law = joblib.load('predictor/model/law-lp.model')
        self.law_ld = joblib.load('predictor/model/law-ld-k=5.model')
        self.accu = joblib.load('predictor/model/accu-lp.model')
        self.accu_ld = joblib.load('predictor/model/accu-ld-k=5.model')
        self.time = joblib.load('predictor/model/time.model')
        self.batch_size = 1024
        self.nor_cut = NormalTokenizer()
        self.init_dnn()
        elapsed_time = time.time() - start_time
        print('init done',elapsed_time)
        # self.cut = thulac.thulac(seg_only=True)
        # self.stopwords_list = []
        print("warning predict_law_svm result.append(y_int + 1) disabled!!!")

    def init_dnn(self):
        with open('predictor/model/tokenizerCreate.pickle', 'rb') as handle:
            self.tokenizer1 = pickle.load(handle)
        self.cnn_model1 = load_keras_model('predictor/model/cnn_model1.json')
        self.cnn_model1.load_weights('predictor/model/cnn_model1.h5')

        self.bgru_model1 = load_keras_model('predictor/model/bgru_model1.json')
        self.bgru_model1.load_weights('predictor/model/bgru_model1.h5')

        self.imdb_model1 = load_keras_model('predictor/model/imdb_model1.json')
        self.imdb_model1.load_weights('predictor/model/imdb_model1.h5')

        self.cnn_model2 = load_keras_model('predictor/model/cnn_model2.json')
        self.cnn_model2.load_weights('predictor/model/cnn_model2.h5')

        self.bgru_model2 = load_keras_model('predictor/model/bgru_model2.json')
        self.bgru_model2.load_weights('predictor/model/bgru_model2.h5')

        self.imdb_model2 = load_keras_model('predictor/model/imdb_model2.json')
        self.imdb_model2.load_weights('predictor/model/imdb_model2.h5')
        with open('lb1.pickle', 'rb') as handle:
            self.lb1 = pickle.load(handle)
        with open('lb2.pickle', 'rb') as handle:
            self.lb2 = pickle.load(handle)

    def predict_law_svm(self, y,vec):
        result = []
        y_str = str(y)
        if y_str != '':
            for res in y_str.split('\n'):
                index1 = res.find(',')
                index2 = res.find(')')
                y_int = int(res[index1 + 1:index2].strip())
                # result.append(y_int + 1)
                result.append(y_int)

        if len(result) == 0:
            y = self.law.predict(vec)
            y_str = str(y)
            if y_str != '':
                for res in y_str.split('\n'):
                    index1 = res.find(',')
                    index2 = res.find(')')
                    y_int = int(res[index1 + 1:index2].strip())
                    result.append(y_int)
                    # result.append(y_int + 1)

        return result

    def predict_accu_svm(self, y,vec):
        result = []
        y_str = str(y)
        if y_str != '':
            for res in y_str.split('\n'):
                index1 = res.find(',')
                index2 = res.find(')')
                y_int = int(res[index1 + 1:index2].strip())
                result.append(y_int)
                # result.append(y_int + 1)
        if len(result) == 0:
            y = self.accu.predict(vec)
            y_str = str(y)
            if y_str != '':
                for res in y_str.split('\n'):
                    index1 = res.find(',')
                    index2 = res.find(')')
                    y_int = int(res[index1 + 1:index2].strip())
                    result.append(y_int)
                    # result.append(y_int + 1)
        return result

    def predict_time(self, y):
        # 返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def predict_svm(self, content):
        fact_temp = [self.nor_cut.tokenize(c) for c in content]
        vec = self.tfidf.transform(fact_temp)
        p1 = self.accu_ld.predict(vec)
        p2 = self.law_ld.predict(vec)
        # p3 = self.time.predict(vec)
        svm_p1=[self.predict_accu_svm(p1[i],vec[i]) for i in range(len(content))]
        svm_p2=[self.predict_law_svm(p2[i],vec[i]) for i in range(len(content))]

        svm_p1 = returnOneHot(svm_p1,self.lb1)
        svm_p2 = returnOneHot(svm_p2,self.lb2)
        # svm_p3=[self.predict_time(p3[i]) for i in range(len(content))]

        return svm_p1,svm_p2

    def predict_dnn(self,content):
        fact = [seg(c) for c in content]
        list_tokenized = self.tokenizer1.texts_to_sequences(fact)
        vec = pad_sequences(list_tokenized, maxlen=MAX_TEXT_LENGTH)
        cnn_p1 = self.cnn_model1.predict(vec, batch_size=1024, verbose=1)
        bgru_p1 = self.bgru_model1.predict(vec, batch_size=1024, verbose=1)
        imdb_p1=self.imdb_model1.predict(vec, batch_size=1024, verbose=1)

        cnn_p2 = self.cnn_model2.predict(vec, batch_size=1024, verbose=1)
        bgru_p2 = self.bgru_model2.predict(vec, batch_size=1024, verbose=1)
        imdb_p2 = self.imdb_model2.predict(vec, batch_size=1024, verbose=1)

        return cnn_p1,bgru_p1,imdb_p1,cnn_p2,bgru_p2,imdb_p2

    def predict_law(self, y):
        # predict_label = argmax(y, axis=1)
        # predict_label = predict_label + 1
        return [transform(c) for c in y]

    def predict_accu(self, y):
        # predict_label = argmax(y, axis=1)
        # predict_label = predict_label + 1
        return [transform(c) for c in y]

    def predict(self, content):
        cnn_p1, bgru_p1, imdb_p1, cnn_p2, bgru_p2, imdb_p2=self.predict_dnn(content)
        svm_p1, svm_p2, svm_p3=self.predict_svm(content)

        p1=(svm_p1*1+cnn_p1*0.8+bgru_p1+imdb_p1)/(3+0.8)

        p2 =(svm_p2*1+cnn_p2*0.8+bgru_p2+imdb_p2)/(3+0.8)
        ret = []
        p1 = self.predict_accu(p1)
        p2 = self.predict_law(p2)
        for i in range(len(content)):
            ret.append({'accusation': p1[i],
                        'articles': p2[i],
                        'imprisonment': 0})
        return ret