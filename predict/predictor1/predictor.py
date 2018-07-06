from sklearn.externals import joblib
from predictor.tokenizers.norm_tokenizer import NormalTokenizer
import time
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
        elapsed_time = time.time() - start_time
        print('init done',elapsed_time)
        # self.cut = thulac.thulac(seg_only=True)
        # self.stopwords_list = []
        print("warning result.append(y_int + 1) disabled!!!")
    def predict_law(self, y,vec):
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

    def predict_accu(self, y,vec):
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

    def predict(self, content):
        fact_temp = [self.nor_cut.tokenize(c) for c in content]
        # words_list = fact_temp.split(' ')
        vec = self.tfidf.transform(fact_temp)
        ret = []
        ans = {}
        p1 = self.accu_ld.predict(vec)
        p2 = self.law_ld.predict(vec)
        p3 = self.time.predict(vec)
        for i in range(len(content)):
            v=vec[i]
            ret.append(
                {'accusation': self.predict_accu(p1[i],v),
                'articles': self.predict_law(p2[i],v),
                'imprisonment': self.predict_time(p3[i]),
                        }
            )
        # p1 = self.predict_accu(vec)
        # p2 = self.predict_law(vec)
        # p3 = self.predict_time(vec)

        # ans['articles'] = [1]
        # ans['imprisonment'] = 1

        # print(ans)
        return ret


