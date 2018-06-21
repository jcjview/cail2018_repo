import mmap

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
import os


def predict2half(predictions):
    y_pred = np.zeros(predictions.shape)
    y_pred[predictions > 0.5] = 1
    return y_pred


def predict2tag(predictions):
    y_pred = np.array(predictions, copy=True)
    for index, x in enumerate(y_pred):
        x[x > 0.5] = 1
        if x.max() < 1:
            x[x == x.max()] = 1
    y_pred[y_pred < 1] = 0
    return y_pred


class F1ScoreCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False,data_test=None):
        super(F1ScoreCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        self.data_test=data_test;
    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('avg_f1_score_val' in self.params['metrics']):
            self.params['metrics'].append('avg_f1_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['avg_f1_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['avg_f1_score_val'] = float('-inf')
        if (self.validation_data):
            predict = self.model.predict(self.validation_data[0],
                                         batch_size=self.predict_batch_size)
            y_predict = predict2half(predict)
            f1 = f1_score(self.validation_data[1], y_predict, average='macro')
            print("macro f1_score %.4f " % f1)
            f2 = f1_score(self.validation_data[1], y_predict, average='micro')
            print("micro f1_score %.4f " % f2)
            avgf1 = (f1 + f2) / 2
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] = avgf1
        if(self.data_test):
            predict = self.model.predict(self.data_test[0],
                                         batch_size=self.predict_batch_size)
            y_predict = predict2tag(predict)
            f1 = f1_score(self.data_test[1], y_predict, average='macro')
            print("test macro f1_score %.4f " % f1)
            f2 = f1_score(self.data_test[1], y_predict, average='micro')
            print("test micro f1_score %.4f " % f2)
            avgf1 = (f1 + f2) / 2
            print("test avg_f1_score %.4f " % (avgf1))
            logs['avgf1_test'] = avgf1

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(word_index, Emed_path, Embed_npy):
    if (os.path.exists(Embed_npy)):
        return np.load(Embed_npy)
    print('Indexing word vectors')
    embeddings_index = {}
    file_line = get_num_lines(Emed_path)
    print('lines ', file_line)
    with open(Emed_path, encoding='utf-8') as f:
        for line in tqdm(f, total=file_line):
            values = line.split()
            if (len(values) < embedding_dims):
                print(values)
                continue
            word = ' '.join(values[:-embedding_dims])
            coefs = np.asarray(values[-embedding_dims:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = MAX_FEATURES  # min(MAX_FEATURES, len(word_index))
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, embedding_dims))

    # embedding_matrix = np.zeros((nb_words, embedding_dims))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count += 1
    np.save(Embed_npy, embedding_matrix)
    print('Null word embeddings: %d' % (nb_words - count))
    print('not Null word embeddings: %d' % count)
    print('embedding_matrix shape', embedding_matrix.shape)
    # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def judger(label_true, y_predict):
    result = 0
    l1, l2, l3 = label_true
    p1, p2, p3 = y_predict
    p2[p2 > 0.5] = 1
    p2[p2 < 0.5] = 0
    p3[p3 > 0.5] = 1
    p3[p3 < 0.5] = 0
    # p1 = np.reshape(p1, (-1,))
    # p2 = np.reshape(p2, (-1,))
    # p3 = np.reshape(p3, (-1,))
    for i in range(len(y_predict)):
        yp = round(p1[i][0])
        dp = p2[i][0]
        lp = p3[i][0]

        yt = l1[i][0]
        dt = l2[i][0]
        lt = l3[i][0]

        sc = 0
        if dt == 1:
            if dp ==1:
                sc = 1
        elif lt == 1:
            if lp==1:
                sc = 1
        else:
            v1 =yt
            v2 = yp
            v = abs(np.log(v1 + 1) - np.log(v2 + 1))
            if v <= 0.2:
                sc = 1
            elif v <= 0.4:
                sc = 0.8
            elif v <= 0.6:
                sc = 0.6
            elif v <= 0.8:
                sc = 0.4
            elif v <= 1.0:
                sc = 0.2
            else:
                sc = 0
        sc = sc * 1.0
        result += sc
    return result / len(y_predict)


class ImprisonCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(ImprisonCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('avg_f1_score_val' in self.params['metrics']):
            self.params['metrics'].append('avg_f1_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['avg_f1_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['avg_f1_score_val'] = float('-inf')
        if (self.validation_data):
            y_predict = self.model.predict(self.validation_data[0],
                                           batch_size=self.predict_batch_size)
            label = self.validation_data[1], self.validation_data[2], self.validation_data[3]
            logs['avg_f1_score_val'] = judger(label, y_predict)
