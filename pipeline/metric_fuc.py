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
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(F1ScoreCallback, self).__init__()
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
            predict = self.model.predict(self.validation_data[0],
                                           batch_size=self.predict_batch_size)
            y_predict=predict2half(predict)
            f1 = f1_score(self.validation_data[1], y_predict, average='macro')
            print("macro f1_score %.4f " % f1)
            f2 = f1_score(self.validation_data[1], y_predict, average='micro')
            print("micro f1_score %.4f " % f2)
            avgf1 = (f1 + f2) / 2
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] = avgf1


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
