import os
import pickle
import time

import h5py
import keras
import numpy as np
import pandas as pd
from config import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from metric_fuc import predict2tag, get_embedding_matrix, F1ScoreCallback
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from bgru_model import *
from cnn_model import *

def get_model(model_str='cnn_model1', embedding_matrix=None):
    m = eval(model_str)(embedding_matrix)
    return m


def generate_batch_data_random(x, y, batch_size, tokenizer):
    """逐步提取batch数据到显存，降低对显存的占用"""

    # print("x.shape",x.shape)
    idx = np.arange(len(y))
    print("batch.shape", idx.shape)
    np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in range(len(y) // batch_size + 1)]
    while (True):
        for i in batches:
            list_tokenized_text = tokenizer.texts_to_sequences(x[i])
            arr = pad_sequences(list_tokenized_text, maxlen=MAX_TEXT_LENGTH)
            # print('x[i]',x[i].shape)
            # print('y[i]',y[i].shape)
            yield arr, y[i]


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

df = pd.read_csv(input_file, compression='infer', encoding="utf-8")
text = df['text'].values

if (os.path.exists(TRAIN_HDF5)):
    print('load tokenizer,train')
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    X_train = text
    y_accu = outh5file['train_label']
    law_label_y = outh5file['train_label']
    time_label_y = outh5file['time_label_y']
    nb_words = 0
    # X_train = np.array(X_train, copy=True)
    y_accu = np.array(y_accu, copy=True)
    law_label_y = np.array(law_label_y, copy=True)
    time_label_y = np.array(time_label_y, copy=True)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
else:
    print('init train h5')

    label = df['accu_label'].values

    lb_y = MultiLabelBinarizer()
    label = [set([int(i) for i in str(row).split(";")]) for row in label]
    y_accu = lb_y.fit_transform(label)
    print('accu y shape', y_accu.shape)

    law_label = df['law_label'].values
    law_label = [set([int(i) for i in str(row).split(";")]) for row in law_label]
    lb_law = MultiLabelBinarizer()
    law_label_y = lb_law.fit_transform(law_label)
    print('law y shape', law_label_y.shape)

    time_label = df['time_label'].values
    time_label_y = keras.utils.to_categorical(time_label, num_classes=time_class_num)
    print('time_label y shape', time_label_y.shape)

    # tokenizer = Tokenizer(num_words=MAX_FEATURES)
    # tokenizer.fit_on_texts(list(text))
    # list_tokenized_text = tokenizer.texts_to_sequences(text)
    X_train = text
    # print('x shape', X_train.shape)
    # nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
    # print("nb_words", nb_words)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    # outh5file.create_dataset('train_token', data=X_train)
    outh5file.create_dataset('train_label', data=y_accu)
    outh5file.create_dataset('law_label_y', data=law_label_y)
    outh5file.create_dataset('time_label_y', data=time_label_y)

timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

split1 = -int(len(y_accu) * 0.2)
split2 = -int(len(y_accu) * 0.1)
split = split1 + split2
x_train = X_train[:split]
y_train = y_accu[:split]
y_train2 = law_label_y[:split]
y_train3 = time_label_y[:split]

x_val = X_train[split:split2]
y_val = y_accu[split:split2]
y_val2 = law_label_y[split:split2]
y_val3 = time_label_y[split:split2]

x_test = X_train[split2:]
y_test = y_accu[split2:]
y_test2 = law_label_y[split2:]
y_test3 = time_label_y[split2:]

import sys

model_name = sys.argv[1]
print(model_name)

if "2" in model_name:
    y_train = y_train2
    y_val = y_val2
    y_test = y_test2

if "3" in model_name:
    y_train = y_train3
    y_val = y_val3
    y_test = y_test3

list_val_text = tokenizer.texts_to_sequences(x_val)
x_val = pad_sequences(list_val_text, maxlen=MAX_TEXT_LENGTH)

print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)
model = get_model(model_name, embedding_matrix1)
early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=5, verbose=1)
bst_model_path = model_name + '_bestweight_valid_%s.h5' % timeStr
# bst_model_path = 'cnn_weight1.h5'
csv_logger = keras.callbacks.CSVLogger('./log/' + model_name + '_log.csv', append=True, separator=';')
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True)

model.fit_generator(generator=generate_batch_data_random(x_train, y_train, fit_batch_size, tokenizer),
                    steps_per_epoch=len(y_train) / fit_batch_size,
                    epochs=fit_epoch,
                    shuffle=False,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[F1ScoreCallback(),early_stopping, model_checkpoint, csv_logger])

# hist = model.fit(x_train, y_train,
#                  validation_data=(x_val, y_val),
#                  epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
#                  verbose=1,
#                  callbacks=[F1ScoreCallback(), early_stopping, model_checkpoint, csv_logger]
#                  )
model.load_weights(bst_model_path)

list_val_text = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(list_val_text, maxlen=MAX_TEXT_LENGTH)

predict = model.predict(x_test, batch_size=1024)

predict1 = np.array(predict, copy=True)
predict1[predict1 > 0.5] = 1
predict1[predict1 < 0.5] = 0
macro_f1 = f1_score(y_test, predict1, average="macro")
micro_f1 = f1_score(y_test, predict1, average="micro")
print("macro_f1", macro_f1)
print("micro_f1", micro_f1)
print(macro_f1 / 2 + micro_f1 / 2)

y_pred = predict2tag(predict)
f1 = f1_score(y_test, y_pred, average='macro')
print("macro f1_score %.4f " % f1)
f2 = f1_score(y_test, y_pred, average='micro')
print("micro f1_score %.4f " % f2)
avgf1 = (f1 + f2) / 2
print("avg_f1_score %.4f " % (avgf1))
