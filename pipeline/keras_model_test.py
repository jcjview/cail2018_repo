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
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from bgru_model import *

def get_model(model_str='cnn_model1',embedding_matrix=None):
    m = eval(model_str)(embedding_matrix)
    return m




if (os.path.exists(TRAIN_HDF5)):
    print('load tokenizer,train')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    X_train = outh5file['train_token']
    y_accu = outh5file['train_label']
    law_label_y = outh5file['train_label']
    time_label_y = outh5file['time_label_y']
    nb_words = 0
    X_train = np.array(X_train, copy=True)
    y_accu = np.array(y_accu, copy=True)
    law_label_y = np.array(law_label_y, copy=True)
    time_label_y = np.array(time_label_y, copy=True)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
else:
    print('init train h5')
    df = pd.read_csv(input_file, compression='infer',encoding="utf-8")
    text = df['text'].values
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

    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(text))
    list_tokenized_text = tokenizer.texts_to_sequences(text)
    X_train = pad_sequences(list_tokenized_text, maxlen=MAX_TEXT_LENGTH)
    print('x shape', X_train.shape)
    nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
    print("nb_words", nb_words)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    outh5file.create_dataset('train_token', data=X_train)
    outh5file.create_dataset('train_label', data=y_accu)
    outh5file.create_dataset('law_label_y', data=law_label_y)
    outh5file.create_dataset('time_label_y', data=time_label_y)

timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

split1 = -17492
split2 = -32508
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

model_name=sys.argv[1]
print(model_name)

weight_name=sys.argv[2]
print(weight_name)

if "2" in model_name:
    y_train=y_train2
    y_val=y_val2
    y_test=y_test2

if "3" in model_name:
    y_train=y_train3
    y_val = y_val3
    y_test = y_test3



print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)
with open(model_name) as fr:
    model_json=fr.read()
model = keras.models.model_from_json(model_json) #get_model(model_name,embedding_matrix1)
early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=5, verbose=1)
bst_model_path = weight_name #model_name + '_bestweight_valid_%s.h5' % timeStr
# bst_model_path = 'cnn_weight1.h5'
csv_logger = keras.callbacks.CSVLogger('./log/' + model_name + '_log.csv', append=True, separator=';')
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True)
# hist = model.fit(x_train, y_train,
#                  validation_data=(x_val, y_val),
#                  epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
#                  verbose=1,
#                  callbacks=[F1ScoreCallback(), early_stopping, model_checkpoint, csv_logger]
#                  )
model.load_weights(bst_model_path)

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


tp = [1.0] * class_num
fp = [1.0] * class_num
fn = [1.0] * class_num
tn = [1.0] * class_num
acc = [0.0] * class_num
recall = [0.0] * class_num
f1 = [0.0] * class_num

for i in range(law_class_num):
    tn[i], fp[i], fn[i], tp[i] = confusion_matrix(y_val[:, i], predict1[:, i]).ravel()
    acc[i] = tp[i] / (fp[i] + tp[i])
    recall[i] = tp[i] / (fn[i] + tp[i])
    f1[i] = 2 * acc[i] * recall[i] / (acc[i] + recall[i])
out = pd.DataFrame({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'acc': acc, 'recall': recall, 'f1': f1})
out.to_csv('confusion_matrix2.csv', index=False)
accall=np.mean(acc)
recallall=np.mean(recall)
macro_f1=2*accall*recallall/(accall+recallall)