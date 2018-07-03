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
from capsule_model import *
from attention import *
from keras_tqdm import TQDMCallback
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def get_model(model_str='cnn_model1', embedding_matrix=None):
    m = eval(model_str)(embedding_matrix)
    return m


import sys

model_name = sys.argv[1]
print(model_name)

if (os.path.exists(TRAIN_HDF5)):
    print('load tokenizer,train')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    All_train = outh5file['train_token']
    y_accu = outh5file['train_label']
    law_label_y = outh5file['law_label_y']
    time_label_y = outh5file['time_label_y']
    time_life_y = outh5file['time_life_y']
    time_death_y = outh5file['time_death_y']
    nb_words = 0
    All_train = np.array(All_train, copy=True)
    y_accu = np.array(y_accu, copy=True)
    law_label_y = np.array(law_label_y, copy=True)
    print('law y shape', law_label_y.shape)
    time_label_y = np.array(time_label_y, copy=True)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
else:
    print('init train h5')
    df = pd.read_csv(input_file, compression='infer', encoding="utf-8")
    df=df.fillna(0)
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

    time_label_y = df['time_label'].values
    time_death_y = df['time_death'].values
    time_life_y = df['time_life'].values
    # time_label_y = keras.utils.to_categorical(time_label, num_classes=time_class_num)
    # print('time_label y shape', time_label_y.shape)

    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(text))
    list_tokenized_text = tokenizer.texts_to_sequences(text)
    All_train = pad_sequences(list_tokenized_text, maxlen=MAX_TEXT_LENGTH)
    print('x shape', All_train.shape)
    nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
    print("nb_words", nb_words)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    outh5file.create_dataset('train_token', data=All_train)
    outh5file.create_dataset('train_label', data=y_accu)
    outh5file.create_dataset('law_label_y', data=law_label_y)
    outh5file.create_dataset('time_label_y', data=time_label_y)
    outh5file.create_dataset('time_death_y', data=time_death_y)
    outh5file.create_dataset('time_life_y', data=time_life_y)

timeStr = '2018-06-30_14:36:04'#time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

split1 = -17131
split2 = -32508
split = split1 + split2
X_train = All_train[:split]
y_train = y_accu[:split]
y_train2 = law_label_y[:split]
y_train3 = time_label_y[:split]
y_traind3 = time_death_y[:split]
y_trainl3 = time_life_y[:split]

x_val = All_train[split:split2]
y_val = y_accu[split:split2]
y_val2 = law_label_y[split:split2]
y_val3 = time_label_y[split:split2]
y_vald3 = time_death_y[split:split2]
y_vall3 = time_label_y[split:split2]

x_test = All_train[split2:]
y_test = y_accu[split2:]
y_test2 = law_label_y[split2:]
y_test3 = time_label_y[split2:]
y_testd3 = time_death_y[split2:]
y_testl3 = time_life_y[split2:]

if "1" in model_name:
    print(1)


# if "2" in model_name:
#     print("2")
#     y_train = y_train2
#     y_val = y_val2
#     y_test = y_test2
#
# if "3" in model_name:
#     y_train = [y_train3,y_traind3,y_trainl3]
#     y_val = [y_val3,y_testd3,y_testl3]
#     y_test = [y_test3,y_testd3,y_testl3]

# x_train=np.concatenate((x_train,x_test))
# y_train=np.concatenate((y_train,y_test))
print('X_train shape', X_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)
#df = pd.read_csv(input_file, compression='infer', encoding="utf-8")
#label = df['accu_label'].values
#label = [set([int(i) for i in str(row).split(";")]) for row in label]
#lb_y = MultiLabelBinarizer()
#lb_y.fit_transform(label)

class_list=[156,173,197,184,104,105,123,40,26,171,32,96,193,114,21,180,186,42,31,143,195,29,136,44,90,37,28,131,49,77,174,199]

seed = 20180626
cv_folds = 10
from sklearn.model_selection import KFold

skf = KFold(n_splits=cv_folds, random_state=seed, shuffle=True)
pred_oob = np.zeros(shape=y_train.shape)
pred_test=np.zeros(shape=y_test.shape)

count = 0
for ind_tr, ind_te in skf.split(X_train, y_train):

    x_tr = X_train[ind_tr]
    x_v = X_train[ind_te]
    y_tr = y_train[ind_tr]
    y_v = y_train[ind_te]

    x_v1=np.concatenate((x_v,x_val))
    y_v1=np.concatenate((y_v,y_val))

#    train_sample_weight = np.zeros((y_tr.shape[0],), np.float32)
#array_label = lb_y.inverse_transform(y_train)
#for i in range(0, y_tr.shape[0]):
#        train_sample_weight[i] = 1
#        for l in array_label[i]:
#            if l in class_list:
#                train_sample_weight[i] = 2
#                break

    model = get_model(model_name, embedding_matrix1)
    early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=10, verbose=1)
    bst_model_path = model_name + '_weight_%d_%s.h5' % (count, timeStr)
    csv_logger = keras.callbacks.CSVLogger('./log/' + model_name + '_log.csv', append=True, separator=';')
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                       save_best_only=True, verbose=1, save_weights_only=True)
    print("model_"+bst_model_path)
    hist = model.fit(x_tr, y_tr,
                     validation_data=(x_v1, y_v1),
                     epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
                     verbose=2,
                     callbacks=[F1ScoreCallback(data_test=(x_test,y_test)), early_stopping, model_checkpoint, csv_logger],
                     # sample_weight=train_sample_weight,
                     )
    count += 1
    model.load_weights(bst_model_path)
    model.save("model_"+bst_model_path)
    predict = model.predict(x_v, batch_size=1024,verbose=2)
    pred_oob[ind_te] = predict
    p_test = model.predict(x_test, batch_size=1024)
    pred_test += p_test
#print cv score
y_pred = predict2tag(pred_oob)
f1 = f1_score(y_train, y_pred, average='macro')
print("cv macro f1_score %.4f " % f1)
f2 = f1_score(y_train, y_pred, average='micro')
print("cv micro f1_score %.4f " % f2)
cvavgf1 = (f1 + f2) / 2
print("cv avg_f1_score %.4f " % (cvavgf1))

#print test score
pred_test /= cv_folds
y_pred = predict2tag(pred_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("test macro f1_score %.4f " % f1)
f2 = f1_score(y_test, y_pred, average='micro')
print("test micro f1_score %.4f " % f2)
avgf1 = (f1 + f2) / 2
print("test avg_f1_score %.4f " % (avgf1))

print(timeStr,cvavgf1,avgf1)
