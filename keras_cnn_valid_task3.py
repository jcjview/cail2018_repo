import os
import pickle
import time
import numpy as np
import h5py
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers import GlobalMaxPool1D, Embedding, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, concatenate, \
    Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer,LabelBinarizer

from metric_fuc import predict2tag, get_embedding_matrix, F1ScoreCallback
from config import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_name = 'valid_task3'


# def imp2class(time_label):
#     arr=np.zeros(shape=(time_label.shape[0],),dtype=np.int)
#     for i,time in enumerate(time_label):
#         if time<=0:
#             arr[i]=0
#         elif time<=1:
#             arr[i]=1
#         elif time<=2:
#             arr[i]=2
#         elif time<=3:
#             arr[i]=3
#         elif time<=4:
#             arr[i]=4
#         elif time<=6:
#             arr[i]=6
#         elif time<8:
#             arr[i]=8
#         elif time<10:
#             arr[i]=10
#         elif time<13:
#             arr[i] = 12
#         elif time <= 16:
#             arr[i] = 15
#         elif time <= 20:
#             arr[i] = 18
#         elif time < 25:
#             arr[i] = 23
#         elif time <= 31:
#             arr[i] = 28
#         elif time <= 38:
#             arr[i] = 35
#         elif time <= 47:
#             arr[i] = 43
#         elif time <= 58:
#             arr[i] = 53
#         elif time <= 72:
#             arr[i] = 66
#         elif time <= 88:
#             arr[i] = 80
#         elif time <= 108:
#             arr[i] = 98
#         elif time <= 133:
#             arr[i] = 121
#         elif time <= 163:
#             arr[i] = 148
#         elif time <= 200:
#             arr[i] = 182
#         elif time <= 245:
#             arr[i] = 223
#         elif time <= 300:
#             arr[i] = 273
#     return arr



def get_model(embedding_matrix,num_imp):
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    x = Embedding(MAX_TEXT_LENGTH, embedding_dims, embedding_matrix=embedding_matrix,
                  trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
    p1 = GlobalMaxPool1D()(x)
    p2 = GlobalAveragePooling1D()(x)

    conc = concatenate([p1, p2])
    fc1 = Dense(256, activation='relu')(conc)
    fc1 = Dropout(0.1)(fc1)

    fc2 = Dense(128, activation='relu')(fc1)
    fc2 = Dropout(0.1)(fc2)

    out = Dense(num_imp, activation='softmax')(fc2)

    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if (os.path.exists(TRAIN_HDF5)):
    print('load tokenizer,train')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    X_train = outh5file['train_token']
    y_accu = outh5file['train_label']
    law_label_y = outh5file['law_label_y']
    time_label_y = outh5file['time_label_y']
    time_life_y = outh5file['time_life_y']
    time_death_y = outh5file['time_death_y']
    nb_words = 0
    X_train = np.array(X_train, copy=True)
    y_accu = np.array(y_accu, copy=True)
    law_label_y = np.array(law_label_y, copy=True)
    print('law y shape', law_label_y.shape)
    time_label_y = np.array(time_label_y, copy=True)
    embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
else:
    print('init train h5')
    df = pd.read_csv(input_file, compression='infer', encoding="utf-8")
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
    outh5file.create_dataset('time_death_y', data=time_death_y)
    outh5file.create_dataset('time_life_y', data=time_life_y)

timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


y_test_class = imp2class(time_label_y)
le = LabelBinarizer()
time_label_y1=le.fit_transform(y_test_class)

split1 = -17131
split2 = -32508
split = split1 + split2
x_train = X_train[:split]
y_train = y_accu[:split]
y_train2 = law_label_y[:split]
y_train3 = time_label_y1[:split]
y_traind3 = time_death_y[:split]
y_trainl3 = time_life_y[:split]

x_val = X_train[split:split2]
y_val = y_accu[split:split2]
y_val2 = law_label_y[split:split2]
y_val3 = time_label_y1[split:split2]
y_vald3 = time_death_y[split:split2]
y_vall3 = time_label_y1[split:split2]

x_test = X_train[split2:]
y_test = y_accu[split2:]
y_test2 = law_label_y[split2:]
y_test3 = time_label_y1[split2:]
y_testd3 = time_death_y[split2:]
y_testl3 = time_life_y[split2:]

model = get_model(embedding_matrix1,time_label_y1.shape[1])

# if "2" in model_name:
#     print("2")
#     y_train = y_train2
#     y_val = y_val2
#     y_test = y_test2
#
# if "3" in model_name:
print("3")

y_train = y_train3
y_val = y_val3
y_test=y_test3


# x_train=np.concatenate((x_train,x_test))
# y_train=np.concatenate((y_train,y_test))
print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)

early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=5, verbose=1)
bst_model_path = model_name + '_bestweight_valid_%s.h5' % timeStr
# bst_model_path = 'cnn_weight1.h5'
csv_logger = CSVLogger('./log/' + model_name + '_log.csv', append=True, separator=';')
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True)
print("fit_batch_size {}", fit_batch_size)
hist = model.fit(x_train, y_train,
                 validation_data=(x_val, y_val),
                 epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
                 verbose=2,
                 callbacks=[ImprisonCallback(date), early_stopping, model_checkpoint, csv_logger],
                 # sample_weight=train_sample_weight,
                 )
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

bst_model_path = model_name + "test_%.5f.h5" % avgf1
model.save_weights(bst_model_path)
