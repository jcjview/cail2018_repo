# input_file = "../input/process_10k.csv"
from keras import Input
from keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, LSTM, BatchNormalization, merge, \
    Dense, PReLU, Dropout
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# no pre embeded 0.7668
input_file = "../process_big.csv.bz2"
# SEP = "\t"
SEP = ","
w2vpath = '../vectors.txt'
# w2vpath = '../baike.128.no_truncate.glove.txt'
embedding_matrix_path = './matrix_glove.npy'
kernel_name = "cnnbig1"
word_index_path = "worddict.pkl"
TRAIN_HDF5 = "train_hdf5_200k_200.h5"

import h5py
import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, mean_absolute_error

MAX_TEXT_LENGTH = 500
MAX_FEATURES = 200000
embedding_dims = 200
dr = 0.2
dropout_p = 0.1
fit_batch_size = 64
fit_epoch = 40

class_num = 202
law_class_num = 183
time_class_num = 9


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
            y_predict = self.model.predict(self.validation_data[0],
                                           batch_size=self.predict_batch_size)
            y_predict[y_predict >= 0.5] = 1
            y_predict[y_predict < 0.5] = 0
            f1 = f1_score(self.validation_data[1], y_predict, average='macro')
            print("macro f1_score %.4f " % f1)
            f2 = f1_score(self.validation_data[1], y_predict, average='micro')
            print("micro f1_score %.4f " % f2)
            avgf1 = (f1 + f2) / 2
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] = avgf1


def get_model(embedding_matrix, nb_words):
    input_tensor = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                # weights=[embedding_matrix],
                                input_length=MAX_TEXT_LENGTH,
                                trainable=False)
    emb1 = embedding_layer(input_tensor)
    emb1 = SpatialDropout1D(0.2)(emb1)
    # 1D convolutions that can iterate over the word vectors
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = keras.layers.Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = keras.layers.Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = keras.layers.Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Run inputs through embedding

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = keras.layers.GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(emb1)
    glob2a = keras.layers.GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(emb1)
    glob3a = keras.layers.GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(emb1)
    glob4a = keras.layers.GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(emb1)
    glob5a = keras.layers.GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(emb1)
    glob6a = keras.layers.GlobalAveragePooling1D()(conv6a)

    mergea = keras.layers.concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])

    # The MLP that determines the outcome
    x = keras.layers.Dropout(0.2)(mergea)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(300, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization()(x)
    output_layer = keras.layers.Dense(class_num, activation="sigmoid")(x)
    # output_law = keras.layers.Dense(law_class_num, activation="sigmoid")(x)
    # output_time = keras.layers.Dense(time_class_num, activation="softmax")(x)

    model = keras.models.Model(input_tensor, output_layer)
    loss1 = 'binary_crossentropy'
    loss2 = 'categorical_crossentropy'
    model.compile(loss=[loss1], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open(kernel_name+ "_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model


from tqdm import tqdm
import mmap
import os


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(word_index, Emed_path, Embed_npy):
    # if (os.path.exists(Embed_npy)):
    #     return np.load(Embed_npy)
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


df = pd.read_csv(input_file,compression='infer', encoding="utf-8")
text = df['text'].values
label = df['accu_label'].values
from sklearn.preprocessing import MultiLabelBinarizer

lb_y = MultiLabelBinarizer()
label = [set([int(i) for i in str(row).split(";")]) for row in label]
y = lb_y.fit_transform(label)
print('y shape', y.shape)

law_label = df['law_label'].values
law_label = [set([int(i) for i in str(row).split(";")]) for row in law_label]
lb_law = MultiLabelBinarizer()
law_label_y = lb_law.fit_transform(law_label)
print('law y shape', law_label_y.shape)

time_label = df['time_label'].values
time_label_y = keras.utils.to_categorical(time_label, num_classes=time_class_num)
print('y shape', time_label_y.shape)
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

if (os.path.exists(TRAIN_HDF5)):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    X_train = outh5file['train_token']
    y = outh5file['train_label']
    # law_label_y = outh5file['train_label']
    # time_label_y = outh5file['time_label_y']
    nb_words = 0
    X_train = np.array(X_train, copy=True)
    y = np.array(y, copy=True)
    # law_label_y = np.array(law_label_y, copy=True)
    # time_label_y = np.array(time_label_y, copy=True)
    embedding_matrix1 = np.load(embedding_matrix_path)
else:

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
    outh5file.create_dataset('train_label', data=y)
    outh5file.create_dataset('law_label_y', data=law_label_y)
    outh5file.create_dataset('time_label_y', data=time_label_y)
# import pickle


import time

timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
from sklearn.utils import class_weight

idx = np.random.permutation(len(y))
X_train=X_train[idx]
y=y[idx]
split1=int(len(y)*0.8)
split2 = int(len(y)*0.9)
print("12",split1,split2)

x_train = X_train[:split1]
y_train = y[:split1]
y_train3 = time_label_y[:split1]

x_val = X_train[split1:split2]
y_val = y[split1:split2]
y_val3 = time_label_y[split1:split2]

x_test = X_train[split2:]
y_test = y[split2:]
y_tese3 = time_label_y[split2:]

# # idx = np.random.permutation(len(y))
# # X_train=X_train[idx]
# # y=y[idx]
# # law_label_y=law_label_y[idx]
# # time_label_y=time_label_y[idx]
# split1 = -17492
# split2 = -32508
# split = split1 + split2
# x_train = X_train[:split]
# y_train = y[:split]
# y_train2 = law_label_y[:split]
# y_train3 = time_label_y[:split]
#
# x_val = X_train[split:split2]
# y_val = y[split:split2]
# y_val2 = law_label_y[split:split2]
# y_val3 = time_label_y[split:split2]
#
# x_test = X_train[split2:]
# y_test = y[split2:]
# y_test2 = law_label_y[split2:]
# y_tese3 = time_label_y[split2:]

print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_val.shape)
print('y_test shape', y_test.shape)
model = get_model(embedding_matrix1, nb_words)
early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max', patience=5, verbose=1)
# bst_model_path = kernel_name + '_weight_valid_%s.h5' % timeStr
bst_model_path = 'cnn_weight1.h5'
csv_logger = keras.callbacks.CSVLogger('./log/' + bst_model_path + '_log.csv', append=True, separator=';')
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True)
hist = model.fit(x_train, y_train,
                 validation_data=(x_val, y_val),
                 epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
                 verbose=1,
                 callbacks=[F1ScoreCallback(), early_stopping, model_checkpoint, csv_logger]
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


def predict2tag(predictions):
    y_pred = np.array(predictions, copy=True)
    for index, x in enumerate(y_pred):
        x[x > 0.5] = 1
        if x.max() < 1:
            x[x == x.max()] = 1
    y_pred[y_pred < 1] = 0
    return y_pred


y_pred = predict2tag(predict)
f1 = f1_score(y_test, y_pred, average='macro')
print("macro f1_score %.4f " % f1)
f2 = f1_score(y_test, y_pred, average='micro')
print("micro f1_score %.4f " % f2)
avgf1 = (f1 + f2) / 2
print("avg_f1_score %.4f " % (avgf1))
