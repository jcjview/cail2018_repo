# input_file = "../input/process_10k.csv"
input_file = "../process.csv"
# SEP = "\t"
SEP = ","
w2vpath = '../baike.128.no_truncate.glove.txt'
embedding_matrix_path = './matrix_glove.npy'
kernel_name = "bilstm"
word_index_path = "worddict.pkl"
TRAIN_HDF5="train_hdf5.h5"
import h5py
import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score

MAX_TEXT_LENGTH = 300
MAX_FEATURES = 100000
embedding_dims = 128
dr = 0.2
dropout_p = 0.1
fit_batch_size = 256
fit_epoch = 22

class_num = 202

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
            avgf1=(f1 + f2) / 2
            print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] =avgf1




def get_model(embedding_matrix, nb_words):
    input_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=True)
    # seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr))
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.CuDNNGRU(256))

    x = seq_embedding_layer(keras.layers.SpatialDropout1D(0.2)(words_embedding_layer(input_tensor)))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dr)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(dr)(x)
    output_layer = keras.layers.Dense(class_num, activation="softmax")(x)
    model = keras.models.Model(input_tensor, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy",
                                                                                     # f1_score_metrics
                                                                                     ])
    model.summary()
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


df = pd.read_csv(input_file, encoding="utf-8")
text = df['text'].values
label = df['accu_label'].values
from sklearn.preprocessing import LabelEncoder
# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(label)
# convert integers to dummy variables (one hot encoding)
y = keras.utils.to_categorical(encoded_Y,num_classes=class_num)
print('y shape',y.shape)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(text))
list_tokenized_text = tokenizer.texts_to_sequences(text)
X_train = pad_sequences(list_tokenized_text, maxlen=MAX_TEXT_LENGTH)
print('x shape',X_train.shape)
nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
print("nb_words", nb_words)
embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
outh5file = h5py.File(TRAIN_HDF5, 'w')
outh5file.create_dataset('train_token', data=X_train)
outh5file.create_dataset('train_label', data=y)

# outh5file = h5py.File(TRAIN_HDF5, 'r')
# X_train = outh5file['train_token']
# test = outh5file['test_token']
# y = outh5file['train_label']
# embedding_matrix1 = np.load(embedding_matrix_path)
# nb_words=MAX_FEATURES


seed = 20180426
cv_folds = 4
from sklearn.model_selection import KFold

skf = KFold(n_splits=cv_folds, random_state=seed, shuffle=True)
pred_oob = np.zeros(shape=y.shape)
# print(pred_oob.shape)
count = 0
import time


timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
for ind_tr, ind_te in skf.split(X_train, y):
    x_train = X_train[ind_tr]
    x_val = X_train[ind_te]
    y_train = y[ind_tr]
    y_val = y[ind_te]
    print('x_train shape',x_train.shape)
    print('x_val shape',x_val.shape)
    print('y_train shape',y_train.shape)
    print('y_val shape',y_val.shape)
    model = get_model(embedding_matrix1, nb_words)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    bst_model_path = kernel_name + '_weight_%d_%s.h5' % (count, timeStr)
    csv_logger = keras.callbacks.CSVLogger('./log/' + bst_model_path + '_log.csv', append=True, separator=';')
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
                                       save_best_only=True, verbose=1, save_weights_only=False)
    hist = model.fit(x_train, y_train,
                     validation_data=(x_val, y_val),
                     epochs=fit_epoch, batch_size=fit_batch_size, shuffle=False,
                     verbose=1,
                     callbacks=[F1ScoreCallback(),early_stopping, model_checkpoint]
                     )
    predict=model.predict(x_val,batch_size=1024)
    pred_oob[ind_te] = predict
    # predict[predict > 0.5] = 1
    # predict[predict < 0.5] = 0
    # y_label_true = np.argmax(y_val)
    # print('y_label_true',y_label_true.shape)
    # predict_label_true = np.argmax(predict)
    # print('predict_label_true', predict.shape)
    # macro_f1 = f1_score(y_label_true, predict_label_true, average="macro")
    # micro_f1 = f1_score(y_label_true, predict_label_true, average="micro")
    # print("macro_f1", macro_f1)
    # print("micro_f1", micro_f1)
    # print(macro_f1/2+micro_f1/2)

    count+=1
    # break
pred_oob[pred_oob>0.5]=1
pred_oob[pred_oob<0.5]=0

# y_label_true = np.argmax(y)
# predict_label_true =  np.argmax(pred_oob)
macro_f1=f1_score(y,pred_oob,average="macro")
micro_f1=f1_score(y,pred_oob,average="micro")
print("macro_f1",macro_f1)
print("micro_f1",micro_f1)
print(macro_f1/2+micro_f1/2)