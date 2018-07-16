"""
# Create a directory and mount Google Drive using that directory.
!mkdir -p drive
!google-drive-ocamlfuse drive

print('Files in Drive:')
!ls -lh drive/colab
!mv drive/colab/input/baike.txt.bz2 ./datalab/
!bzip2 -d ./datalab/baike.txt.bz2
!ls ./datalab/
"""
input_file = "../process.csv"
SEP = ","
w2vpath = '../baike.128.truncate.glove.txt'
embedding_matrix_path = './matrix_glove.npy'
kernel_name = "bgru_cnn_task3"
word_index_path = "worddict.pkl"
TRAIN_HDF5 = "train_hdf5.h5"
import h5py
import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score

MAX_TEXT_LENGTH = 300
nb_words=MAX_FEATURES = 100000
embedding_dims = 128
dr = 0.2
dropout_p = 0.1
fit_batch_size = 256
fit_epoch = 30

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
            y_predict, predict2, predict3 = self.model.predict(self.validation_data[0],
                                           batch_size=self.predict_batch_size)
            y_predict[y_predict >= 0.5] = 1
            y_predict[y_predict < 0.5] = 0
            f1 = f1_score(self.validation_data[1], y_predict, average='macro')
            # print("macro f1_score %.4f " % f1)
            f2 = f1_score(self.validation_data[1], y_predict, average='micro')
            # print("micro f1_score %.4f " % f2)
            avgf1=(f1 + f2) / 2
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] =avgf1



def get_model(embedding_matrix, nb_words):
    input_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=False)
    # seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr,return_sequences=True))
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.CuDNNGRU(256,return_sequences=True))

    x = seq_embedding_layer(keras.layers.SpatialDropout1D(0.2)(words_embedding_layer(input_tensor)))
    x = keras.layers.Conv1D(128, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.concatenate([avg_pool, max_pool])
    output_layer = keras.layers.Dense(class_num, activation="softmax")(x)
    output_law = keras.layers.Dense(law_class_num, activation="softmax")(x)
    output_time = keras.layers.Dense(time_class_num, activation="softmax")(x)

    model = keras.models.Model(input_tensor, [output_layer, output_law, output_time])
    loss1 = 'categorical_crossentropy'
    mse = 'mae'
    model.compile(loss=[loss1, loss1, loss1], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open("bgru_cnn_model.json", "w") as json_file:
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
y = keras.utils.to_categorical(label,num_classes=class_num)
print('y shape',y.shape)

law_label=df['law_label'].values
law_label_y = keras.utils.to_categorical(law_label,num_classes=law_class_num)
print('y shape',law_label_y.shape)

time_label=df['time_label'].values
time_label_y = keras.utils.to_categorical(time_label,num_classes=time_class_num)
print('y shape',time_label_y.shape)

print('y type',type(time_label[0]))

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer(num_words=MAX_FEATURES)
# tokenizer.fit_on_texts(list(text))
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
list_tokenized_text = tokenizer.texts_to_sequences(text)
X_train = pad_sequences(list_tokenized_text, maxlen=MAX_TEXT_LENGTH)
print('x shape',X_train.shape)
nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
print("nb_words", nb_words)
embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)


import time

timeStr = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

x_train = X_train[:155000]
x_test = X_train[-33000:]
x_val=X_train[-50000:-33000]
y_train = y[:155000]
y_test = y[-33000:]
y_train2 = law_label_y[:155000]
y_test2 = law_label_y[-33000:]
y_train3 = time_label_y[:155000]
y_test3 = time_label_y[-33000:]

y_val = y[-50000:-33000]
y_val2 = law_label_y[-50000:-33000]
y_val3 = time_label_y[-50000:-33000]

print('x_train shape', x_train.shape)
print('x_val shape', x_test.shape)
print('y_train shape', y_train.shape)
print('y_val shape', y_test.shape)
model = get_model(embedding_matrix1, nb_words)
early_stopping = EarlyStopping(monitor='avg_f1_score_val', mode='max',patience=5, verbose=1)
bst_model_path = kernel_name + '_weight_valid_%s.h5' % timeStr
csv_logger = keras.callbacks.CSVLogger('./log/' + bst_model_path + '_log.csv', append=True, separator=';')
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val',mode='max',
                                   save_best_only=True, verbose=1, save_weights_only=True)
hist = model.fit(x_train, [y_train,y_train2,y_train3],
                 validation_data=(x_test, [y_test, y_test2, y_test3]),
                 epochs=fit_epoch, batch_size=fit_batch_size, shuffle=True,
                 verbose=1,
                 callbacks=[F1ScoreCallback(),early_stopping, model_checkpoint ]
                 )
model.load_weights(bst_model_path)
predict,predict2,predict3 = model.predict(x_val, batch_size=1024)
predict[predict > 0.5] = 1
predict[predict < 0.5] = 0
macro_f1 = f1_score(y_val, predict, average="macro")
micro_f1 = f1_score(y_val, predict, average="micro")
print("macro_f1", macro_f1)
print("micro_f1", micro_f1)
print(macro_f1 / 2 + micro_f1 / 2)

predict2[predict2 > 0.5] = 1
predict2[predict2 < 0.5] = 0
macro_f1 = f1_score(y_val2, predict2, average="macro")
micro_f1 = f1_score(y_val2, predict2, average="micro")
print("2 macro_f1", macro_f1)
print("2 micro_f1", micro_f1)
print('2',macro_f1 / 2 + micro_f1 / 2)
