kernel_name = "bgru_cnn"
import keras
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, BatchNormalization, Dense, \
    Dropout, concatenate

from config import *

dr = 0.2
dropout_p = 0.1

def bgru_cnn_model1(embedding_matrix):
    input_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=False)
    # seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr,return_sequences=True))
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.CuDNNGRU(256, return_sequences=True))

    x = seq_embedding_layer(keras.layers.SpatialDropout1D(0.2)(words_embedding_layer(input_tensor)))
    x = keras.layers.Conv1D(128, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.concatenate([avg_pool, max_pool])
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    output_layer = keras.layers.Dense(class_num, activation="sigmoid")(x)
    model = keras.models.Model(input_tensor, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open(kernel_name+"_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model

def bgru_cnn_model2(embedding_matrix):
    input_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=False)
    # seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr,return_sequences=True))
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.CuDNNGRU(256, return_sequences=True))
    x = seq_embedding_layer(keras.layers.SpatialDropout1D(0.2)(words_embedding_layer(input_tensor)))
    x = keras.layers.Conv1D(128, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.concatenate([avg_pool, max_pool])
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(law_class_num, activation="sigmoid")(x)

    model = Model(input_tensor, output_layer)
    loss2 = 'binary_crossentropy'
    model.compile(loss=[loss2], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open(kernel_name+"_model2.json", "w") as json_file:
        json_file.write(model_json)
    return model

def bgru_cnn_model3(embedding_matrix):
    input_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=False)
    # seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr,return_sequences=True))
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.CuDNNGRU(256, return_sequences=True))
    x = seq_embedding_layer(keras.layers.SpatialDropout1D(0.2)(words_embedding_layer(input_tensor)))
    x = keras.layers.Conv1D(128, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.concatenate([avg_pool, max_pool])
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(1, activation="linear")(x)
    output_layer2 = Dense(1, activation="sigmoid",name='death')(x)
    output_layer3 = Dense(1, activation="sigmoid",name='life')(x)

    model = Model(input_tensor, [output_layer,output_layer2,output_layer3])
    loss1 = 'binary_crossentropy'
    loss2 = 'mse'
    model.compile(loss=[loss2,loss1,loss1], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open(kernel_name + "_model3.json", "w") as json_file:
        json_file.write(model_json)
    return model
