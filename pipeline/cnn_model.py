kernel_name = "cnn"
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, BatchNormalization, Dense, \
    Dropout, concatenate

from config import *

dr = 0.2
dropout_p = 0.1

def cnn_model1(embedding_matrix):
    input_tensor = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                # weights=[embedding_matrix],
                                input_length=MAX_TEXT_LENGTH,
                                trainable=False)
    emb1 = embedding_layer(input_tensor)
    emb1 = SpatialDropout1D(0.2)(emb1)
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Run inputs through embedding

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    # The MLP that determines the outcome
    x = Dropout(0.2)(mergea)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(class_num, activation="sigmoid")(x)
    model = Model(input_tensor, output_layer)
    loss1 = 'binary_crossentropy'
    model.compile(loss=[loss1], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open("cnn_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model

def cnn_model2():
    input_tensor = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                # weights=[embedding_matrix],
                                input_length=MAX_TEXT_LENGTH,
                                trainable=False)
    emb1 = embedding_layer(input_tensor)
    emb1 = SpatialDropout1D(0.2)(emb1)
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')
    # Run inputs through embedding
    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])

    # The MLP that determines the outcome
    x = Dropout(0.2)(mergea)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # output_layer = Dense(class_num, activation="sigmoid")(x)
    output_layer = Dense(law_class_num, activation="sigmoid")(x)
    # output_time = Dense(time_class_num, activation="softmax")(x)

    model = Model(input_tensor, output_layer)
    loss2 = 'binary_crossentropy'
    loss1 = 'categorical_crossentropy'
    model.compile(loss=[loss2], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open("cnn_model2.json", "w") as json_file:
        json_file.write(model_json)
    return model

def cnn_model3():
    input_tensor = Input(shape=(MAX_TEXT_LENGTH,), dtype='int32')
    embedding_layer = Embedding(MAX_FEATURES,
                                embedding_dims,
                                # weights=[embedding_matrix],
                                input_length=MAX_TEXT_LENGTH,
                                trainable=False)
    emb1 = embedding_layer(input_tensor)
    emb1 = SpatialDropout1D(0.2)(emb1)
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Run inputs through embedding

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])

    # The MLP that determines the outcome
    x = Dropout(0.2)(mergea)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # output_layer = Dense(class_num, activation="sigmoid")(x)
    # output_law = Dense(law_class_num, activation="sigmoid")(x)
    output_layer = Dense(time_class_num, activation="softmax")(x)

    model = Model(input_tensor, output_layer)
    # loss1 = 'binary_crossentropy'
    loss2 = 'categorical_crossentropy'
    model.compile(loss=loss2, optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open("cnn_model3.json", "w") as json_file:
        json_file.write(model_json)
    return model
