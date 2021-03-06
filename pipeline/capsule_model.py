from keras.engine import Layer
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Flatten, CuDNNGRU, BatchNormalization, Dense, \
    Dropout, Bidirectional, K, Activation

from config import *
kernel_name = "capsule"

dropout_p=0.2
gru_len=256
Num_capsule = 16
Dim_capsule = 32
Routings = 5


def capsule_model3(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(dropout_p)(embed_layer)
    x = Bidirectional(
        CuDNNGRU(gru_len, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output_layer = Dense(1, activation="linear")(x)
    output_layer2 = Dense(1, activation="sigmoid",name='death')(x)
    output_layer3 = Dense(1, activation="sigmoid",name='life')(x)

    model = Model(input1, [output_layer,output_layer2,output_layer3])
    loss1 = 'binary_crossentropy'
    loss2 = 'mse'
    model.compile(loss=[loss2,loss1,loss1], optimizer='adam', metrics=["accuracy"])
    model.summary()
    model_json = model.to_json()
    with open(kernel_name + "_model3.json", "w") as json_file:
        json_file.write(model_json)
    return model

def capsule_model2(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(dropout_p)(embed_layer)
    x = Bidirectional(
        CuDNNGRU(gru_len, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(law_class_num, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    model_json = model.to_json()
    with open("capsule_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model

def capsule_model1(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(dropout_p)(embed_layer)
    x = Bidirectional(
        CuDNNGRU(gru_len, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(class_num, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    model_json = model.to_json()
    with open("capsule_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model

def capsule_model2(embedding_matrix):
    input1 = Input(shape=(MAX_TEXT_LENGTH,))
    embed_layer = Embedding(MAX_FEATURES,
                            embedding_dims,
                            input_length=MAX_TEXT_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(dropout_p)(embed_layer)
    x = Bidirectional(
        CuDNNGRU(gru_len, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(law_class_num, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    model_json = model.to_json()
    with open("capsule_model1.json", "w") as json_file:
        json_file.write(model_json)
    return model


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
