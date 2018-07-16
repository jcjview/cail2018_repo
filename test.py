import  keras

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
path='cnn_model1.json'
with open(path, 'r') as json_file:
    loaded_model_json = json_file.read()
cnn_model1 = keras.models.model_from_json(loaded_model_json)

cnn_model1.load_weights('cnn_model1test_0.82287.h5')

cnn_model1.save("model.h5")

keras.models.load_model("model.h5")