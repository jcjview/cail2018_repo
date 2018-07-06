import keras
import re

from sklearn.preprocessing import MultiLabelBinarizer

MAX_TEXT_LENGTH = 500
import jieba
from numpy import arange
special_character_removal = re.compile(r'[@#$%^&*,.【】[]{}；‘，。、？!?“”‘’; \\/"\']', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

word_len=2
# cut = thulac.thulac(seg_only=True)
def seg(text):
    text = special_character_removal.sub('', text)
    text = replace_numbers.sub('NUMBERREPLACE', text)
    seg_list = jieba.cut(text.strip())
    seg_list = [word for word in seg_list if len(word) >= word_len]
    return " ".join(seg_list)

def load_keras_model(path,custom_objects=None):
    with open(path, 'r') as json_file:
        loaded_model_json = json_file.read()
    if custom_objects==None:
        model = keras.models.model_from_json(loaded_model_json)
    else:
        model = keras.models.model_from_json(loaded_model_json,custom_objects=custom_objects)
    return model

def transform(x):
    n = len(x)
    x_return = arange(1, n + 1)[x > 0.5].tolist()
    if len(x_return) == 0:
        x_return = arange(1, n + 1)[x == x.max()].tolist()
    return x_return

def returnOneHot(label,lb):
    label = [set([i for i in row]) for row in label]
    oneHot = lb.transform(label)
    return oneHot