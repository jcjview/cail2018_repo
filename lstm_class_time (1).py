
# coding: utf-8

# In[ ]:


import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data=pd.read_csv('./csv/cail2018_big.csv')
# data.head()


# In[ ]:


from keras.layers import  Input,Embedding,Bidirectional,LSTM,GlobalMaxPool1D,GlobalAveragePooling1D,concatenate,Dense,Activation
from keras.models import  Model
from keras.layers import  Dropout,CuDNNLSTM
max_features=800000
maxlen=800
embed_size=300
def lstm_class_time_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, trainable=True)(inp)
    x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
    p1= GlobalMaxPool1D()(x)
    p2= GlobalAveragePooling1D()(x)


    conc=concatenate([p1,p2])
    fc1=Dense(256,activation='relu')(conc)
    fc1=Dropout(0.1)(fc1)


    fc2=Dense(128,activation='relu')(fc1)
    fc2=Dropout(0.1)(fc2)
    
    out=Dense(25,activation='softmax')(fc2)


    model=Model(inp,out)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    
    


# In[ ]:


from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import  pad_sequences
import os 
import pickle
if os.path.exists("./model/Tokenizer.pkl"):
    with open('./model/Tokenizer.pkl','rb') as f:
        tokenizer=pickle.load(f)
else:
    tokenizer=Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data['fact_cut_wd'].values)
    with open("./model/Tokenizer.pkl",'wb') as f:
        pickle.dump(tokenizer,f)
        
        
    

from sklearn.utils import  shuffle
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import  pad_sequences
data=shuffle(data)

train_data=data

train_data_text_list=train_data['fact_cut_wd'].values
train_data_seq=tokenizer.texts_to_sequences(train_data_text_list)
train_x=pad_sequences(train_data_seq,maxlen)


# In[ ]:


from keras.utils import  to_categorical

train_y=train_data['imp_label'].values
train_y=to_categorical(train_y.reshape(-1,1))


# In[ ]:


model=lstm_class_time_model()


# In[ ]:



from keras.callbacks import  EarlyStopping
model.fit(train_x,train_y,epochs=2,batch_size=128,validation_split=0.05,callbacks=[EarlyStopping()])
model.save("./model/lstm_class_time.model")
