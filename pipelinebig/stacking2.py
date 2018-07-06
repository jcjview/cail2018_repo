import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.model_selection import KFold
from config import *
import pandas as pd
import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from metric_fuc import predict2tag


def multi_lr(X_train, Y_train, X_val, y_val):
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    # bst_model_path = 'lr_weight_valid.h5'
    # model_checkpoint = ModelCheckpoint(bst_model_path, monitor='avg_f1_score_val', mode='max',
    #                                    save_best_only=True, verbose=1, save_weights_only=True)

    model.fit(X_train, Y_train, batch_size=512, epochs=200, verbose=0,
              callbacks=[early_stopping],
              validation_data=(X_val, y_val)
              )
    return model


def fit_cv(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits)
    estimators_model = []
    predict_oob = np.zeros(shape=y.shape)
    for train, valid in kf.split(X):
        X_train_ = X[train]
        y_train_ = y[train]
        X_valid_ = X[valid]
        y_valid_ = y[valid]
        estimator = multi_lr(X_train_, y_train_, X_valid_, y_valid_)
        estimators_model.append(estimator)
        predict = estimator.predict(X_valid_)
        predict_oob[valid] = predict
    return estimators_model, predict_oob


def predict_cv(estimators_model, x_test):
    predict_baging = []
    for model in estimators_model:
        predict = model.predict(x_test, batch_size=1024)
        predict_baging.append(predict)
    return np.sum(predict_baging) / len(predict_baging)

def calc(y_test,predict):
    y_pred = predict2tag(predict)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("macro f1_score %.4f " % f1)
    f2 = f1_score(y_test, y_pred, average='micro')
    print("micro f1_score %.4f " % f2)
    avgf1 = (f1 + f2) / 2
    print("avg_f1_score %.4f " % (avgf1))
    return avgf1
df = pd.read_csv(input_file, compression='infer', encoding="utf-8")
label = df['accu_label'].values

lb_y = MultiLabelBinarizer()
label = [set([int(i) for i in str(row).split(";")]) for row in label]
y_accu = lb_y.fit_transform(label)
print('accu y shape', y_accu.shape)

law_label = df['law_label'].fillna(0).values
law_label = [set([int(i) for i in str(row).split(";")]) for row in law_label]
lb_law = MultiLabelBinarizer()
law_label_y = lb_law.fit_transform(law_label)
print('law y shape', law_label_y.shape)

input_file1 = 'svm.csv'
df1 = pd.read_csv(input_file1, compression='infer', encoding="utf-8")

label1 = df1['accu_label'].values
law_label1 = df1['law_label'].fillna(0).values
label1 = [set([int(i) for i in str(row).split(";")]) for row in label1]
law_label1 = [set([int(i) for i in str(row).split(";")]) for row in law_label1]

test_accu3 = lb_y.transform(label1)
test_law3 = lb_law.transform(law_label1)

print('test_accu3 shape', test_accu3.shape)
print('test_law3 shape', test_law3.shape)

test1 = np.load("cnn_model2_test.npy")
test2 = np.load("bgru_model2_test.npy")
# test4 = np.load("imdb_model1_test.npy")

y_accu=law_label_y
test_accu3=test_law3
split1 = -17131
split2 = -32508
split = split1 + split2
y_test = y_accu[split2:]

calc(y_test,test1)

calc(y_test,test2)

calc(y_test,test_accu3)

blend=(test_accu3*0.9+test1+test2)/3
calc(y_test,blend)

blend=(test_accu3*0.95+test1+test2)/3
calc(y_test,blend)

blend=(test_accu3*0.85+test1+test2)/3
calc(y_test,blend)


blend=(test_accu3*0.9+test1*0.8+test2)/(2+0.8)
calc(y_test,blend)

X_valid = np.concatenate((test1, test2, test_accu3), axis=1)
label1 = np.array(y_test, copy=True)
label2 = np.array(y_test, copy=True)
y_valid_multilabel = y_test  # np.concatenate((y_test,label1,label2),axis=1)

print('x shape', X_valid.shape)
print('y_valid_multilabel', y_valid_multilabel.shape)

# estimators_model, predict = fit_cv(X_valid, y_valid_multilabel)
# predict=predict_cv(estimators_model,X_valid)
#
# calc(y_test,predict)


