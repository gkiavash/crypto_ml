import logging
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers import Flatten, LSTM, GlobalMaxPooling1D, Input, Bidirectional
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


DATETIME_STR_FORMAT = '%Y-%m-%d %H:%M'


def to_supervised(train, n_out=7): # (x,1,y)
    train = train.tolist()
    for row_num in range(len(train)):
        for row_seq in range(1, n_out):
            ind_adding = row_num-row_seq
            if ind_adding < 0:
                ind_adding = 0
            train[row_num].append(train[ind_adding][0])
    return np.array(train)


def nn_train(df, INPUT_DATASET_PATH, OUTPUT_FULL_PATH, return_model):
    logging.info("###### " + OUTPUT_FULL_PATH + " ######")
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, header=0, index_col=0)

    df.dropna(inplace=True)

    cols_drop = ['open', 'high', 'low', 'close', 'symbol']
    df.drop(cols_drop, axis=1, inplace=True)

    print(df.head())

    properties = list(df.columns.values)
    properties.remove('signal_buy')

    # print('Check if o exists')
    # for i in properties:
    #     if df[i].sum() == 0.:
    #         print(i)
    #         properties.remove(i)

    X = df[properties]
    X.astype('float64')
    y = df['signal_buy']
    y.astype('category')

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)

    scaler = preprocessing.MinMaxScaler(
        # (-1, 1)
    )
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_train.shape[0], X_train.shape[1], type(X_train.shape[0]))
    print(X_test.shape[0], X_test.shape[1], type(X_test.shape[0]))

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(X_train.shape, X_test.shape)

    X_train = to_supervised(X_train, 10)
    X_test = to_supervised(X_test, 10)
    print(X_train.shape, X_test.shape)

    model = Sequential()
    model.add(LSTM(
        11,
        activation='relu',
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True,
    ))
    # model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(5, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=512, validation_split=0.3, shuffle=False)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    if return_model:
        return model
    model.save(
        OUTPUT_FULL_PATH,
        # save_format="h5"
    )


def nn_load(NN_PATH, X_test):
    model = keras.models.load_model(NN_PATH)
    yhat = model.predict(X_test)
    print(yhat)