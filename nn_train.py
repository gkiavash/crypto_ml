import logging
import pandas as pd
import numpy as np
import joblib

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

import config as current_config


def to_supervised(train, n_out=7):  # (x,1,y)
    train = train.tolist()
    for row_num in range(len(train)):
        for row_seq in range(1, n_out):
            ind_adding = row_num-row_seq
            if ind_adding < 0:
                ind_adding = 0
            train[row_num].append(train[ind_adding][0])
    return np.array(train)


def dataset_load(current_config=current_config, test_config=None, test=False):
    if test:
        INPUT_DATASET_PATH = test_config.INDICATORS_DATASET_FULL_PATH
    else:
        INPUT_DATASET_PATH = current_config.INDICATORS_DATASET_FULL_PATH

    df = pd.read_csv(INPUT_DATASET_PATH, header=0, index_col=0)
    df.dropna(inplace=True)
    df.drop(['open', 'high', 'low', 'close', 'symbol'], axis=1, inplace=True)

    if not test:
        is_hol = df['signal_buy'] == 1
        df_try = df[is_hol]
        df = df.append(df_try)
        df = df.append(df_try)
        df = df.append(df_try)

    print(df.head())
    print(df.groupby(['signal_buy']).count())
    print(df.corr(method='pearson'))

    properties = list(df.columns.values)
    properties.remove('signal_buy')

    X = df[properties].astype('float32')
    y = df['signal_buy'].astype('category')

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    y = to_categorical(y)

    if test:
        scaler = joblib.load(current_config.SCALE_FULL_PATH)
        scaler.clip = False
        X = scaler.fit_transform(X)

        print(X.shape[0], X.shape[1], type(X.shape[0]))
        X_train = X.reshape(X.shape[0], 1, X.shape[1])
        print(X_train.shape)
        X_train = to_supervised(X_train, 8)
        print(X_train.shape,)
        return X_train, y

    scaler = preprocessing.MinMaxScaler(
        (-1, 1)
    )
    X = scaler.fit_transform(X)
    joblib.dump(scaler, current_config.SCALE_DIRECTORY)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_train.shape[0], X_train.shape[1], type(X_train.shape[0]))
    print(X_test.shape[0], X_test.shape[1], type(X_test.shape[0]))

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(X_train.shape, X_test.shape)

    X_train = to_supervised(X_train, 8)
    X_test = to_supervised(X_test, 8)
    print(X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test


def model_lstm(X_train):
    model = Sequential()
    model.add(LSTM(
        11,
        activation='relu',
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True
        # stateful=False
    ))
    # model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(5, activation='relu'))
    # model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='tanh'))
    # model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=[tf.keras.metrics.CategoricalAccuracy()])
    print(model.summary())
    return model


def nn_train():
    NEW_FILE_NAME_ = 'BTCUSDT_5m_1 Jan, 2019_30 Jul, 2021_6_steps_0.006_percent_profit_indicators'

    X_train, X_test, y_train, y_test = dataset_load(current_config=current_config)
    model = model_lstm(X_train)

    weights = {0: 1, 1: 1}
    model.fit(X_train, y_train, epochs=200, batch_size=4096, validation_split=0.3, shuffle=True, class_weight=weights)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    model.save(
        current_config.NEURAL_NETWORK_FULL_PATH,
        # save_format="h5"
    )
