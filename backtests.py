import tensorflow as tf
import numpy as np
from binance.client import Client

import Binance_load_raw_dataset
import utils
import config as current_config
import config_test


class Config:
    def __init__(self):
        self.SYMBOL = 'BTCUSDT'
        self.INTERVAL = Client.KLINE_INTERVAL_5MINUTE
        self.RAW_DATASET_START_DATETIME = "1 Aug, 2021"
        self.RAW_DATASET_END_DATETIME = "20 Aug, 2021"

        self.NEW_FILE_NAME_WITHOUT_EXTENSION = '{SYMBOL}_{INTERVAL}_{START}_{END}'.format(
            SYMBOL=self.SYMBOL,
            INTERVAL=self.INTERVAL,
            START=self.RAW_DATASET_START_DATETIME,
            END=self.RAW_DATASET_END_DATETIME
        )

        self.RAW_DATASET_DIRECTORY = 'datasets_test'
        self.RAW_DATASET_FULL_PATH = '{DIRECTORY}/{FILENAME}.csv'.format(
            DIRECTORY=self.RAW_DATASET_DIRECTORY,
            FILENAME=self.NEW_FILE_NAME_WITHOUT_EXTENSION
        )


config_model_test = Config()

Binance_load_raw_dataset.get_data(current_config=config_model_test)

model = tf.keras.models.load_model(current_config.NEURAL_NETWORK_FULL_PATH)

X_test, y_test = utils.dataset_test_load(config_test.INDICATORS_DATASET_FULL_PATH)

yhat = model.predict(X_test)
yhat_ = np.argmax(yhat, axis=1)
y_test_ = np.argmax(y_test, axis=1)

print(yhat.shape)
print(yhat_.shape)
print(y_test.shape)
print(y_test_.shape)

# for i in range(100):
# print(yhat_[i], yhat[i], y_test[i])

yhat_p = []
for i in range(len(yhat_)):
    if yhat_[i] == 1:
        yhat_p.append(yhat[i, 1])
print(yhat_p)
print(sum(yhat_p) / len(yhat_p))

#################  num of ones predicted
yhat_1 = 0
yhat_0 = 0
for i in yhat_:
    if i == 1:
        yhat_1 += 1
    else:
        yhat_0 += 1
print('num of ones predicted: ', yhat_0, yhat_1)

#################  num of ones actual
y_test_1 = 0
y_test_0 = 0
for i in y_test_:
    if i == 1:
        y_test_1 += 1
    else:
        y_test_0 += 1
print('num of ones actual: ', y_test_0, y_test_1)

#################  num of ones actual and predicted together
yeke_p = []
yeke_ok = 0
yeke_wrong = 0
for i in range(len(yhat_)):
    if yhat_[i] == 1 and y_test_[i] == 1:
        yeke_ok += 1
        yeke_p.append(yhat[i, 1])
    elif yhat_[i] == 1 and y_test_[i] == 0:
        yeke_wrong += 1
print('num of ones actual and predicted together: ', yeke_wrong, yeke_ok)

ave_rate = sum(yeke_p) / len(yeke_p)
print('ave_rate: ', ave_rate)
ave_rate = 0.95

#################  num of ones actual and predicted togheder above a rate
yeke_ok_ = 0
yeke_wrong_ = 0
for i in range(len(yhat)):
    if yhat[i, 1] >= ave_rate and y_test_[i] == 1:
        yeke_ok_ += 1
    elif yhat[i, 1] >= ave_rate and y_test_[i] == 0:
        yeke_wrong_ += 1
print('num of ones actual and predicted togheder above a rate: ', yeke_wrong_, yeke_ok_)
print('rate of 1 prediction: ', (yeke_wrong_ + yeke_ok_) / len(X_test))
