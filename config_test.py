import os
import logging

from binance.client import Client

logging.basicConfig(
    filename='Binance_api.log',
    level=logging.INFO,
    datefmt='%d-%b-%y %H:%M:%S'
)


SYMBOL = 'BTCUSDT'
INTERVAL = Client.KLINE_INTERVAL_5MINUTE
RAW_DATASET_START_DATETIME = "1 Aug, 2021"
RAW_DATASET_END_DATETIME = "20 Aug, 2021"

NEW_FILE_NAME_WITHOUT_EXTENSION = '{SYMBOL}_{INTERVAL}_{START}_{END}'.format(
    SYMBOL=SYMBOL,
    INTERVAL=INTERVAL,
    START=RAW_DATASET_START_DATETIME,
    END=RAW_DATASET_END_DATETIME
)

RAW_DATASET_DIRECTORY = 'datasets_raw'
RAW_DATASET_FULL_PATH = '{DIRECTORY}/{FILENAME}.csv'.format(
    DIRECTORY=RAW_DATASET_DIRECTORY,
    FILENAME=NEW_FILE_NAME_WITHOUT_EXTENSION
)
if not os.path.exists(RAW_DATASET_DIRECTORY):
    os.makedirs(RAW_DATASET_DIRECTORY)


SIGNAL_MAX_MINUTE_LATER = 6
SIGNAL_MIN_PERCENT_PROFIT = 0.006
SIGNAL_DATASET_DIRECTORY = 'datasets_signal'
SIGNAL_DATASET_FILE_NAME = NEW_FILE_NAME_WITHOUT_EXTENSION+'_{steps}_steps_{percent}_percent_profit'.format(
    steps=SIGNAL_MAX_MINUTE_LATER,
    percent=SIGNAL_MIN_PERCENT_PROFIT
)

SIGNAL_DATASET_FULL_PATH = '{DIRECTORY}/{FILENAME}.csv'.format(
    DIRECTORY=SIGNAL_DATASET_DIRECTORY,
    FILENAME=SIGNAL_DATASET_FILE_NAME
)
if not os.path.exists(SIGNAL_DATASET_DIRECTORY):
    os.makedirs(SIGNAL_DATASET_DIRECTORY)


INDICATORS_DATASET_DIRECTORY = 'datasets_indicators'
INDICATORS_DATASET_FILE_NAME = SIGNAL_DATASET_FILE_NAME + '_indicators'
INDICATORS_DATASET_FULL_PATH = '{DIRECTORY}/{FILENAME}.csv'.format(
    DIRECTORY=INDICATORS_DATASET_DIRECTORY,
    FILENAME=INDICATORS_DATASET_FILE_NAME
)
if not os.path.exists(INDICATORS_DATASET_DIRECTORY):
    os.makedirs(INDICATORS_DATASET_DIRECTORY)


NEURAL_NETWORK_DIRECTORY = 'neural.networks'
NEURAL_NETWORK_FULL_PATH = '{DIRECTORY}/{FILENAME}'.format(
    DIRECTORY=NEURAL_NETWORK_DIRECTORY,
    FILENAME=INDICATORS_DATASET_FILE_NAME
)
if not os.path.exists(INDICATORS_DATASET_DIRECTORY):
    os.makedirs(INDICATORS_DATASET_DIRECTORY)
