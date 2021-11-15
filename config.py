import os
import logging

from binance.client import Client

logging.basicConfig(filename='Binance_api.log', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')


class Config:
    def __init__(self, SYMBOL, INTERVAL, RAW_DATASET_START_DATETIME, RAW_DATASET_END_DATETIME, SIGNAL_MAX_MINUTE_LATER, SIGNAL_MIN_PERCENT_PROFIT, SIGNAL_MIN_PERCENT_LOSS,):
        self.SYMBOL = SYMBOL
        self.INTERVAL = INTERVAL
        self.RAW_DATASET_START_DATETIME = RAW_DATASET_START_DATETIME
        self.RAW_DATASET_END_DATETIME = RAW_DATASET_END_DATETIME
        self.SIGNAL_MAX_MINUTE_LATER = SIGNAL_MAX_MINUTE_LATER
        self.SIGNAL_MIN_PERCENT_PROFIT = SIGNAL_MIN_PERCENT_PROFIT
        self.SIGNAL_MIN_PERCENT_LOSS = SIGNAL_MIN_PERCENT_LOSS
        self.create_vars_and_dirs()

    def create_vars_and_dirs(self):
        self.NEW_FILE_NAME_WITHOUT_EXTENSION = f'{self.SYMBOL}_{self.INTERVAL}_{self.RAW_DATASET_START_DATETIME}_{self.RAW_DATASET_END_DATETIME}'

        self.RAW_DATASET_DIRECTORY = 'datasets_raw'
        self.RAW_DATASET_FULL_PATH = f'{self.RAW_DATASET_DIRECTORY}/{self.NEW_FILE_NAME_WITHOUT_EXTENSION}.csv'
        if not os.path.exists(self.RAW_DATASET_DIRECTORY):
            os.makedirs(self.RAW_DATASET_DIRECTORY)


        self.SIGNAL_DATASET_DIRECTORY = 'datasets_signal'
        self.SIGNAL_DATASET_FILE_NAME = self.NEW_FILE_NAME_WITHOUT_EXTENSION+f'_{self.SIGNAL_MAX_MINUTE_LATER}_steps_{self.SIGNAL_MIN_PERCENT_PROFIT}_percent_profit'
        self.SIGNAL_DATASET_FULL_PATH = f'{self.SIGNAL_DATASET_DIRECTORY}/{self.SIGNAL_DATASET_FILE_NAME}.csv'
        if not os.path.exists(self.SIGNAL_DATASET_DIRECTORY):
            os.makedirs(self.SIGNAL_DATASET_DIRECTORY)


        self.INDICATORS_DATASET_DIRECTORY = 'datasets_indicators'
        self.INDICATORS_DATASET_FILE_NAME = self.SIGNAL_DATASET_FILE_NAME + '_indicators'
        self.INDICATORS_DATASET_FULL_PATH = f'{self.INDICATORS_DATASET_DIRECTORY}/{self.INDICATORS_DATASET_FILE_NAME}.csv'
        if not os.path.exists(self.INDICATORS_DATASET_DIRECTORY):
            os.makedirs(self.INDICATORS_DATASET_DIRECTORY)


        self.NEURAL_NETWORK_DIRECTORY = 'neural.networks'
        self.NEURAL_NETWORK_FULL_PATH = f'{self.NEURAL_NETWORK_DIRECTORY}/{self.INDICATORS_DATASET_FILE_NAME}'
        if not os.path.exists(self.NEURAL_NETWORK_DIRECTORY):
            os.makedirs(self.NEURAL_NETWORK_DIRECTORY)


        self.SCALE_DIRECTORY = 'datasets_scale'
        self.SCALE_FULL_PATH = f'{self.SCALE_DIRECTORY}/{self.INDICATORS_DATASET_FILE_NAME}.gz'
        if not os.path.exists(self.SCALE_DIRECTORY):
            os.makedirs(self.SCALE_DIRECTORY)
