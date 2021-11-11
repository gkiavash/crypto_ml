import logging
import copy

import tensorflow as tf
import numpy as np
from binance.client import Client

import Binance_load_raw_dataset
import utils
import nn_train
import config as current_config
import config_test


logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("backtest.log")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)


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


def load_predict(current_config=current_config, config_test=config_test):
    Binance_load_raw_dataset.get_data(current_config=config_test)
    print('Data downloaded')

    model = tf.keras.models.load_model(current_config.NEURAL_NETWORK_FULL_PATH)
    print('Model loaded')

    X_test, y_test = nn_train.dataset_load(
        current_config=current_config,
        test_config=config_test,
        test=True
    )

    yhat = model.predict(X_test)
    return yhat, y_test


def percent_calc(price_new, price_base):
    return round((price_new - price_base) / price_base, 6)


def btc_buy(wallet_usdt_, wallet_btc_, btc_qty, btc_price, rate_fee_transaction):
    wallet_usdt_ = wallet_usdt_ - btc_qty * btc_price
    if wallet_usdt_ < 0:
        raise Exception('Not enough USDT')
    wallet_btc_ = wallet_btc_ + btc_qty - (btc_qty * rate_fee_transaction)

    log_str = f'Bought {btc_qty} with price {btc_price}, total_cost {btc_qty * btc_price}, ' \
              f'wallet_usdt: {wallet_usdt_}, wallet_btc: {wallet_btc_}'

    print(log_str)
    logger.info(log_str)
    return wallet_usdt_, wallet_btc_


def btc_sell(wallet_usdt_, wallet_btc_, btc_qty, btc_price, rate_fee_transaction):
    wallet_btc_ = wallet_btc_ - btc_qty
    if wallet_btc_ < -0.001:
        print(btc_qty, wallet_btc_)
        raise Exception('Not enough BTC')
    wallet_usdt_ = wallet_usdt_ + (btc_qty * btc_price) - (btc_qty * btc_price * rate_fee_transaction)

    log_str = f'Sold {btc_qty} with price {btc_price}, wallet_usdt: {wallet_usdt_}, wallet_btc: {wallet_btc_}'
    print(log_str)
    logger.info(log_str)
    return wallet_usdt_, wallet_btc_


def positions_check(positions, wallet_usdt, wallet_btc, btc_price, rate_fee_transaction, rate_sell_profit, rate_sell_stop_loss):
    positions_copy = copy.deepcopy(positions)
    for index, position in enumerate(positions):
        if percent_calc(btc_price, position['btc_price']) >= rate_sell_profit \
                or percent_calc(btc_price, position['btc_price']) < (rate_sell_stop_loss - 2 * rate_fee_transaction):
            wallet_usdt, wallet_btc = btc_sell(
                wallet_usdt_=wallet_usdt,
                wallet_btc_=wallet_btc,
                btc_qty=position['btc_qty'],
                btc_price=btc_price,
                rate_fee_transaction=rate_fee_transaction
            )
            positions_copy.remove(position)
            logger.info(f'Bought: {position["btc_price"]}, Sold: {btc_price}, amount: {position["btc_qty"]}')
            logger.info(f'wallet_usdt: {wallet_usdt}, wallet_btc: {wallet_btc}')
    return positions_copy, wallet_usdt, wallet_btc


def get_model_final_decision(model, input_seq):
    yhat = model.predict(input_seq)

    if len(yhat[0]) == 1:
        return True if yhat[0][0] >= 0.5 else False
    elif len(yhat[0]) == 2:
        yhat_ = np.argmax(yhat, axis=1)
        return True if yhat_[0] == 1 else False


col_names = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close'
}


def backtest(
        X_test,
        model,
        wallet_usdt=10000.0,
        wallet_btc=0.5,
        btc_qty=0.1,
        rate_fee_transaction=0.001,
        rate_sell_profit=0.008,
        rate_sell_stop_loss=-0.006
):
    positions = [
        # {
        #     'btc_price': 12000,
        #     'btc_qty': 0.05
        # }
    ]

    for row in X_test:
        if get_model_final_decision(model, row):
            wallet_usdt, wallet_btc = btc_buy(
                wallet_usdt_=wallet_usdt,
                wallet_btc_=wallet_btc,
                btc_qty=btc_qty,
                btc_price=row[col_names['close']],
                rate_fee_transaction=rate_fee_transaction
            )
            positions.append({'btc_price': row[col_names['close']],
                              'btc_qty': btc_qty*(1-rate_fee_transaction)})
        positions_check(
            positions,
            wallet_usdt,
            wallet_btc,
            btc_qty,
            rate_fee_transaction,
            rate_sell_profit,
            rate_sell_stop_loss
        )
        print(f'CURRENT: wallet_usdt: {wallet_usdt}, wallet_btc: {wallet_btc}')

