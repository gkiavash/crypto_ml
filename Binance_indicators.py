import logging
import pandas as pd

from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, PSARIndicator, MACD

import config as current_config


def add_indicators(df, INPUT_DATASET_PATH, OUTPUT_FULL_PATH, return_df, col_names):
    logging.info("###### " + OUTPUT_FULL_PATH + " ######")
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, header=0, delimiter=",")
        logging.info('dataset found: {}'.format(df.columns.tolist()))

    df.dropna(inplace=True)
    df.insert(1, 'ones', 1)

    indicator_bb = BollingerBands(close=df[col_names['close']], window=21, window_dev=2)
    # Add Bollinger Bands features
    # df['bb_bbm'] = indicator_bb.bollinger_mavg().round(2)
    # df['bb_bbh'] = indicator_bb.bollinger_hband().round(2)
    # df['bb_bbl'] = indicator_bb.bollinger_lband().round(2)

    # Add Bollinger Band high indicator
    # df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    # Add Bollinger Band low indicator
    # df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add Width Size Bollinger Bands
    # df['bb_bbw'] = indicator_bb.bollinger_wband().round(6)
    df['bb_bbp'] = indicator_bb.bollinger_pband().round(4)
    # df['bb_bbp'] = df.apply(lambda row: -row['bb_bbp'] + 1, axis=1).round(4)
    print('Done: ', 'Boll')

    df['rsi_6_'] = RSIIndicator(close=df[col_names['close']], window=6).rsi().round(2)
    # df['rsi_12_'] = RSIIndicator(close=df[col_names['close']], window=12).rsi().round(2)
    # df['rsi_24_'] = RSIIndicator(close=df[col_names['close']], window=24).rsi().round(2)

    # df['bb_bbp'] = (df['ones']/df['bb_bbp']).round(4).abs().apply(lambda x: x if x < 10000 else 10000)
    # df['rsi_6_'] = (df['ones']/df['rsi_6_']).round(4)
    print('Done: ', 'RSI')

    df['ema_7_'] = EMAIndicator(close=df[col_names['close']], window=7, ).ema_indicator().round(2)
    df['ema_25_'] = EMAIndicator(close=df[col_names['close']], window=25, ).ema_indicator().round(2)
    df['ema_99_'] = EMAIndicator(close=df[col_names['close']], window=99, ).ema_indicator().round(2)

    # df['ema_7_25'] = df.apply(lambda row: 1 if row['ema_7_'] > row['ema_25_'] else 0, axis=1).astype('category')
    # df['ema_7_99'] = df.apply(lambda row: 1 if row['ema_7_'] > row['ema_99_'] else 0, axis=1).astype('category')
    # df['ema_25_99'] = df.apply(lambda row: 1 if row['ema_25_'] > row['ema_99_'] else 0, axis=1).astype('category')

    df['ema_7_25'] = (df['ema_7_'] / df['ema_25_']).round(3)
    df['ema_25_99'] = (df['ema_25_'] / df['ema_99_']).round(3)
    df['ema_7_99'] = (df['ema_7_'] / df['ema_99_']).round(3)
    df.drop([
        'ema_7_',
        'ema_25_',
        'ema_99_',
    ], axis=1, inplace=True)
    print('Done: ', 'EMA')

    MACDIndcator = MACD(close=df[col_names['close']], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = MACDIndcator.macd().round(2)
    df['macd_diff'] = MACDIndcator.macd_diff().round(2)
    df['macd_signal'] = MACDIndcator.macd_signal().round(2)
    print('Done: ', 'MACD')

    df['sar'] = PSARIndicator(high=df[col_names['high']], low=df[col_names['low']], close=df[col_names['close']],
                              step=0.02, max_step=0.2).psar().round(2)
    df['sar_'] = df.apply(lambda row: 1 if row['sar'] < row[col_names['close']] else 0, axis=1).astype('category')
    df.drop(['sar'], axis=1, inplace=True)
    print('Done: ', 'SAR')

    df['hour_percent_profit'] = df.apply(
        lambda row: (row[col_names['close']] - row[col_names['open']]) / row[col_names['open']],
        axis=1
    ).round(4)
    print('Done: ', 'hour_percent_profit')

    df['candle'] = df.apply(
        lambda row: (row[col_names['close']] - row[col_names['open']]) /
                    (row[col_names['high']] - row[col_names['low']])
        if row[col_names['high']] - row[col_names['low']] != 0 else 0,
        axis=1
    ).round(4)
    print('Done: ', 'candle')

    df.drop(['ones'], axis=1, inplace=True)
    if return_df:
        return df

    df.set_index('unix', inplace=True)
    df.to_csv(
        OUTPUT_FULL_PATH,
        encoding='utf-8',
        index=True
    )
