import logging

import pandas
import pandas as pd

from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, PSARIndicator, MACD


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


def add_indicators_BollingerBands(df, col_names, window=21, window_dev=2, ones_reverse=False):
    indicator_bb = BollingerBands(close=df[col_names['close']], window=window, window_dev=window_dev)
    # df['bb_bbm'] = indicator_bb.bollinger_mavg().round(2)
    # df['bb_bbh'] = indicator_bb.bollinger_hband().round(2)
    # df['bb_bbl'] = indicator_bb.bollinger_lband().round(2)

    # Add Bollinger Band high indicator
    # df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    # Add Bollinger Band low indicator
    # df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add Width Size Bollinger Bands
    # df['bb_bbw'] = indicator_bb.bollinger_wband().round(6)
    df['bb_bbp'] = indicator_bb.bollinger_pband().round(3)

    if ones_reverse:
        df.insert(1, 'ones', 1)
        df['bb_bbp'] = (df['ones']/df['bb_bbp']).round(4).abs().apply(lambda x: x if x < 10000 else 10000)
        df.drop(['ones'], axis=1, inplace=True)
    print('Done: ', 'Boll')
    return df


def add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False):
    df[f'rsi_{window}_'] = RSIIndicator(close=df[col_names['close']], window=window).rsi().round(2)

    logging.info(f'for rsi_{window}_')
    logging.info(df[df[f'rsi_{window}_'] > 60].groupby(['signal_buy']).count())

    if ones_reverse:
        df.insert(1, 'ones', 1)
        df['rsi_{}_'.format(window)] = (df['ones']/df['rsi_{}_'.format(window)]).round(4)
        df.drop(['ones'], axis=1, inplace=True)
    print('Done: ', 'RSI')
    return df


def add_indicators_EMAIndicator(df, col_names, w1=7, w2=25, w3=99, extra=None):
    df['ema_{}_'.format(w1)] = EMAIndicator(close=df[col_names['close']], window=w1).ema_indicator().round(2)
    df['ema_{}_'.format(w2)] = EMAIndicator(close=df[col_names['close']], window=w2).ema_indicator().round(2)
    df['ema_{}_'.format(w3)] = EMAIndicator(close=df[col_names['close']], window=w3).ema_indicator().round(2)

    if extra == 'binary':
        df[f'ema_{w1}_{w2}'] = df.apply(lambda row: 1 if row[f'ema_{w1}_'] > row[f'ema_{w2}_'] else 0, axis=1)\
            .astype('category')
        df[f'ema_{w1}_{w3}'] = df.apply(lambda row: 1 if row[f'ema_{w1}_'] > row[f'ema_{w3}_'] else 0, axis=1)\
            .astype('category')
        df[f'ema_{w2}_{w3}'] = df.apply(lambda row: 1 if row[f'ema_{w2}_'] > row[f'ema_{w3}_'] else 0, axis=1)\
            .astype('category')

    elif extra == 'propostional':
        df[f'ema_{w1}_{w2}'] = (df[f'ema_{w1}_'] / df[f'ema_{w2}_']).round(3)
        df[f'ema_{w1}_{w3}'] = (df[f'ema_{w1}_'] / df[f'ema_{w3}_']).round(3)
        df[f'ema_{w2}_{w3}'] = (df[f'ema_{w2}_'] / df[f'ema_{w3}_']).round(3)
    if extra is not None:
        df.drop(['ema_7_', 'ema_25_', 'ema_99_'], axis=1, inplace=True)
    print('Done: ', 'EMA')
    return df


def add_indicators_MACDIndcator(df, col_names, window_slow=26, window_fast=12, window_sign=9):
    MACDIndcator = MACD(close=df[col_names['close']], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    df['macd'] = MACDIndcator.macd().round(2)
    df['macd_diff'] = MACDIndcator.macd_diff().round(2)
    df['macd_signal'] = MACDIndcator.macd_signal().round(2)
    print('Done: ', 'MACD')
    return df


def add_indicators_PSARIndicator(df, col_names):
    df['sar'] = PSARIndicator(
        high=df[col_names['high']],
        low=df[col_names['low']],
        close=df[col_names['close']],
        step=0.02,
        max_step=0.2
    ).psar().round(2)

    df['sar_'] = df.apply(lambda row: 1 if row['sar'] < row[col_names['close']] else 0, axis=1).astype('category')

    df.drop(['sar'], axis=1, inplace=True)
    print('Done: ', 'SAR')
    return df


def add_indicators_hour_percent_profit(df, col_names):
    df['hour_percent_profit'] = df.apply(
        lambda row: (row[col_names['close']] - row[col_names['open']]) / row[col_names['open']],
        axis=1
    ).round(4)
    print('Done: ', 'hour_percent_profit')
    return df


def add_indicators_candle(df, col_names):
    df['candle'] = df.apply(
        lambda row: (row[col_names['close']] - row[col_names['open']]) /
                    (row[col_names['high']] - row[col_names['low']])
        if row[col_names['high']] - row[col_names['low']] != 0 else 0,
        axis=1
    ).round(4)
    print('Done: ', 'candle')
    return df


def add_m(df, column, shift, need_fraction):
    df[f'{column}_{shift}'] = df[column].shift(shift)
    if not need_fraction:
        df[f'{column}_{shift}_m'] = df.apply(lambda row: (row[column] - row[f'{column}_{shift}']), axis=1).round(4)
    else:
        df[f'{column}_{shift}_m'] = df.apply(
            lambda row: (row[column] - row[f'{column}_{shift}']) / row[f'{column}_{shift}'], axis=1).round(4)

    logging.info(f'for {column}_{shift}_m < 0')
    logging.info(df[df[f'{column}_{shift}_m'] < 0].groupby(['signal_buy']).count())

    logging.info(f'for {column}_{shift}_m > 0')
    logging.info(df[df[f'{column}_{shift}_m'] > 0].groupby(['signal_buy']).count())
    # df.drop([f'{column}_{shift}'], axis=1, inplace=True)
    return df


def add_counters(df):
    rsi_window = 6
    # df[f'rsi_{rsi_window}_count_{20}_rows'] = sum([1 if df[f'rsi_{rsi_window}_'].shift(i) <= 40 else 0 for i in range(21)])
    # df[f'bb_bbp_count_{20}_rows'] = sum([1 if df['bb_bbp'].shift(i) <= 0.5 else 0 for i in range(21)])

    col_names = []
    for i in range(21):
        col_names.append(f'rsi_{rsi_window}_count_{i}_rows')
        df[f'rsi_{rsi_window}_count_{i}_rows'] = df[f'rsi_{rsi_window}_'].shift(i)
    df[f'rsi_{rsi_window}_count'] = df.apply(lambda row: sum([1 if row[c] <= 40 else 0 for c in col_names]), axis=1)
    df.drop(col_names, axis=1, inplace=True)

    col_names = []
    for i in range(21):
        col_names.append(f'bb_bbp_count_{i}_rows')
        df[f'bb_bbp_count_{i}_rows'] = df['bb_bbp'].shift(i)
    df[f'bb_bbp_count_count'] = df.apply(lambda row: sum([1 if row[c] <= 0.5 else 0 for c in col_names]), axis=1)
    df.drop(col_names, axis=1, inplace=True)

    return df


def add_price_change_counter(df, col_names):
    def v(row):
        res = []
        for index, col_name in enumerate(col_names):
            try:
                res.append(1 if row[col_names[index]]*row[col_names[index+1]] <= 0 else 0)
            except:
                res.append(0)
        return sum(res)

    window = 15

    col_names = []
    for i in range(window):
        col_names.append(f'hour_percent_profit_count_{i}_rows')
        df[f'hour_percent_profit_count_{i}_rows'] = df[f'hour_percent_profit'].shift(i)

    df[f'hour_percent_profit_count'] = df.apply(lambda row: v(row), axis=1)
    df.drop(col_names, axis=1, inplace=True)
    return df


def add_multiply(df, col_1, col_2):
    return df[col_1] * df[col_2]


def df_logging(df, INPUT_DATASET_PATH):
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, delimiter=",")

    logging.info(f'for bb_bbp < 0.4')
    logging.info(df[df[f'bb_bbp'] < 0.4].groupby(['signal_buy']).count())

    logging.info(f'for bb_bbp < 0.3')
    logging.info(df[df[f'bb_bbp'] < 0.3].groupby(['signal_buy']).count())

    logging.info(f'for bb_bbp < 0.2')
    logging.info(df[df[f'bb_bbp'] < 0.3].groupby(['signal_buy']).count())

    logging.info('for sar_')
    logging.info(df[df['sar_'] == 1].groupby(['signal_buy']).count())

    w1 = 7
    w2 = 25
    w3 = 99

    logging.info(f'for ema_{w1}_{w2}')
    logging.info(df[df[f'ema_{w1}_{w2}'] == 1].groupby(['signal_buy']).count())

    logging.info(f'for ema_{w1}_{w3}')
    logging.info(df[df[f'ema_{w1}_{w3}'] == 1].groupby(['signal_buy']).count())

    logging.info(f'for ema_{w2}_{w3}')
    logging.info(df[df[f'ema_{w2}_{w3}'] == 1].groupby(['signal_buy']).count())

    logging.info(f'for ema_{w1}_{w2}_{w3}')
    logging.info(df[(df[f'ema_{w1}_{w2}'] == 1) &
                    (df[f'ema_{w2}_{w3}'] == 1) &
                    (df[f'ema_{w1}_{w3}'] == 1)].groupby(['signal_buy']).count())


def add_indicators_v2(df, INPUT_DATASET_PATH, OUTPUT_FULL_PATH, return_df, col_names):
    logging.info("###### Adding Indicators ######")
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, header=0, delimiter=",")
        # logging.info('dataset found: {}'.format(df.columns.tolist()))
    df.dropna(inplace=True)

    # df = add_indicators_cs1(df, col_names)
    # df = add_indicators_cs2(df, col_names)
    df = add_indicators_cs3(df, col_names)

    if return_df:
        return df

    logging.info(f"###### Saving to {OUTPUT_FULL_PATH} ######")
    df.set_index('unix', inplace=True)
    df.to_csv(
        OUTPUT_FULL_PATH,
        encoding='utf-8',
        index=True
    )


def add_indicators_cs1(df, col_names):
    df = add_indicators_BollingerBands(df, col_names, window=21, window_dev=2, ones_reverse=False)
    df = add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False)
    df = add_indicators_EMAIndicator(df, col_names, w1=7, w2=25, w3=99, extra='binary')
    df = add_indicators_MACDIndcator(df, col_names, window_slow=26, window_fast=12, window_sign=9)
    df = add_indicators_PSARIndicator(df, col_names)
    df = add_indicators_hour_percent_profit(df, col_names)
    df = add_indicators_candle(df, col_names)
    return df


def add_indicators_cs2(df, col_names):
    df = add_indicators_BollingerBands(df, col_names, window=21, window_dev=2, ones_reverse=False)
    df = add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False)
    df = add_indicators_EMAIndicator(df, col_names, w1=7, w2=25, w3=99, extra='binary')
    # df = add_indicators_MACDIndcator(df, col_names, window_slow=26, window_fast=12, window_sign=9)
    df = add_indicators_PSARIndicator(df, col_names)
    df = add_indicators_hour_percent_profit(df, col_names)
    # df = add_indicators_candle(df, col_names)
    # df = add_m(df, column='bb_bbp', shift=3, need_fraction=False)
    df = add_counters(df)

    return df


def add_indicators_cs3(df, col_names):
    # combine time only in bollinger and rsi
    df = add_indicators_BollingerBands(df, col_names, window=21, window_dev=2, ones_reverse=False)
    df = add_indicators_RSIIndicator(df, col_names, window=6, ones_reverse=False)
    df = add_indicators_hour_percent_profit(df, col_names)
    df = add_counters(df)
    df = add_price_change_counter(df, col_names)
    return df
