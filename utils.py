import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import config as current_config

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


def add_previous_rows(df, columns, shifts):
    for col in columns:
        for shift in shifts:
            df['{}_{}'.format(col, shift)] = df[col].shift(shift)
    return df


def percent_calc(price_new, price_base):
    return round((price_new - price_base) / price_base, 6)


def signal_buy_(
    df,
    INPUT_DATASET_PATH,
    OUTPUT_FULL_PATH,
    max_minutes_later,
    min_percent_profit,
    min_percent_loss,
    col_names,
    return_df,
    header_row,
    index_col,
    need_reverse=False,
    lower_bound=False,
    remove_extra_cols=True,
):
    logging.info("###### signal_buy_: "+OUTPUT_FULL_PATH+" ######")
    if df is None:
        df = pd.read_csv(INPUT_DATASET_PATH, header=header_row, delimiter=",")
        logging.info('dataset found: {}'.format(df.columns.tolist()))

    if need_reverse:
        df = df[::-1].reset_index()

    col_close = col_names['close']
    col_high = col_names['high']

    col_names_max = [f'{col_names["high"]}_{shift}' for shift in range(1, max_minutes_later + 1)]
    for shift in range(1, max_minutes_later + 1):
        df[f'{col_names["high"]}_{shift}'] = df[col_high].shift(-shift)
    logging.info('Done: shift')

    df['percent_max_future_rows'] = df.apply(
        lambda row: round((row[col_names_max].max() - row[col_close]) / row[col_close], 4),
        axis=1
    ).round(4)
    logging.info('Done: percent_max_future_rows')

    if lower_bound:
        logging.info('lower_bound: True')
        col_names_min = [f'{col_names["low"]}_{shift}' for shift in range(1, max_minutes_later + 1)]
        for shift in range(1, max_minutes_later + 1):
            df[f'{col_names["low"]}_{shift}'] = df[col_names['low']].shift(-shift)
        logging.info('Done: shift lower_bound')

        df['percent_min_future_rows'] = df.apply(
            lambda row: round((row[col_names_min].min() - row[col_close]) / row[col_close], 4),
            axis=1
        ).round(4)
        logging.info('Done: percent_min_future_rows lower_bound')

        df['signal_buy'] = df.apply(
            lambda row: 1 if row['percent_max_future_rows'] > min_percent_profit and
                             row['percent_min_future_rows'] > min_percent_loss else 0, axis=1,
        )
        logging.info('Done: signal_buy with lower bound')
        col_names_max += col_names_min + ['percent_min_future_rows']
    else:
        df['signal_buy'] = df.apply(lambda row: 1 if row['percent_max_future_rows'] > min_percent_profit else 0, axis=1)
        logging.info('Done: signal_buy without lower bound')

    col_names_max += ['percent_max_future_rows',]

    if need_reverse:
        col_names_max += ['index']

    if remove_extra_cols:
        df.drop(col_names_max, axis=1, inplace=True)

    logging.info(df.head())
    logging.info(df.columns.tolist())
    logging.info(df.groupby(['signal_buy']).count())

    if return_df:
        return df

    logging.info('Saving signal data in {}'.format(OUTPUT_FULL_PATH))
    df.set_index(index_col, inplace=True)
    df.to_csv(
        OUTPUT_FULL_PATH,
        encoding='utf-8',
        index=True
    )


def to_supervised(train, n_out=7): # (x,1,y)
    train = train.tolist()
    for row_num in range(len(train)):
        for row_seq in range(1, n_out):
            ind_adding = row_num-row_seq
            if ind_adding < 0:
                ind_adding = 0
            train[row_num].append(train[ind_adding][0])
    return np.array(train)
