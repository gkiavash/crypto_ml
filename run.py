import logging

import Binance_load_raw_dataset
import Binance_indicators
import nn_train
import backtests
import combine
import utils
from config import Config
import config_test
from binance.client import Client


col_names = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close'
}

current_config = Config(
    SYMBOL='BTCUSDT',
    INTERVAL=Client.KLINE_INTERVAL_15MINUTE,
    RAW_DATASET_START_DATETIME="1 Jan, 2021",
    RAW_DATASET_END_DATETIME="30 Aug, 2021",
    SIGNAL_MAX_MINUTE_LATER=8,
    SIGNAL_MIN_PERCENT_PROFIT=0.01,
    SIGNAL_MIN_PERCENT_LOSS=-0.003,
)

# Binance_load_raw_dataset.get_data(current_config, overwrite=True)

# utils.signal_buy_(
#     df=None,
#     INPUT_DATASET_PATH=current_config.RAW_DATASET_FULL_PATH,
#     OUTPUT_FULL_PATH=current_config.SIGNAL_DATASET_FULL_PATH,
#     max_minutes_later=current_config.SIGNAL_MAX_MINUTE_LATER,
#     min_percent_profit=current_config.SIGNAL_MIN_PERCENT_PROFIT,
#     min_percent_loss=current_config.SIGNAL_MIN_PERCENT_LOSS,
#     col_names=col_names,
#     return_df=False,
#     header_row=0,
#     index_col='unix',
#     need_reverse=False,
#     lower_bound=False,
#     remove_extra_cols=True,
# )

# Binance_indicators.add_indicators_v2(
#     df=None,
#     INPUT_DATASET_PATH=current_config.SIGNAL_DATASET_FULL_PATH,
#     OUTPUT_FULL_PATH=current_config.INDICATORS_DATASET_FULL_PATH,
#     return_df=False,
#     col_names=col_names
# )

# Binance_indicators.df_logging(df=None, INPUT_DATASET_PATH=current_config.INDICATORS_DATASET_FULL_PATH)

# model = nn_train.nn_train(current_config)

# yhat, y_test = backtests.load_predict(current_config=current_config, config_test=config_test)
# nn_train.eval_count_days(yhat, y_test)


# NOTE: COMBINE
config_bigger = Config(
    SYMBOL='BTCUSDT',
    INTERVAL=Client.KLINE_INTERVAL_30MINUTE,
    RAW_DATASET_START_DATETIME="1 Jan, 2021",
    RAW_DATASET_END_DATETIME="30 Aug, 2021",
    SIGNAL_MAX_MINUTE_LATER=8,
    SIGNAL_MIN_PERCENT_PROFIT=0.01,
    SIGNAL_MIN_PERCENT_LOSS=-0.003,
)

# combine.combine(current_config, config_bigger)
