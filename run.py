import Binance_load_raw_dataset
import Binance_indicators
import utils
import config as current_config


# Binance_load_raw_dataset.get_data()

col_names = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close'
}

# utils.signal_buy_(
#     df=None,
#     INPUT_DATASET_PATH=current_config.RAW_DATASET_FULL_PATH,
#     OUTPUT_FULL_PATH=current_config.SIGNAL_DATASET_FULL_PATH,
#     max_minutes_later=current_config.SIGNAL_MAX_MINUTE_LATER,
#     min_percent_profit=current_config.SIGNAL_MIN_PERCENT_PROFIT,
#     col_names=col_names,
#     return_df=False,
#     header_row=0,
#     index_col='unix',
#     need_reverse=False,
#     lower_bound=False
# )

Binance_indicators.add_indicators(
    df=None,
    INPUT_DATASET_PATH=current_config.SIGNAL_DATASET_FULL_PATH,
    OUTPUT_FULL_PATH=current_config.INDICATORS_DATASET_FULL_PATH,
    return_df=False,
    col_names=col_names
)
