import logging

import pandas as pd


def combine(config_smaller_interval, config_bigger_interval):
    df_smaller_interval = pd.read_csv(config_smaller_interval.INDICATORS_DATASET_FULL_PATH, header=0, delimiter=",")
    df_smaller_interval.set_index('unix', inplace=True)

    df_bigger_interval = pd.read_csv(config_bigger_interval.INDICATORS_DATASET_FULL_PATH, header=0, delimiter=",")
    df_bigger_interval.set_index('unix', inplace=True)
    needed_cols = ['bb_bbp', 'rsi_6_']
    needed_cols_new = {i: f'{i}_{config_bigger_interval.INTERVAL}' for i in needed_cols}
    df_bigger_interval = df_bigger_interval[needed_cols]
    df_bigger_interval.rename(needed_cols_new, axis=1, inplace=True)

    df = pd.concat([df_smaller_interval, df_bigger_interval], axis=1)

    for i in needed_cols_new.keys():
        df[f'{i}_*_{needed_cols_new[i]}'] = (df[i]*df[needed_cols_new[i]]).round(3)

    df.fillna(method='ffill', inplace=True)
    df.to_csv('combined.csv', encoding='utf-8', index=True)

# plotting
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# from run import current_config
#
# df = pd.read_csv(current_config.INDICATORS_DATASET_FULL_PATH, header=0, delimiter=",")
# with sns.axes_style('white'):
#     sns.relplot(x="rsi_6_count", y="rsi_6_", hue="signal_buy", data=df)
#     plt.show()
