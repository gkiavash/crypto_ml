import os
import logging
import numpy as np
import pandas as pd

from binance.client import Client

import config as current_config

logging.basicConfig(
    filename='Binance_api.log',
    level=logging.INFO,
    datefmt='%d-%b-%y %H:%M:%S'
)

api_key = 'm7sioX7ArO9qtdtPuOH6VRIkqiz2Dqki4A8FUqF7zheLvgogdux1UZ9KoHGNj3bT'
api_secret = 'oqzZkc4nlvq77uUfs5L08sWYKxlpk32MmJ0su3yLLkARx2YvDGS4CAHQ3TFvqGOv'
proxies = {
    'http': 'http://10.10.1.10:3128',
    'https': 'http://10.10.1.10:1080'
}
client = Client(api_key, api_secret,)


klines = client.get_historical_klines(
    current_config.SYMBOL,
    current_config.INTERVAL,
    current_config.RAW_DATASET_START_DATETIME,
    current_config.RAW_DATASET_END_DATETIME
)
klines_ = np.array(klines)
df = pd.DataFrame(
    {
        'unix': klines_[:, 0],
        'open': klines_[:, 1],
        'high': klines_[:, 2],
        'low': klines_[:, 3],
        'close': klines_[:, 4],
    }
)
df['symbol'] = current_config.SYMBOL

if not os.path.exists(current_config.RAW_DATASET_DIRECTORY):
    os.makedirs(current_config.RAW_DATASET_DIRECTORY)

df.to_csv(current_config.RAW_DATASET_FULL_PATH, encoding='utf-8', index=False)
