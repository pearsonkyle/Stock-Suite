import os
import time 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alpha_api import AlphaVantage

rate_limit = 5 # requests per minute

if __name__ == '__main__':

    AV = AlphaVantage(api_key=os.environ['ALPHAVANTAGE_API_KEY'])

    TIME_PERIOD = 10*365 # 10 years
    SERIES_TYPE = 'close'

    # set up output directory    
    outdir = 'data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    SYMBOLS = [
        'IBM', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE',
        'AMD', 'TSM', 'QCOM', 'XPEV', 'NIO', 'LAZR', 'TXN', 'NTDOY',
        'CTXR', 'CMPS', 'PFE', 'MRNA', 'ATOS', 'AMC', 'BA', 'CRSR', 
        'X', 'V', 'GOOGL', 'LHX', 'TWLO', 'DIS', 'SBUX', 'F',
    ]

    for SYMBOL in SYMBOLS:

        # define dict of queries to make while respecting rate limit
        queries = {
            'daily': {
                'function': AV.daily,
                'kwargs': { 'symbol': SYMBOL, 'outputsize': 'full' }
            },
            'sma':{
                'function': AV.simple_moving_average,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'series_type': SERIES_TYPE, 'time_period': TIME_PERIOD }
            },
            'ema':{
                'function': AV.exponential_moving_average,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'series_type': SERIES_TYPE, 'time_period': TIME_PERIOD }
            },
            'rsi':{
                'function': AV.relative_strength_index,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'time_period': TIME_PERIOD, 'series_type': SERIES_TYPE }
            },
            'stoch':{
                'function': AV.stochastic_oscillator,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'fastkperiod': 5, 'slowkperiod': 3, 'slowdperiod': 3, 'slowkmatype': 0, 'slowdmatype': 0 }
            },
            'adx':{
                'function': AV.average_directional_movement,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'time_period': TIME_PERIOD }
            },
            'cci':{
                'function': AV.commodity_channel_index,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'time_period': TIME_PERIOD }
            },
            'aroon':{
                'function': AV.aroon,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'time_period': TIME_PERIOD }
            },
            'bbands':{
                'function': AV.bollinger_bands,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily', 'time_period': TIME_PERIOD }
            },
            'obv':{
                'function': AV.on_balance_volume,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily' }
            },
            'ad':{
                'function': AV.chaikin_a_d,
                'kwargs': { 'symbol': SYMBOL, 'interval': 'daily' }
            }
        }

        data = {}

        # make queries
        for key, value in queries.items():
            print('Querying {} for {}'.format(key, SYMBOL))
            data[key] = value['function'](**value['kwargs'])
            time.sleep(60/rate_limit)

        # create csv from data dict
        data_df = pd.DataFrame.from_dict(data['daily']['Time Series (Daily)'], orient='index')
        for key in data.keys():
            if key == 'daily':
                continue
            # key that's not Meta Data
            for subskey in data[key].keys():
                if subskey == 'Meta Data':
                    continue
                df = pd.DataFrame.from_dict(data[key][subskey], orient='index')
                data_df = data_df.join(df)

        remove_these_cols = ['5. adjusted close', '6. volume', '7. dividend amount', '8. split coefficient']
        data_df = data_df.drop(remove_these_cols, axis=1)
        # rename the number from these columns: 1. open 2. high   3. low 4. close
        data_df = data_df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'})
        
        # remove rows with nans
        data_df = data_df.dropna()
        
        # save as csv with symbol as filename
        filename = os.path.join(outdir, '{}.csv'.format(SYMBOL))
        data_df.to_csv(filename)

        # save as pickle with symbol as filename
        # filename = os.path.join(outdir, '{}.pkl'.format(SYMBOL))
        # with open(filename, 'wb') as f:
        #     pickle.dump(data, f)

        time.sleep(60/rate_limit)
