import os
import time
import requests
import functools
import numpy as np
import pandas as pd

class AlphaVantage():
    # https://www.alphavantage.co/documentation/

    def __init__(self, api_url='https://www.alphavantage.co/query?', api_key=None):
        self.api_url = api_url
        self.api_key = api_key

    def get_request(endpoint=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # send request and add api key
                kwargs['apikey'] = self.api_key
                r = requests.get(os.path.join(self.api_url, endpoint), params=kwargs)
                # catch errors and return json error
                if r.status_code == 200:
                    return r.json()
                else:
                    return { 'error': r.status_code }
            return wrapper
        return decorator

    @get_request(endpoint='query')
    def query(self, function='TIME_SERIES_DAILY_ADJUSTED', symbol='IBM', outputsize='full'):
        """ returns open, high, low, close, adjusted close, volume, dividend amount, and split coefficient """
        pass

    def daily(self, symbol='IBM', outputsize='full', function='TIME_SERIES_DAILY_ADJUSTED'):
        return self.query(function=function, symbol=symbol, outputsize=outputsize)

    def weekly(self, symbol='IBM', function='TIME_SERIES_WEEKLY'):
        return self.query(function=function, symbol=symbol, outputsize=outputsize)

    def quote(self, symbol='IBM', function='GLOBAL_QUOTE'):
        """
        A lightweight alternative to the time series APIs, this service returns the latest price and volume information for a ticker of your choice.

        {'Global Quote': {'01. symbol': 'IBM',
            '02. open': '133.2350',
            '03. high': '133.9000',
            '04. low': '131.5500',
            '05. price': '132.1600',
            '06. volume': '3508083',
            '07. latest trading day': '2023-07-06',
            '08. previous close': '134.2400',
            '09. change': '-2.0800',
            '10. change percent': '-1.5495%'}}
        """
        return self.query(function=function, symbol=symbol)

    def top_gainers_losers(self, function='TOP_GAINERS_LOSERS'):
        """
        This endpoint returns the top 20 gainers, losers, and the most active traded tickers in the US market.
        
        dict_keys(['metadata', 'last_updated', 'top_gainers', 'top_losers', 'most_actively_traded'])

        {'ticker': 'META',
        'price': '291.99',
        'change_amount': '-2.38',
        'change_percentage': '-0.8085%',
        'volume': '47356401'}
        """
        return self.query(function=function)
    
    # This API returns the global price of copper in monthly, quarterly, and annual horizons.
    def copper(self, function='COPPER', interval='monthly'):
        """
        dict_keys(['name', 'interval', 'unit', 'data'])

        {'date': '2017-01-01', 'value': '5754.55952380952'},
        """
        return self.query(function=function, interval=interval)

    # NEWS_SENTIMENT
    def news_sentiment(self, function='NEWS_SENTIMENT', tickers='IBM'):
        """
        Get live and historical market news & sentiment data

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        tickers : str
            A comma-separated string of ticker symbols (required)

        topics : str
            A comma-separated string of topics to filter the news by (optional)
            Examples: blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets,
            economy_fiscal, economy_monetary, economy_macro, energy_transportation, finance,
            life_sciences, manufacturing, real_estate, retail_wholesale, technology

        time_from and time_to : str
            A date in the format 20220410T0130 (optional)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['items', 'sentiment_score_definition', 'relevance_score_definition', 'feed'])

            feed-keys: ['title', 'url', 'time_published', 'authors', 'summary', 'banner_image', 'source', 
            'category_within_source', 'source_domain', 'topics', 'overall_sentiment_score', 
            'overall_sentiment_label', 'ticker_sentiment']
        """
        return self.query(function=function, tickers=tickers)

    def simple_moving_average(self, function='SMA', symbol='IBM', interval='daily', series_type='close', time_period=60):
        """
        This API returns the simple moving average (SMA) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        series_type : str
            The desired price type in the time series. Four types are supported:
            close, open, high, low (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: SMA'])
            example: '2019-07-25': {'SMA': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, series_type=series_type, time_period=time_period)

    def exponential_moving_average(self, function='EMA', symbol='IBM', interval='daily', series_type='close', time_period=60):
        """
        This API returns the exponential moving average (EMA) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        series_type : str
            The desired price type in the time series. Four types are supported:
            close, open, high, low (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: EMA'])
            example: '2019-07-25': {'EMA': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, series_type=series_type, time_period=time_period)

    def stochastic_oscillator(self, function='STOCH', symbol='IBM', interval='daily', fastkperiod=5, slowkperiod=3, slowdperiod=3, slowkmatype=0, slowdmatype=0):
        """
        This API returns the stochastic oscillator (STOCH) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        fastkperiod : int
            The time period of the fastk moving average. Positive integers are accepted (required)

        slowkperiod : int
            The time period of the slowk moving average. Positive integers are accepted (required)

        slowdperiod : int
            The time period of the slowd moving average. Positive integers are accepted (required)

        slowkmatype : int
            The type of moving average to be used for slowk. 
            Four types are supported: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 
            6=KAMA, 7=MAMA, 8=T3 (required)

        slowdmatype : int
            The type of moving average to be used for slowd. 
            Four types are supported: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 
            6=KAMA, 7=MAMA, 8=T3 (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: STOCH'])
            example: '2019-07-25': {'SlowK': '107.8685', 'SlowD': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, fastkperiod=fastkperiod, 
                          slowkperiod=slowkperiod, slowdperiod=slowdperiod, slowkmatype=slowkmatype, slowdmatype=slowdmatype)

    def relative_strength_index(self, function='RSI', symbol='IBM', interval='daily', time_period=60, series_type='close'):
        """
        This API returns the relative strength index (RSI) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        series_type : str
            The desired price type in the time series. Four types are supported:
            close, open, high, low (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: RSI'])
            example: '2019-07-25': {'RSI': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    def average_directional_movement(self, function='ADX', symbol='IBM', interval='daily', time_period=60):
        """
        This API returns the average directional movement (ADX) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: ADX'])
            example: '2019-07-25': {'ADX': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, time_period=time_period)

    def commodity_channel_index(self, function='CCI', symbol='IBM', interval='daily', time_period=60):
        """
        This API returns the commodity channel index (CCI) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: CCI'])
            example: '2019-07-25': {'CCI': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, time_period=time_period)

    def aroon(self, function='AROON', symbol='IBM', interval='daily', time_period=60):
        """
        This API returns the aroon (AROON) values. Tracks the number of periods since the most recent n-day high and low.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: AROON'])
            example: '2019-07-25': {'Aroon Up': '107.8685', 'Aroon Down': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, time_period=time_period)

    def bollinger_bands(self, function='BBANDS', symbol='IBM', interval='daily', time_period=60, series_type='close'):
        """
        This API returns the bollinger bands (BBANDS) values.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly (required)

        time_period : int
            The number of data points used to calculate each moving average value. 
            Positive integers are accepted (required)

        series_type : str
            The desired price type in the time series. Four types are supported:
            close, open, high, low (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: BBANDS'])
            example: '2019-07-25': {'Real Upper Band': '107.8685', 'Real Middle Band': '107.8685', 'Real Lower Band': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval, time_period=time_period, series_type=series_type)

    def on_balance_volume(self, function='OBV', symbol='IBM', interval='daily'):
        """
        This API returns the on balance volume (OBV) values. Tracks the cumulative sum of volume.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: daily, weekly, monthly (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: OBV'])
            example: '2019-07-25': {'OBV': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval)

    def chaikin_a_d(self, function='AD', symbol='IBM', interval='daily'):
        """
        This API returns the chaikin a/d (AD) values. Tracks the cumulative sum of volume.

        Parameters
        ----------
        function : str
            The API function to call (required)
        
        symbol : str
            A ticker symbol (required)

        interval : str
            The time interval between two consecutive data points in the time series. 
            The following values are supported: daily, weekly, monthly (required)

        Returns
        -------
        dict
            A dictionary containing the results of the query
            dict_keys(['Meta Data', 'Technical Analysis: AD'])
            example: '2019-07-25': {'Chaikin A/D': '107.8685'},
        """
        return self.query(function=function, symbol=symbol, interval=interval)

if __name__ == '__main__':

    AV = AlphaVantage(api_key=os.environ['ALPHAVANTAGE_API_KEY'])

    #daily = AV.daily(symbol='IBM', outputsize='compact')
    #weekly = AV.weekly(symbol='IBM')
    #quote = AV.quote(symbol='IBM')
    #tickers = AV.top_gainers_losers()
    #copper = AV.copper(interval='monthly')
    #news = AV.news_sentiment(tickers='IBM')
    #sma = AV.simple_moving_average(symbol='IBM', interval='daily', series_type='open', time_period=60)
    #ema = AV.exponential_moving_average(symbol='IBM', interval='daily', series_type='open', time_period=60)
    #stoch = AV.stochastic_oscillator(symbol='IBM', interval='daily', fastkperiod=5, slowkperiod=3, slowdperiod=3, slowkmatype=0, slowdmatype=0)
    #rsi = AV.relative_strength_index(symbol='IBM', interval='daily', time_period=60, series_type='open')
    #adx = AV.average_directional_movement(symbol='IBM', interval='daily', time_period=60)
    #cci = AV.commodity_channel_index(symbol='IBM', interval='daily', time_period=60)
    #aroon = AV.aroon(symbol='IBM', interval='daily', time_period=60)
    #bbands = AV.bollinger_bands(symbol='IBM', interval='daily', time_period=60)
    #obv = AV.on_balance_volume(symbol='IBM', interval='daily')
    ad = AV.chaikin_a_d(symbol='IBM', interval='daily')
