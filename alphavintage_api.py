import requests
import pandas as pd
from datetime import datetime, timedelta

# Alpha Vantage API credentials
ALPHA_VANTAGE_API_KEY = '6RZTXKD3H4L7ST8X6RZTXKD3H4L7ST8X'
STOCKS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
ALPHA_VANTAGE_URL = 'https://www.alphavantage.co/query'

# Function to fetch intraday data from Alpha Vantage
def fetch_intraday_data(symbol, interval='15min', outputsize='compact'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': outputsize
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    data = response.json()
    
    if f'Time Series ({interval})' in data:
        df = pd.DataFrame.from_dict(data[f'Time Series ({interval})'], orient='index', dtype='float')
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print(f"Failed to fetch data for {symbol}: {data.get('Note', 'Unknown error')}")
        return pd.DataFrame()

# Function to fetch technical indicators from Alpha Vantage
def fetch_technical_indicator(symbol, indicator, interval='daily', time_period=20):
    params = {
        'function': indicator,
        'symbol': symbol,
        'interval': interval,
        'time_period': time_period,
        'series_type': 'close',
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    data = response.json()
    if 'Technical Analysis' in data:
        key = list(data['Technical Analysis'].keys())[0]
        df = pd.DataFrame.from_dict(data['Technical Analysis'][key], orient='index', dtype='float')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print(f"Failed to fetch indicator {indicator} for {symbol}: {data.get('Note', 'Unknown error')}")
        return pd.DataFrame()

# Function to fetch STOCH indicator
def fetch_stoch(symbol, interval='daily', fastkperiod=5, slowkperiod=3, slowdperiod=3):
    params = {
        'function': 'STOCH',
        'symbol': symbol,
        'interval': interval,
        'fastkperiod': fastkperiod,
        'slowkperiod': slowkperiod,
        'slowdperiod': slowdperiod,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    data = response.json()
    if 'Technical Analysis: STOCH' in data:
        df = pd.DataFrame.from_dict(data['Technical Analysis: STOCH'], orient='index', dtype='float')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print(f"Failed to fetch STOCH for {symbol}: {data.get('Note', 'Unknown error')}")
        return pd.DataFrame()

# Function to calculate all technical indicators for a stock
def calculate_indicators(symbol):
    df = fetch_intraday_data(symbol)
    if not df.empty:
        # Fetch SMA
        sma = fetch_technical_indicator(symbol, 'SMA', interval='15min', time_period=20)
        df = df.join(sma.rename(columns={'SMA': 'SMA'}))
        
        # Fetch EMA
        ema = fetch_technical_indicator(symbol, 'EMA', interval='15min', time_period=20)
        df = df.join(ema.rename(columns={'EMA': 'EMA'}))
        
        # Fetch RSI
        rsi = fetch_technical_indicator(symbol, 'RSI', interval='15min', time_period=14)
        df = df.join(rsi.rename(columns={'RSI': 'RSI'}))
        
        # Fetch Bollinger Bands
        bbands = fetch_technical_indicator(symbol, 'BBANDS', interval='15min')
        df = df.join(bbands.rename(columns={'Real Lower Band': 'BB_lower', 'Real Upper Band': 'BB_upper', 'Real Middle Band': 'BB_middle'}))
        
        # Fetch STOCH
        stoch = fetch_stoch(symbol, interval='15min')
        df = df.join(stoch.rename(columns={'SlowK': 'STOCH_SlowK', 'SlowD': 'STOCH_SlowD'}))
        
        print(f"Calculated indicators for {symbol}:")
        print(df.tail())
    else:
        print(f"No data to calculate indicators for {symbol}")

# Calculate indicators for all stocks
for stock in STOCKS:
    calculate_indicators(stock)
