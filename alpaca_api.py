import asyncio
import websockets
import json
import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import technical

# Alpaca API credentials
API_KEY = "PKS0ZKV4QX27O16UK7KZ"
API_SECRET = "XogrtL7YG4OAUWUWMfIwhrWyBEbaMPC5YzuzsySK"
BASE_URL = "https://paper-api.alpaca.markets/v2"
STOCKS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
stream_url = "wss://stream.data.alpaca.markets/v2/iex"

# Initialize a dictionary to store data for each stock
data = defaultdict(lambda: pd.DataFrame(columns=['timestamp', 'price']))

def fetch_historical_data(symbol, start_date, end_date, timeframe='1Day'):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }
    params = {
        'start': start_date,
        'end': end_date,
        'timeframe': timeframe,
        'limit': 1000
    }
    print(f"Requesting historical data for {symbol} from {start_date} to {end_date} with timeframe {timeframe}")
    response = requests.get(url, headers=headers, params=params)
    print(f"API Response for {symbol}: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if 'bars' in data:
            df = pd.DataFrame(data['bars'])
            df['timestamp'] = pd.to_datetime(df['t'])
            df.set_index('timestamp', inplace=True)
            print(f"Fetched historical data for {symbol}:")
            print(df.head())
            return df[['o', 'h', 'l', 'c', 'v']]
        else:
            print(f"No historical data found in the response for {symbol}: {data}")
            return pd.DataFrame()
    else:
        print(f"Failed to fetch data for {symbol}: {response.text}")
        return pd.DataFrame()

# Calculate date range
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

#     API Response for AAPL: 403
# Failed to fetch data for AAPL: {"message":"subscription does not permit querying recent SIP data"}


# Fetch historical data for all stocks
historical_data = {}

for stock in STOCKS:
    historical_data[stock] = fetch_historical_data(stock, start_date, end_date)

# Initialize DataFrame with historical data
for stock in STOCKS:
    if not historical_data[stock].empty:
        data[stock] = historical_data[stock][['c']].rename(columns={'c': 'price'}).reset_index()

async def handle_ws_message():
    async with websockets.connect(stream_url) as websocket:
        auth_data = {
            'action': 'auth',
            'key': API_KEY,
            'secret': API_SECRET
        }

        await websocket.send(json.dumps(auth_data))
        print(f">>> Sent auth: {auth_data}")

        auth_response = await websocket.recv()
        auth_response_data = json.loads(auth_response)
        print(f"<<< Received auth response: {auth_response_data}")

        listen_message = {
            'action': 'subscribe',
            'trades': STOCKS
        }

        await websocket.send(json.dumps(listen_message))
        print(f">>> Sent subscribe: {listen_message}")

        while True:
            response = await websocket.recv()
            trade_data = json.loads(response)
            print(f"<<< Received: {trade_data}")
            if isinstance(trade_data, list):
                for trade in trade_data:
                    if trade['T'] == 't':
                        symbol = trade['S']
                        timestamp = pd.to_datetime(trade['t'])
                        price = trade['p']

                        # Append new trade data to the DataFrame
                        new_row = pd.DataFrame([[timestamp, price]], columns=['timestamp', 'price'])
                        data[symbol] = pd.concat([data[symbol], new_row])
                        data[symbol].reset_index(drop=True, inplace=True)

                        # Calculate indicators
                        calculate_indicators(symbol)

def calculate_indicators(symbol):
    df = data[symbol]

    # Calculate moving average (MA)
    df['MA20'] = df['price'].rolling(window=20).mean()

    # Calculate relative strength index (RSI)
    df['RSI'] = technical.compute_rsi(df['price'])

    # Calculate MACD and Signal line
    df['MACD'], df['Signal'] = technical.alpaca_compute_macd(df['price'])

    # Print latest indicators for demonstration purposes
    if not df.empty:
        latest_data = df.iloc[-1]
        print(f"Symbol: {symbol}, Timestamp: {latest_data['timestamp']}, Price: {latest_data['price']}, MA20: {latest_data['MA20']}, RSI: {latest_data['RSI']}, MACD: {latest_data['MACD']}, Signal: {latest_data['Signal']}")

# Run the WebSocket handling function
loop = asyncio.get_event_loop()
loop.run_until_complete(handle_ws_message())
