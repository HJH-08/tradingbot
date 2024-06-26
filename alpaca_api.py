import asyncio
import websockets
import json
import alpaca_trade_api as tradeapi

API_KEY = "PKS0ZKV4QX27O16UK7KZ"
API_SECRET = "XogrtL7YG4OAUWUWMfIwhrWyBEbaMPC5YzuzsySK"
BASE_URL = "https://paper-api.alpaca.markets/v2"
STOCKS = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN']
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

# Correct WebSocket URL for Alpaca paper trading data stream
#stream_url = "wss://stream.data.sandbox.alpaca.markets/v2/sip"
stream_url = "wss://stream.data.alpaca.markets/v2/test"

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
            data = json.loads(response)
            print(f"<<< Received: {data}")

# Run the WebSocket handling function
loop = asyncio.get_event_loop()
loop.run_until_complete(handle_ws_message())
