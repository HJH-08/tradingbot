import yfinance as yf

# Fetch data for AAPL
aapl = yf.Ticker("AAPL")

# Get historical market data
data = aapl.history(period="max")

# Display the data
print(data)
