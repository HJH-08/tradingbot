def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def compute_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    exp1 = data.ewm(span=fastperiod, adjust=False).mean()
    exp2 = data.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd - signal