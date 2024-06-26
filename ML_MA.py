import quandl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_and_save_model(stock_symbol, api_key, base_dir='models'):
    # Get the data from Quandl
    quandl.ApiConfig.api_key = api_key
    data = quandl.get(f'WIKI/{stock_symbol}')

    # Data Preprocessing
    data.dropna(inplace=True)

    # Feature Engineering
    data['MA10'] = data['Adj. Close'].rolling(window=10).mean()
    data['MA50'] = data['Adj. Close'].rolling(window=50).mean()
    data['Returns'] = data['Adj. Close'].pct_change()
    data['Future Price'] = data['Adj. Close'].shift(-1)
    data.dropna(inplace=True)

    # Prepare Data for Machine Learning
    X = data[['Adj. Close', 'MA10', 'MA50', 'Returns']]
    y = data['Future Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to disk
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    model_filename = os.path.join(base_dir, f'{stock_symbol}_model.pkl')
    joblib.dump(model, model_filename)
    print(f'Model for {stock_symbol} saved to {model_filename}')

    # Evaluate the Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{stock_symbol} - Mean Squared Error: {mse}')
    print(f'{stock_symbol} - R^2 Score: {r2}')

    return model

api_key = "g-cGHeLRESQ5Qx8gBUkD"
#stocks = ['AAPL', 'MSFT', 'GOOG', 'FB']
stocks = ['AAPL']

def load_model(stock_symbol, base_dir='models'):
    model_filename = os.path.join(base_dir, f'{stock_symbol}_model.pkl')
    
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        print(f'Model for {stock_symbol} loaded from {model_filename}')
        return model
    else:
        print(f'Model file for {stock_symbol} not found. Train and save the model first.')
        return None

for stock in stocks:
    model = load_model(stock)
    if not model:
        train_and_save_model(stock, api_key)


# Assume X_test_AAPL is the test data for AAPL
# Make predictions
#y_pred = model.predict(X_test_AAPL)
# Evaluate and/or use the predictions


