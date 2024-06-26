import quandl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import technical

def train_and_save_model(stock_symbol, api_key, base_dir='models'):
    # Get the data from Quandl
    quandl.ApiConfig.api_key = api_key
    data = quandl.get(f'WIKI/{stock_symbol}')
    print(data)
    # Data Preprocessing
    data.dropna(inplace=True)

    # Feature Engineering
    data['MA10'] = data['Adj. Close'].rolling(window=10).mean()
    data['MA50'] = data['Adj. Close'].rolling(window=50).mean()
    data['Returns'] = data['Adj. Close'].pct_change()
    data['RSI'] = technical.compute_rsi(data['Adj. Close'])
    data['MACD'] = technical.compute_macd(data['Adj. Close'])
    data['Future Price'] = data['Adj. Close'].shift(-1)
    data.dropna(inplace=True)

    # Prepare Data for Machine Learning
    features = ['Adj. Close', 'MA10', 'MA50', 'Returns', 'RSI', 'MACD']
    X = data[features]
    y = data['Future Price']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train multiple models with different random states and select the best one
    best_mse = float('inf')
    best_model = None
    best_random_state = None

    for random_state in range(100):
        # Split the data into training and testing sets with different random states
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        
        # Train the Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate the Model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print performance metrics
        print(f'Random State: {random_state}, Mean Squared Error: {mse}, R^2 Score: {r2}')
        
        # Save the model if it has the best performance so far
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_random_state = random_state

    # Save the best model to disk
    model_filename = os.path.join(base_dir, f'{stock_symbol}_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f'Best model saved to {model_filename} with random state {best_random_state} and MSE {best_mse}')

api_key = "g-cGHeLRESQ5Qx8gBUkD"
# stocks = ['AAPL', 'MSFT', 'GOOG', 'FB']
stocks = ['GOOG']

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


