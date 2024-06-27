import numpy as np
import quandl
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import joblib
import os
import technical
import matplotlib.pyplot as plt

# Define the functions to compute indicators
# (Add the functions here from the previous section)

# LONG TERM IDEA
# def exponential_smoothing(series, alpha):
#     result = [series.iloc[0]]  # first value is same as series
#     for n in range(1, len(series)):
#         result.append(alpha * series.iloc[n] + (1 - alpha) * result[n-1])
#     return result

def train_and_save_model(stock_symbol, api_key, base_dir='models'):
    # Get the data from Quandl
    quandl.ApiConfig.api_key = api_key
    data = quandl.get(f'WIKI/{stock_symbol}')
    
    # Data Preprocessing
    data.dropna(inplace=True)
    print(data)
    # Apply exponential smoothing to the adjusted close prices
    alpha = 0.1  # smoothing factor
    #data['Smoothed_Adj_Close'] = exponential_smoothing(data['Adj. Close'], alpha)



    # Feature Engineering
    data['RSI'] = technical.compute_rsi(data['Adj. Close'])
    data['Stochastic_Oscillator'] = technical.compute_stochastic_oscillator(data)
    data['MACD'] = technical.compute_macd(data['Adj. Close'])
    data['Williams_R'] = technical.compute_williams_r(data)
    data['ROC'] = technical.compute_roc(data['Adj. Close'])
    data['OBV'] = technical.compute_obv(data)
    data['Future Price'] = data['Adj. Close'].shift(-1)
    data.dropna(inplace=True)

    # Prepare Data for Machine Learning
    features = ['RSI', 'Stochastic_Oscillator', 'MACD', 'Williams_R', 'ROC', 'OBV']
    X = data[features]
    y = data['Future Price']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'bootstrap': [True, False]
    }

        # Plot the smoothed adjusted close prices
    # plt.figure(figsize=(14, 7))
    # plt.plot(data['Adj. Close'])
    # plt.title('Smoothed Adjusted Close Prices')
    # plt.show()

    # # Plot the target variable distribution
    # plt.figure(figsize=(14, 7))
    # plt.hist(data['Future Price'], bins=50)
    # plt.title('Future Price Distribution')
    # plt.show()

    # Print the first few rows of the data with the indicators
    print(data[['RSI', 'Stochastic_Oscillator', 'MACD', 'Williams_R', 'ROC', 'OBV']].head())


    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Perform Grid Search with 5-Fold Cross-Validation
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    # grid_search.fit(X, y)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X, y)

    # Save the best model to disk
    best_model = random_search.best_estimator_
    model_filename = os.path.join(base_dir, f'{stock_symbol}_model.pkl')
    joblib.dump(best_model, model_filename)
    print(f'Best model saved to {model_filename}')
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best MSE: {-random_search.best_score_}')  # Convert negative MSE to positive

def load_model(stock_symbol, base_dir='models'):
    model_filename = os.path.join(base_dir, f'{stock_symbol}_model.pkl')
    
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        print(f'Model for {stock_symbol} loaded from {model_filename}')
        return model
    else:
        print(f'Model file for {stock_symbol} not found. Train and save the model first.')
        return None

api_key = "g-cGHeLRESQ5Qx8gBUkD"
stocks = ['GOOG']

for stock in stocks:
    model = load_model(stock)
    if not model:
        train_and_save_model(stock, api_key)

# Example usage after loading the model
# Assume X_test_GOOG is the test data for GOOG
# X_test_GOOG should be preprocessed similarly to the training data
# y_pred = model.predict(X_test_GOOG)
# Evaluate and/or use the predictions
