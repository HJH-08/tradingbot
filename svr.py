import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime

# Fetch data for AAPL
aapl = yf.Ticker("AAPL")
data = aapl.history(period="max")

def plot_svr_regression_with_bands(days):
    # Limit data to the last 'days' number of days
    recent_data = data.tail(days)
    
    # Prepare data for regression
    recent_data['Date'] = recent_data.index.map(datetime.toordinal)
    X = recent_data['Date'].values.reshape(-1, 1)
    y = recent_data['Close'].values
    
    # Create and train the SVR model
    model = SVR(kernel='rbf', C=100, gamma='auto')
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate the residuals
    residuals = y - y_pred
    
    # Calculate the standard deviation of the residuals
    std_dev = np.std(residuals)
    
    # Calculate the upper and lower bands (2 standard deviations away)
    upper_band = y_pred + 2 * std_dev
    lower_band = y_pred - 2 * std_dev
    
    # Calculate R^2 value
    r2 = model.score(X, y)
    
    # Plot original data, the regression line, and the bands
    plt.figure(figsize=(14, 7))
    plt.plot(recent_data.index, y, label='Actual Prices', color='blue')
    plt.plot(recent_data.index, y_pred, label='SVR Regression', color='red')
    plt.plot(recent_data.index, upper_band, label='Upper Band (2 STD)', color='green', linestyle='--')
    plt.plot(recent_data.index, lower_band, label='Lower Band (2 STD)', color='green', linestyle='--')
    plt.fill_between(recent_data.index, lower_band, upper_band, color='green', alpha=0.1)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'SVR Regression of AAPL Close Prices with Bands (2 STD) for the Last {days} Days\nR^2 = {r2:.2f}')
    plt.legend()
    plt.show()

# Example usage: Plot SVR regression with bands for the last 180 days
plot_svr_regression_with_bands(180)
