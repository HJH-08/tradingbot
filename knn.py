import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch data from Yahoo Finance
ticker = 'SPY'  # Example ticker
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Feature Engineering
data['Open_Close'] = data['Open'] / data['Close']
data['High_Low'] = data['High'] / data['Low']
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
data['SMA'] = SMAIndicator(data['Close'], window=20).sma_indicator()
data['MACD'] = MACD(data['Close']).macd()

# Lagged features
data['Lag1_RSI'] = data['RSI'].shift(1)
data['Lag1_SMA'] = data['SMA'].shift(1)
data['Lag1_MACD'] = data['MACD'].shift(1)

# Drop rows with NaN values
data = data.dropna()

# Define target variable: if the close price is higher than the previous close, label as 1 (up), else 0 (down)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# Select features
features = ['Open_Close', 'High_Low', 'RSI', 'SMA', 'MACD', 'Lag1_RSI', 'Lag1_SMA', 'Lag1_MACD']
X = data[features]
y = data['Target']

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 31)}
knn = KNeighborsClassifier(metric='euclidean')
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best k value
best_k = grid_search.best_params_['n_neighbors']
print(f'Best k: {best_k}')

# Train the k-NN model with the best k
knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualization of accuracy for different k values
results = pd.DataFrame(grid_search.cv_results_)
plt.plot(results['param_n_neighbors'], results['mean_test_score'])
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validated Accuracy')
plt.title('k-NN Hyperparameter Tuning')
plt.grid(True)
plt.show()


