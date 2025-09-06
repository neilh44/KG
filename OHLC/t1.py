import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fetch BTCUSDT 1-minute data from Binance Futures
url = "https://fapi.binance.com/fapi/v1/klines"
params = {
    'symbol': 'BTCUSDT',
    'interval': '1m',
    'limit': 1000
}

response = requests.get(url, params=params)
data = response.json()

# Extract close prices
closes = np.array([float(candle[4]) for candle in data])

# Prepare data for regression
X = np.arange(len(closes)).reshape(-1, 1)  # Time index
y = closes

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate Pythagorean components
y_pred = model.predict(X)
y_mean = np.mean(y)

TSS = np.sum((y - y_mean)**2)           # Total Sum of Squares
ESS = np.sum((y_pred - y_mean)**2)      # Explained Sum of Squares  
RSS = np.sum((y - y_pred)**2)           # Residual Sum of Squares

# Verify Pythagorean theorem
print(f"TSS: {TSS:.2f}")
print(f"ESS: {ESS:.2f}")
print(f"RSS: {RSS:.2f}")
print(f"ESS + RSS: {ESS + RSS:.2f}")
print(f"Pythagorean validation: {abs(TSS - (ESS + RSS)) < 1e-6}")

# Model statistics
r2 = r2_score(y, y_pred)
print(f"\nR²: {r2:.6f}")
print(f"Slope: {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.2f}")

# Prediction equation
print(f"\nEquation: y = {model.coef_[0]:.6f} * x + {model.intercept_:.2f}")

# Predict next close price
next_x = len(closes)
next_close = model.predict([[next_x]])[0]
print(f"\nCurrent close: ${closes[-1]:.2f}")
print(f"Predicted next close: ${next_close:.2f}")
print(f"Change: ${next_close - closes[-1]:.2f}")