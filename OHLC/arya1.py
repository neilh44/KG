import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_market_data():
    kline_url = "https://fapi.binance.com/fapi/v1/klines"
    kline_params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 168}  # 1 week hourly
    klines = requests.get(kline_url, params=kline_params).json()
    
    return klines

klines = get_market_data()
R = 3438

# Extract data
closes = np.array([float(k[4]) for k in klines])
volumes = np.array([float(k[5]) for k in klines])
n = len(closes)

print(f"Data points: {n}")

# === SIMPLE CELESTIAL FORCES (What Actually Works) ===

# 1. Time cycles (the ONLY predictable force)
hours = np.arange(n)
daily_cycle = np.sin(2 * np.pi * hours / 24)  # 24h cycle
weekly_cycle = np.sin(2 * np.pi * hours / (24 * 7))  # Weekly cycle

# 2. Momentum (trend continuation) 
returns = np.diff(np.log(closes), prepend=0)
momentum = np.array([np.mean(returns[max(0,i-11):i+1]) for i in range(n)])

# 3. Mean reversion strength
ma_24 = np.array([np.mean(closes[max(0,i-23):i+1]) for i in range(n)])
reversion = (closes - ma_24) / ma_24

# === JYA-KOJYA ON NORMALIZED DISTANCES ===
d_momentum = np.tanh(momentum * 50)  # Scale for jya input
d_reversion = np.tanh(reversion * 2)

# Apply Aryabhata's sine/cosine transforms
jya_momentum = R * np.sin(d_momentum)
kojya_momentum = R * np.cos(d_momentum)

jya_reversion = R * np.sin(d_reversion) 
kojya_reversion = R * np.cos(d_reversion)

# Celestial feature matrix
features = np.column_stack([
    daily_cycle,
    weekly_cycle,
    jya_momentum,
    kojya_momentum,
    jya_reversion,
    kojya_reversion
])

# === PREDICT DIRECTION, NOT EXACT PRICE ===
# Target: next hour's return (more predictable than price)
target_returns = np.roll(returns, -1)  # Next hour return
target_returns[-1] = 0  # Can't predict beyond data

# Remove last point (no future return)
features = features[:-1]
target_returns = target_returns[:-1]
n = len(features)

# Train/test split
split = int(0.8 * n)
X_train, X_test = features[:split], features[split:]
y_train, y_test = target_returns[:split], target_returns[split:]

# Simple linear model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n=== Aryabhata's Return Prediction ===")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Overfitting Gap: {train_r2 - test_r2:.6f}")

# Current state
current_return_pred = model.predict(features[-1:].reshape(1, -1))[0]
current_price = closes[-1]
next_price_pred = current_price * (1 + current_return_pred)

print(f"\nCurrent Price: ${current_price:.2f}")
print(f"Predicted Return: {current_return_pred:.6f}")
print(f"Next Hour Target: ${next_price_pred:.2f}")
print(f"Direction: {'📈 UP' if current_return_pred > 0 else '📉 DOWN'}")

# === REALITY CHECK ===
if test_r2 > 0.02:
    print(f"\n✅ Weak but real celestial signal detected")
    confidence = min(test_r2 * 100, 15)  # Cap at 15%
    print(f"Confidence: {confidence:.1f}%")
elif test_r2 > 0:
    print(f"\n⚠️ Extremely weak signal")
else:
    print(f"\n❌ No predictable celestial forces found")
    print(f"Market is pure chaos at this timeframe")

# Force contributions
print(f"\n=== Active Forces ===")
print(f"Daily cycle: {daily_cycle[-1]:.4f}")
print(f"Weekly cycle: {weekly_cycle[-1]:.4f}") 
print(f"Momentum: {d_momentum[-1]:.4f}")
print(f"Reversion: {d_reversion[-1]:.4f}")

print(f"\n💡 To reach R² = 1, you need:")
print(f"   • Higher timeframe (4h/daily)")
print(f"   • External data (options, on-chain)")
print(f"   • Or accept that 5min moves are mostly random")