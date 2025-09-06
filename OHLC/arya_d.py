import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_market_data():
    # Daily OHLCV for 1 year
    kline_url = "https://fapi.binance.com/fapi/v1/klines"
    kline_params = {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 365}
    klines = requests.get(kline_url, params=kline_params).json()
    
    return klines

klines = get_market_data()
R = 3438

# Extract data
closes = np.array([float(k[4]) for k in klines])
volumes = np.array([float(k[5]) for k in klines])
highs = np.array([float(k[2]) for k in klines])
lows = np.array([float(k[3]) for k in klines])
n = len(closes)

print(f"Daily data points: {n}")

# === DAILY CELESTIAL FORCES (Where Patterns Exist) ===

# 1. Long-term trend force (50-day MA)
ma_50 = np.array([np.mean(closes[max(0,i-49):i+1]) for i in range(n)])
trend_strength = (closes - ma_50) / ma_50
d_trend = np.tanh(trend_strength * 3)

# 2. Mean reversion force (20-day)
ma_20 = np.array([np.mean(closes[max(0,i-19):i+1]) for i in range(n)])
reversion_force = (closes - ma_20) / ma_20
d_reversion = np.tanh(reversion_force * 5)

# 3. Volume surge force (institutional activity)
vol_ma_20 = np.array([np.mean(volumes[max(0,i-19):i+1]) for i in range(n)])
volume_surge = volumes / (vol_ma_20 + 1e-8)
d_volume = np.tanh((volume_surge - 1) * 2)

# 4. Volatility expansion/contraction
daily_returns = np.diff(np.log(closes), prepend=0)
volatility = np.abs(daily_returns)
vol_ma_10 = np.array([np.mean(volatility[max(0,i-9):i+1]) for i in range(n)])
vol_regime = volatility / (vol_ma_10 + 1e-8)
d_volatility = np.tanh((vol_regime - 1) * 3)

# 5. Range position (support/resistance structure)
range_20_high = np.array([np.max(highs[max(0,i-19):i+1]) for i in range(n)])
range_20_low = np.array([np.min(lows[max(0,i-19):i+1]) for i in range(n)])
range_position = (closes - range_20_low) / (range_20_high - range_20_low + 1e-8)
d_structure = np.tanh(4 * (range_position - 0.5))

# 6. Momentum persistence (trend continuation)
momentum_5 = np.array([np.mean(daily_returns[max(0,i-4):i+1]) for i in range(n)])
d_momentum = np.tanh(momentum_5 * 20)

# === ARYABHATA'S JYA-KOJYA TRANSFORMS ===
celestial_forces = [d_trend, d_reversion, d_volume, d_volatility, d_structure, d_momentum]
celestial_coords = []

for force in celestial_forces:
    jya = R * np.sin(force)
    kojya = R * np.cos(force) 
    celestial_coords.extend([jya, kojya])

# Stack into feature matrix
features = np.column_stack(celestial_coords)

# === TARGET: NEXT DAY'S RETURN ===
log_returns = np.diff(np.log(closes))  # Length: n-1
target_returns = np.roll(log_returns, -1)  # Shift left
target_returns[-1] = 0  # No future data

# Align arrays: make both same length
min_length = min(len(features), len(target_returns))
features = features[:min_length]
target_returns = target_returns[:min_length]

print(f"Aligned - Features: {features.shape}, Targets: {target_returns.shape}")

# Train/test split (80/20)
split = int(0.8 * min_length)
X_train, X_test = features[:split], features[split:]
y_train, y_test = target_returns[:split], target_returns[split:]

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test: X={X_test.shape}, y={y_test.shape}")

# Aryabhata's celestial model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n=== Daily Celestial Prediction Results ===")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Overfitting Gap: {train_r2 - test_r2:.6f}")

# Current prediction
current_return_pred = model.predict(features[-1:].reshape(1, -1))[0]
current_price = closes[-1]
next_day_price = current_price * np.exp(current_return_pred)

print(f"\nCurrent Price: ${current_price:.2f}")
print(f"Predicted Return: {current_return_pred:.6f}")
print(f"Next Day Target: ${next_day_price:.2f}")
print(f"Direction: {'📈 UP' if current_return_pred > 0 else '📉 DOWN'}")

# Model evaluation
if test_r2 > 0.15:
    print(f"\n✅ Strong celestial signal detected!")
    confidence = min(test_r2 * 100, 25)
    print(f"Prediction confidence: {confidence:.1f}%")
elif test_r2 > 0.05:
    print(f"\n⚠️ Weak but valid celestial pattern")
elif test_r2 > 0:
    print(f"\n📊 Minimal pattern detected")
else:
    print(f"\n❌ No celestial order in daily movements")

# Active forces
print(f"\n=== Current Celestial State ===")
print(f"Trend Force: {d_trend[-1]:.4f}")
print(f"Reversion Force: {d_reversion[-1]:.4f}")
print(f"Volume Force: {d_volume[-1]:.4f}")
print(f"Volatility Force: {d_volatility[-1]:.4f}")
print(f"Structure Force: {d_structure[-1]:.4f}")
print(f"Momentum Force: {d_momentum[-1]:.4f}")

# Feature importance (which "celestial body" matters most)
importances = np.abs(model.coef_)
force_names = ['Daily Sin', 'Daily Cos', 'Weekly Sin', 'Weekly Cos', 
              'Trend Jya', 'Trend Kojya', 'Revert Jya', 'Revert Kojya',
              'Volume Jya', 'Volume Kojya', 'Vol Jya', 'Vol Kojya',
              'Structure Jya', 'Structure Kojya', 'Mom Jya', 'Mom Kojya']

top_force_idx = np.argmax(importances)
print(f"\nDominant celestial force: {force_names[top_force_idx]}")
print(f"Force strength: {importances[top_force_idx]:.2f}")

# Path to R² = 1
target_r2 = 1.0
current_gap = target_r2 - test_r2
print(f"\n🎯 Gap to R² = 1: {current_gap:.4f}")
print(f"Need {current_gap/0.15:.1f}x more celestial forces to approach perfection")