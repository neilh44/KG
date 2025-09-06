import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_market_data():
    kline_url = "https://fapi.binance.com/fapi/v1/klines"
    kline_params = {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 200}  # Shorter for stability
    klines = requests.get(kline_url, params=kline_params).json()
    return klines

klines = get_market_data()
R = 3438

# Extract data
closes = np.array([float(k[4]) for k in klines])
volumes = np.array([float(k[5]) for k in klines])
n = len(closes)

print(f"Daily data points: {n}")

# === CORE CELESTIAL FORCES ===
# 1. Trend vs mean reversion
ma_20 = np.array([np.mean(closes[max(0,i-19):i+1]) for i in range(n)])
trend_force = (closes - ma_20) / closes
d_trend = np.tanh(trend_force * 10)

# 2. Volume momentum  
vol_ma = np.array([np.mean(volumes[max(0,i-9):i+1]) for i in range(n)])
vol_ratio = volumes / (vol_ma + 1e-8)
d_volume = np.tanh((vol_ratio - 1) * 2)

# 3. Volatility regime
returns = np.diff(np.log(closes), prepend=0)
vol_current = np.abs(returns)
vol_ma_10 = np.array([np.mean(vol_current[max(0,i-9):i+1]) for i in range(n)])
vol_expansion = vol_current / (vol_ma_10 + 1e-8)
d_volatility = np.tanh((vol_expansion - 1) * 3)

# === MACRO CORRELATION FORCES ===
# Create realistic macro proxies from BTC's own patterns
time_idx = np.arange(n)

# SPY correlation (risk-on/risk-off)
risk_on_cycle = np.sin(time_idx * 2 * np.pi / 60) * 0.05  # 60-day cycle
spy_correlation = np.cumsum(returns * risk_on_cycle) / 100
d_spy = np.tanh(spy_correlation)

# DXY inverse correlation (dollar strength)
dxy_cycle = -np.cos(time_idx * 2 * np.pi / 90) * 0.03  # 90-day cycle
dxy_strength = np.cumsum(returns * dxy_cycle) / 100
d_dxy = np.tanh(dxy_strength)

# Gold safe-haven correlation
fear_events = (vol_expansion > 2.0).astype(float)  # High vol periods
gold_correlation = np.cumsum(fear_events) / n
d_gold = np.tanh((gold_correlation - 0.5) * 4)

# === JYA-KOJYA TRANSFORMS ===
all_forces = [d_trend, d_volume, d_volatility, d_spy, d_dxy, d_gold]
features_list = []

for force in all_forces:
    jya = R * np.sin(force)
    kojya = R * np.cos(force)
    features_list.extend([jya, kojya])

# Create feature matrix
features = np.column_stack(features_list)
print(f"Feature matrix shape: {features.shape}")

# === PREDICT TOMORROW'S DIRECTION ===
target = np.roll(returns, -1)[:-1]  # Next day return, remove last
features = features[:-1]  # Align

# Train/test split
split = int(0.75 * len(features))
X_train, X_test = features[:split], features[split:]
y_train, y_test = target[:split], target[split:]

# Model
model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n=== Enhanced Celestial Results ===")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}") 
print(f"Gap: {train_r2 - test_r2:.6f}")

# Current prediction
next_return = model.predict(features[-1:].reshape(1, -1))[0]
current_price = closes[-1]
next_price = current_price * (1 + next_return)

print(f"\nCurrent: ${current_price:.2f}")
print(f"Next Day: ${next_price:.2f}")
print(f"Direction: {'📈' if next_return > 0 else '📉'} ({next_return*100:.2f}%)")

# Force analysis
print(f"\n=== Celestial Forces ===")
print(f"Trend: {d_trend[-1]:.4f}")
print(f"Volume: {d_volume[-1]:.4f}")
print(f"Volatility: {d_volatility[-1]:.4f}")
print(f"SPY Correlation: {d_spy[-1]:.4f}")
print(f"DXY Strength: {d_dxy[-1]:.4f}")
print(f"Gold Fear: {d_gold[-1]:.4f}")

# Reality check
if test_r2 > 0.1:
    print(f"\n🎯 Strong macro signal! R² = {test_r2:.3f}")
elif test_r2 > 0.02:
    print(f"\n📊 Weak but real signal detected")
else:
    print(f"\n⚡ Even with macro forces, market chaos dominates")
    print(f"This is why hedge funds need insider info for R² > 0.3")