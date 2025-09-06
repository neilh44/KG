import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fetch BTCUSDT futures data + market depth
def get_market_data():
    # OHLCV data
    kline_url = "https://fapi.binance.com/fapi/v1/klines"
    kline_params = {'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 1000}
    klines = requests.get(kline_url, params=kline_params).json()
    
    # Order book depth
    depth_url = "https://fapi.binance.com/fapi/v1/depth"
    depth_params = {'symbol': 'BTCUSDT', 'limit': 500}
    depth = requests.get(depth_url, params=depth_params).json()
    
    return klines, depth

klines, order_book = get_market_data()

# Extract price data
closes = np.array([float(candle[4]) for candle in klines])
volumes = np.array([float(candle[5]) for candle in klines])
n = len(closes)

# Aryabhata's radius
R = 3438

# === FUNDAMENTAL MARKET MECHANICS (like celestial bodies) ===

# 1. MARKET MAKER POSITIONS (like planetary positions)
# Liquidity center of gravity
bids = np.array([[float(b[0]), float(b[1])] for b in order_book['bids'][:50]])
asks = np.array([[float(a[0]), float(a[1])] for a in order_book['asks'][:50]])

bid_center = np.sum(bids[:, 0] * bids[:, 1]) / np.sum(bids[:, 1])  # Volume-weighted bid center
ask_center = np.sum(asks[:, 0] * asks[:, 1]) / np.sum(asks[:, 1])  # Volume-weighted ask center
liquidity_center = (bid_center + ask_center) / 2

# Distance from liquidity center (like earth-moon distance)
d_liquidity = np.abs(closes[-1] - liquidity_center) / closes[-1]

# 2. INSTITUTIONAL FLOW (like gravitational pull)
# Large volume vs small volume ratio
volume_ma = np.array([np.mean(volumes[max(0,i-19):i+1]) for i in range(n)])
volume_ratio = volumes / (volume_ma + 1e-8)
large_volume_periods = volume_ratio > 2.0  # Institutional activity

# Flow momentum (weighted by institutional activity)
price_changes = np.diff(closes)
institutional_flow = np.mean(price_changes[large_volume_periods[-len(price_changes):]])
d_flow = institutional_flow / np.std(price_changes)

# 3. MARKET STRUCTURE TENSION (like celestial stress)
# Support/Resistance tension
lookback = 100
recent_highs = closes[-lookback:]
recent_lows = closes[-lookback:]

resistance = np.percentile(recent_highs, 95)
support = np.percentile(recent_lows, 5)
current_price = closes[-1]

# Distance from equilibrium (middle of support-resistance)
equilibrium = (resistance + support) / 2
d_structure = (current_price - equilibrium) / (resistance - support + 1e-8)

# 4. OPTIONS GAMMA EXPOSURE PROXY (futures funding rate effect)
# Funding rate creates artificial gravity
try:
    funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    funding_params = {'symbol': 'BTCUSDT'}
    funding_data = requests.get(funding_url, params=funding_params).json()
    funding_rate = float(funding_data['lastFundingRate'])
except:
    funding_rate = 0.0001  # Default if API fails

d_funding = funding_rate * 1000  # Scale for jya-kojya

print(f"=== Aryabhata's Market Celestial Mechanics ===")
print(f"Liquidity Center: ${liquidity_center:.2f}")
print(f"Institutional Flow: {institutional_flow:.2f}")
print(f"Structure Tension: {d_structure:.4f}")
print(f"Funding Gravity: {funding_rate:.6f}")

# Apply jya-kojya to fundamental market distances
jya_liquidity = R * np.sin(d_liquidity * 10)  # Scale factor
kojya_liquidity = R * np.cos(d_liquidity * 10)

jya_flow = R * np.sin(d_flow)
kojya_flow = R * np.cos(d_flow)

jya_structure = R * np.sin(d_structure * np.pi)  # Already normalized
kojya_structure = R * np.cos(d_structure * np.pi)

jya_funding = R * np.sin(d_funding)
kojya_funding = R * np.cos(d_funding)

# Combine all celestial forces
celestial_forces = np.array([jya_liquidity, kojya_liquidity, jya_flow, kojya_flow, 
                            jya_structure, kojya_structure, jya_funding, kojya_funding]).reshape(1, -1)

# For historical fitting, create simplified model using last values
historical_features = np.tile(celestial_forces, (n, 1))

# Add some variation based on recent price movements
for i in range(n):
    scale_factor = 1 + (i - n/2) / (n * 10)  # Slight time variation
    historical_features[i] *= scale_factor

# Aryabhata's celestial model
model = LinearRegression()
model.fit(historical_features, closes)
y_pred = model.predict(historical_features)

print(f"Celestial Model R²: {r2_score(closes, y_pred):.6f}")

# Predict next price using current market celestial positions
next_prediction = model.predict(celestial_forces)[0]

print(f"\nCurrent close: ${closes[-1]:.2f}")
print(f"Aryabhata's celestial prediction: ${next_prediction:.2f}")
print(f"Market force change: ${next_prediction - closes[-1]:.2f}")

# Market force analysis
print(f"\n=== Celestial Market Forces ===")
print(f"Liquidity pull: {d_liquidity:.4f} (distance from MM center)")
print(f"Institutional gravity: {d_flow:.4f} (flow momentum)")
print(f"Structure tension: {d_structure:.4f} (support/resistance)")
print(f"Funding gravity: {d_funding:.4f} (artificial pull)")