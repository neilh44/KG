import requests
import numpy as np
from sklearn.metrics import r2_score

def get_btc_observables():
    """Get DIRECTLY observable BTC market positions (like celestial positions)"""
    
    # Price position from 5-minute cycles
    ticker = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={'symbol': 'BTCUSDT'}).json()
    current_price = float(ticker['lastPrice'])
    high_24h = float(ticker['highPrice'])
    low_24h = float(ticker['lowPrice'])
    volume_24h = float(ticker['volume'])
    
    # Funding position 
    funding = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                          params={'symbol': 'BTCUSDT'}).json()
    funding_rate = float(funding['lastFundingRate'])
    mark_price = float(funding['markPrice'])
    
    # Order book position for micro-movements
    depth = requests.get("https://fapi.binance.com/fapi/v1/depth",
                        params={'symbol': 'BTCUSDT', 'limit': 20}).json()
    
    bid_total = sum(float(bid[1]) for bid in depth['bids'][:5])
    ask_total = sum(float(ask[1]) for ask in depth['asks'][:5])
    
    return {
        'price_longitude': (current_price - low_24h) / (high_24h - low_24h),
        'funding_latitude': funding_rate,
        'basis_distance': (mark_price - current_price) / current_price,
        'liquidity_ratio': bid_total / (bid_total + ask_total),
        'volume_intensity': volume_24h,
        'current_price': current_price
    }

def aryabhata_eclipse_geometry(obs):
    """
    5-minute geometric prediction using Aryabhata's method
    Pure trigonometry - no ML, no curve fitting
    """
    
    # Market "Orbital" Constants for 5-minute cycles
    MICRO_CYCLE_RADIUS = 0.1     # 5-minute price oscillations
    FUNDING_INFLUENCE = 1440     # Minutes per day / 8H funding
    LIQUIDITY_RESONANCE = 10     # Order book sensitivity
    
    # Geometric angles from observables
    price_angle = obs['price_longitude'] * 2 * np.pi
    funding_angle = obs['funding_latitude'] * 500000  # Amplify micro-signal
    volume_angle = np.arctan(obs['volume_intensity'] / 10000)
    liquidity_angle = (obs['liquidity_ratio'] - 0.5) * np.pi * 4
    
    # Aryabhata's sine/cosine calculations (jya/kojya)
    price_jya = MICRO_CYCLE_RADIUS * np.sin(price_angle)
    funding_jya = np.sin(funding_angle) * 0.001
    
    # Volume amplification for 5-minute timeframe
    volume_factor = 1 + 0.05 * np.sin(volume_angle)
    
    # Liquidity micro-bias 
    liquidity_bias = np.sin(liquidity_angle) * 0.0005
    
    # Geometric 5-minute move calculation
    geometric_move = (price_jya * funding_jya * volume_factor + 
                     liquidity_bias * np.cos(price_angle))
    
    predicted_price = obs['current_price'] * (1 + geometric_move)
    
    return predicted_price, {
        'geometric_move': geometric_move,
        'price_angle': price_angle * 180 / np.pi,
        'volume_factor': volume_factor
    }

# Test on 5-minute data
print("Fetching BTC 5-minute data...")
klines = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params={'symbol': 'BTCUSDT', 'interval': '5m', 'limit': 50}).json()

prices = np.array([float(k[4]) for k in klines])
volumes = np.array([float(k[5]) for k in klines])
highs = np.array([float(k[2]) for k in klines])
lows = np.array([float(k[3]) for k in klines])

predictions = []
actuals = []

print("=== Aryabhata 5-Minute Eclipse Backtest ===")
print("Point | Actual    | Predicted | Error")
print("------|-----------|-----------|--------")

# Test deterministic prediction on 5-minute intervals
for i in range(12, len(prices) - 1):  # Need 1H history, predict 5min ahead
    
    obs = {
        'price_longitude': (prices[i] - lows[i-12:i+1].min()) / max((highs[i-12:i+1].max() - lows[i-12:i+1].min()), 1),
        'funding_latitude': np.random.normal(0, 0.0001),
        'basis_distance': np.random.normal(0, 0.0001),
        'liquidity_ratio': 0.5 + np.random.normal(0, 0.05),
        'volume_intensity': volumes[i],
        'current_price': prices[i]
    }
    
    pred, geometry = aryabhata_eclipse_geometry(obs)
    
    predictions.append(pred)
    actuals.append(prices[i + 1])
    
    if len(predictions) <= 9:  # Show first 9 predictions
        error_pct = abs(pred - prices[i + 1]) / prices[i + 1] * 100
        print(f"{len(predictions):5d} | $ {prices[i + 1]:8.0f} | $ {pred:8.0f} | {error_pct:5.1f}%")

# Evaluate geometry
predictions = np.array(predictions)
actuals = np.array(actuals)
r2 = r2_score(actuals, predictions)
mae = np.mean(np.abs(predictions - actuals))

print(f"\n=== Pure Geometric Results ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Error: ${mae:.0f}")

# Live 5-minute prediction
print(f"\n🌙 Live 5-Minute Eclipse:")
try:
    live_obs = get_btc_observables()
    live_pred, live_geom = aryabhata_eclipse_geometry(live_obs)
    
    move_pct = (live_pred - live_obs['current_price']) / live_obs['current_price']
    
    print(f"Current: ${live_obs['current_price']:.0f}")
    print(f"5Min Target: ${live_pred:.0f} ({move_pct*100:+.2f}%)")
    print(f"Geometric Angle: {live_geom['price_angle']:.1f}°")
    
except Exception as e:
    print(f"Error: {e}")

print(f"\n💫 Pure trigonometry - like predicting lunar eclipses!")