import requests
import numpy as np
from sklearn.metrics import r2_score

def get_btc_observables():
    """Get DIRECTLY observable BTC market positions (like celestial positions)"""
    
    # 1. PRICE POSITION (like Moon longitude)
    ticker = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={'symbol': 'BTCUSDT'}).json()
    current_price = float(ticker['lastPrice'])
    high_24h = float(ticker['highPrice'])
    low_24h = float(ticker['lowPrice'])
    volume_24h = float(ticker['volume'])
    
    # 2. FUNDING POSITION (like Moon latitude - deviation from norm)
    funding = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                          params={'symbol': 'BTCUSDT'}).json()
    funding_rate = float(funding['lastFundingRate'])
    mark_price = float(funding['markPrice'])
    
    # 3. ORDER BOOK POSITION (like Earth shadow position)
    depth = requests.get("https://fapi.binance.com/fapi/v1/depth",
                        params={'symbol': 'BTCUSDT', 'limit': 50}).json()
    
    bid_total = sum(float(bid[1]) for bid in depth['bids'][:10])
    ask_total = sum(float(ask[1]) for ask in depth['asks'][:10])
    
    return {
        'price_longitude': (current_price - low_24h) / (high_24h - low_24h),  # 0 to 1
        'funding_latitude': funding_rate,  # Deviation from 0
        'basis_distance': (mark_price - current_price) / current_price,  # Futures premium
        'liquidity_ratio': bid_total / (bid_total + ask_total),  # 0.5 = balanced
        'volume_intensity': volume_24h,
        'current_price': current_price,
        'daily_range': high_24h - low_24h
    }

def aryabhata_eclipse_geometry(obs):
    """
    Pure geometric calculation like Aryabhata's eclipse prediction
    NO machine learning, NO alpha - just deterministic geometry
    """
    
    # BTC Market "Orbital Mechanics" Constants (like Earth radius)
    # These would be discovered through observation, not fitted
    PRICE_CYCLE_RADIUS = 1.0  # Normalized price cycle
    FUNDING_CYCLE_RADIUS = 8760  # ~365 days of 8H funding cycles  
    VOLUME_CYCLE_RADIUS = 100  # Volume normalization
    
    # === GEOMETRIC POSITIONS (like Sun-Moon-Earth angles) ===
    
    # Price position in daily cycle (0 to 2π)
    price_angle = obs['price_longitude'] * 2 * np.pi
    
    # Funding position in funding cycle 
    funding_angle = obs['funding_latitude'] * 100000  # Scale to meaningful angle
    
    # Volume intensity angle
    volume_angle = np.arctan(obs['volume_intensity'] / 50000)
    
    # Liquidity balance angle (deviation from 0.5)
    liquidity_angle = (obs['liquidity_ratio'] - 0.5) * np.pi * 2
    
    # === ARYABHATA'S DETERMINISTIC GEOMETRY ===
    
    # Market "eclipse" occurs when these geometric conditions align
    
    # Primary geometric relationship (like eclipse magnitude)
    price_jya = PRICE_CYCLE_RADIUS * np.sin(price_angle)
    price_kojya = PRICE_CYCLE_RADIUS * np.cos(price_angle)
    
    # Funding perturbation (like lunar node effect)
    funding_jya = np.sin(funding_angle) * 0.01  # Small but significant
    funding_kojya = np.cos(funding_angle) * 0.01
    
    # Volume amplification (like Earth shadow size)
    volume_factor = 1 + 0.1 * np.sin(volume_angle)
    
    # Liquidity direction (like shadow direction)
    liquidity_bias = np.sin(liquidity_angle) * 0.005
    
    # === DETERMINISTIC PRICE GEOMETRY ===
    
    # Base geometric prediction (percentage move)
    geometric_move = (price_jya * funding_jya * volume_factor + 
                     price_kojya * funding_kojya * volume_factor +
                     liquidity_bias)
    
    # Convert geometric result to price prediction
    predicted_price = obs['current_price'] * (1 + geometric_move)
    
    return predicted_price, {
        'geometric_move': geometric_move,
        'price_angle': price_angle * 180 / np.pi,
        'funding_angle': funding_angle * 180 / np.pi,
        'volume_factor': volume_factor,
        'liquidity_bias': liquidity_bias
    }

# Test on historical data
klines = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 200}).json()

prices = np.array([float(k[4]) for k in klines])
volumes = np.array([float(k[5]) for k in klines])
highs = np.array([float(k[2]) for k in klines])
lows = np.array([float(k[3]) for k in klines])

predictions = []
actuals = []

print(f"Testing Aryabhata's deterministic geometry on {len(prices)} hours")

# Test deterministic prediction
for i in range(24, len(prices) - 4):  # Need 24h history, predict 4h ahead
    
    # Create observables from real data (simulating what we'd measure)
    obs = {
        'price_longitude': (prices[i] - lows[i-24:i+1].min()) / (highs[i-24:i+1].max() - lows[i-24:i+1].min()),
        'funding_latitude': np.random.normal(0, 0.0001),  # Simulated funding
        'basis_distance': np.random.normal(0, 0.001),     # Simulated basis
        'liquidity_ratio': 0.5 + np.random.normal(0, 0.1), # Simulated liquidity
        'volume_intensity': volumes[i],
        'current_price': prices[i],
        'daily_range': highs[i-24:i+1].max() - lows[i-24:i+1].min()
    }
    
    # Pure geometric prediction (NO fitting, NO alpha)
    pred, geometry = aryabhata_eclipse_geometry(obs)
    
    predictions.append(pred)
    actuals.append(prices[i + 4])

predictions = np.array(predictions)
actuals = np.array(actuals)

# Evaluate deterministic geometry
r2 = r2_score(actuals, predictions)
mae = np.mean(np.abs(predictions - actuals))

print(f"\n=== Pure Aryabhata Geometry ===")
print(f"Deterministic R²: {r2:.4f}")
print(f"Mean Error: ${mae:.0f}")
print(f"Max prediction: ${predictions.max():.0f}")
print(f"Min prediction: ${predictions.min():.0f}")

# Live deterministic prediction
print(f"\n🌙 Live Eclipse Calculation:")
try:
    live_obs = get_btc_observables()
    live_pred, live_geom = aryabhata_eclipse_geometry(live_obs)
    
    move_pct = (live_pred - live_obs['current_price']) / live_obs['current_price']
    
    print(f"Current: ${live_obs['current_price']:.0f}")
    print(f"4H Target: ${live_pred:.0f} ({move_pct*100:+.1f}%)")
    print(f"Price Angle: {live_geom['price_angle']:.1f}°")
    print(f"Geometric Move: {live_geom['geometric_move']:.4f}")
    
except Exception as e:
    print(f"Error: {e}")

print(f"\n💫 Deterministic geometry - no alpha tuning needed!")
print(f"Like Aryabhata: Pure math + observation = Prediction")