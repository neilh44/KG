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

# Test across multiple timeframes
timeframes = [
    ('5m', 12, 1, 0.1, 500000, 0.0005),    # 5min: 1H history, predict 1 period
    ('15m', 8, 1, 0.2, 200000, 0.001),     # 15min: 2H history, predict 1 period  
    ('1h', 6, 2, 0.5, 50000, 0.002)        # 1H: 6H history, predict 1 period
]

for interval, history, predict_ahead, radius, funding_amp, liquidity_bias in timeframes:
    print(f"\n{'='*50}")
    print(f"🌙 Aryabhata {interval.upper()} Eclipse Geometry")
    print(f"{'='*50}")
    
    klines = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={'symbol': 'BTCUSDT', 'interval': interval, 'limit': 40}).json()
    
    prices = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    
    predictions = []
    actuals = []
    
    print("Point | Actual    | Predicted | Error")
    print("------|-----------|-----------|--------")
    
    for i in range(history, len(prices) - predict_ahead):
        
        obs = {
            'price_longitude': (prices[i] - lows[i-history:i+1].min()) / max((highs[i-history:i+1].max() - lows[i-history:i+1].min()), 1),
            'funding_latitude': np.random.normal(0, 0.0001),
            'basis_distance': np.random.normal(0, 0.0001),
            'liquidity_ratio': 0.5 + np.random.normal(0, 0.05),
            'volume_intensity': volumes[i],
            'current_price': prices[i]
        }
        
        # Timeframe-specific geometry
        price_angle = obs['price_longitude'] * 2 * np.pi
        funding_angle = obs['funding_latitude'] * funding_amp
        volume_angle = np.arctan(obs['volume_intensity'] / 10000)
        liquidity_angle = (obs['liquidity_ratio'] - 0.5) * np.pi * 4
        
        # Geometric calculations
        price_jya = radius * np.sin(price_angle)
        funding_jya = np.sin(funding_angle) * 0.001
        volume_factor = 1 + 0.05 * np.sin(volume_angle)
        bias = np.sin(liquidity_angle) * liquidity_bias
        
        geometric_move = price_jya * funding_jya * volume_factor + bias * np.cos(price_angle)
        pred = obs['current_price'] * (1 + geometric_move)
        
        predictions.append(pred)
        actuals.append(prices[i + predict_ahead])
        
        if len(predictions) <= 9:
            error_pct = abs(pred - prices[i + predict_ahead]) / prices[i + predict_ahead] * 100
            print(f"{len(predictions):5d} | $ {prices[i + predict_ahead]:8.0f} | $ {pred:8.0f} | {error_pct:5.1f}%")
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r2 = r2_score(actuals, predictions)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"\nR² Score: {r2:.4f} | Mean Error: ${mae:.0f}")

# Live prediction for all timeframes
print(f"\n🌙 Live Eclipse Calculations:")
try:
    live_obs = get_btc_observables()
    
    for interval, _, _, radius, funding_amp, liquidity_bias in timeframes:
        price_angle = live_obs['price_longitude'] * 2 * np.pi
        funding_angle = live_obs['funding_latitude'] * funding_amp
        volume_angle = np.arctan(live_obs['volume_intensity'] / 10000)
        liquidity_angle = (live_obs['liquidity_ratio'] - 0.5) * np.pi * 4
        
        price_jya = radius * np.sin(price_angle)
        funding_jya = np.sin(funding_angle) * 0.001
        volume_factor = 1 + 0.05 * np.sin(volume_angle)
        bias = np.sin(liquidity_angle) * liquidity_bias
        
        geometric_move = price_jya * funding_jya * volume_factor + bias * np.cos(price_angle)
        pred = live_obs['current_price'] * (1 + geometric_move)
        move_pct = (pred - live_obs['current_price']) / live_obs['current_price']
        
        print(f"{interval.upper()}: ${pred:.0f} ({move_pct*100:+.2f}%)")
        
except Exception as e:
    print(f"Error: {e}")

print(f"\n💫 Geometric astronomy across all market cycles!")