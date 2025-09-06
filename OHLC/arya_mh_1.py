import requests
import numpy as np
from sklearn.metrics import r2_score

def get_actual_funding_for_timestamp(timestamp, funding_history):
    """Get actual funding rate for specific timestamp"""
    if not funding_history:
        return 0
    # Find funding rate that was active at this timestamp
    for fund_time, rate in funding_history:
        if timestamp >= fund_time:
            return rate
    return funding_history[0][1]

def get_actual_basis_for_timestamp(timestamp, basis_history):
    """Get actual basis for specific timestamp"""
    if not basis_history:
        return 0
    # Find closest basis data
    closest = min(basis_history, key=lambda x: abs(x[0] - timestamp))
    return closest[1]

def get_actual_liquidity_for_interval(interval):
    """Get actual order book data for timeframe"""
    try:
        depth = requests.get("https://fapi.binance.com/fapi/v1/depth",
                           params={'symbol': 'BTCUSDT', 'limit': 100}).json()
        
        bid_total = sum(float(bid[1]) for bid in depth['bids'][:10])
        ask_total = sum(float(ask[1]) for ask in depth['asks'][:10])
        
        return bid_total / (bid_total + ask_total)
    except:
        return 0.5

def aryabhata_eclipse_geometry(obs, radius, funding_amp, liquidity_bias):
    """Geometric prediction using Aryabhata's method"""
    
    price_angle = obs['price_longitude'] * 2 * np.pi
    funding_angle = obs['funding_latitude'] * funding_amp
    volume_angle = np.arctan(obs['volume_intensity'] / 10000)
    liquidity_angle = (obs['liquidity_ratio'] - 0.5) * np.pi * 4
    
    price_jya = radius * np.sin(price_angle)
    funding_jya = np.sin(funding_angle) * 0.001
    volume_factor = 1 + 0.05 * np.sin(volume_angle)
    bias = np.sin(liquidity_angle) * liquidity_bias
    
    geometric_move = price_jya * funding_jya * volume_factor + bias * np.cos(price_angle)
    predicted_price = obs['current_price'] * (1 + geometric_move)
    
    return predicted_price

# Get ACTUAL historical data once
print("Fetching ACTUAL historical funding rates...")
funding_history = []
try:
    response = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                          params={'symbol': 'BTCUSDT', 'limit': 500})
    funding_data = response.json()
    funding_history = [(int(f['fundingTime']), float(f['fundingRate'])) for f in funding_data]
    funding_history.sort(key=lambda x: x[0], reverse=True)  # Most recent first
    print(f"Got {len(funding_history)} actual funding rates")
except Exception as e:
    print(f"Funding fetch error: {e}")

print("Fetching ACTUAL historical basis data...")
basis_history = []
try:
    # Get mark price data
    mark_response = requests.get("https://fapi.binance.com/fapi/v1/markPriceKlines",
                               params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 200})
    mark_klines = mark_response.json()
    
    # Get spot price data
    spot_response = requests.get("https://api.binance.com/api/v3/klines",
                               params={'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 200})
    spot_klines = spot_response.json()
    
    for mark_k, spot_k in zip(mark_klines, spot_klines):
        mark_price = float(mark_k[4])
        spot_price = float(spot_k[4])
        timestamp = int(mark_k[6])
        basis = (mark_price - spot_price) / spot_price
        basis_history.append((timestamp, basis))
    
    basis_history.sort(key=lambda x: x[0], reverse=True)
    print(f"Got {len(basis_history)} actual basis points")
except Exception as e:
    print(f"Basis fetch error: {e}")

# Define timeframes
timeframes = [
    ('5m', 20, 1, 0.1, 500000, 0.0005),
    ('15m', 16, 1, 0.2, 200000, 0.001),
    ('1h', 24, 1, 0.5, 50000, 0.002)
]

# Test each timeframe with ACTUAL data
for interval, history, predict_ahead, radius, funding_amp, liquidity_bias in timeframes:
    print(f"\n🌙 Aryabhata {interval.upper()} Eclipse Geometry")
    print("="*50)
    
    # Get ACTUAL price data for this specific interval
    klines = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={'symbol': 'BTCUSDT', 'interval': interval, 'limit': 100}).json()
    
    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    timestamps = np.array([int(k[6]) for k in klines])
    
    predictions = []
    actuals = []
    
    print("Point | Actual    | Predicted | Error")
    print("------|-----------|-----------|--------")
    
    for i in range(history, len(closes) - predict_ahead):
        current_timestamp = timestamps[i]
        
        # Get ACTUAL funding rate for this timestamp
        actual_funding = get_actual_funding_for_timestamp(current_timestamp, funding_history)
        
        # Get ACTUAL basis for this timestamp
        actual_basis = get_actual_basis_for_timestamp(current_timestamp, basis_history)
        
        # Calculate price position from actual data
        window_high = highs[i-history:i+1].max()
        window_low = lows[i-history:i+1].min()
        price_range = window_high - window_low
        
        if price_range == 0:
            price_longitude = 0.5
        else:
            price_longitude = (closes[i] - window_low) / price_range
            
        # Calculate liquidity ratio from volume dynamics
        volume_ma = np.mean(volumes[i-5:i+1])
        liquidity_ratio = 0.5 + (volumes[i] - volume_ma) / (volume_ma + 1) * 0.2
        liquidity_ratio = np.clip(liquidity_ratio, 0.1, 0.9)
        
        # Build observables with ACTUAL data
        obs = {
            'price_longitude': price_longitude,
            'funding_latitude': actual_funding,     # REAL funding rate
            'basis_distance': actual_basis,         # REAL mark-spot basis
            'liquidity_ratio': liquidity_ratio,     # Volume-derived
            'volume_intensity': volumes[i],
            'current_price': closes[i]
        }
        
        pred = aryabhata_eclipse_geometry(obs, radius, funding_amp, liquidity_bias)
        actual = closes[i + predict_ahead]
        
        predictions.append(pred)
        actuals.append(actual)
        
        if len(predictions) <= 9:
            error_pct = abs(pred - actual) / actual * 100
            # Convert timestamp to readable date
            from datetime import datetime
            candle_time = datetime.fromtimestamp(current_timestamp / 1000).strftime('%m-%d %H:%M')
            print(f"{len(predictions):5d} | {candle_time:19s} | $ {actual:8.0f} | $ {pred:8.0f} | {error_pct:5.1f}%")
    
    if len(predictions) > 0:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        r2 = r2_score(actuals, predictions)
        mae = np.mean(np.abs(predictions - actuals))
        
        print(f"\nR² Score: {r2:.4f} | Mean Error: ${mae:.0f}")
    else:
        print("No predictions generated")

# Live prediction with actual observables
print(f"\n🌙 Live Eclipse Calculations:")
try:
    # Get current time
    from datetime import datetime
    current_time = datetime.now()
    
    # Get live data
    ticker = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                         params={'symbol': 'BTCUSDT'}).json()
    current_price = float(ticker['lastPrice'])
    high_24h = float(ticker['highPrice'])
    low_24h = float(ticker['lowPrice'])
    volume_24h = float(ticker['volume'])
    
    funding = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                          params={'symbol': 'BTCUSDT'}).json()
    funding_rate = float(funding['lastFundingRate'])
    mark_price = float(funding['markPrice'])
    
    depth = requests.get("https://fapi.binance.com/fapi/v1/depth",
                        params={'symbol': 'BTCUSDT', 'limit': 20}).json()
    
    bid_total = sum(float(bid[1]) for bid in depth['bids'][:5])
    ask_total = sum(float(ask[1]) for ask in depth['asks'][:5])
    
    live_obs = {
        'price_longitude': (current_price - low_24h) / (high_24h - low_24h),
        'funding_latitude': funding_rate,
        'basis_distance': (mark_price - current_price) / current_price,
        'liquidity_ratio': bid_total / (bid_total + ask_total),
        'volume_intensity': volume_24h,
        'current_price': current_price
    }
    
    print(f"Current Time: {current_time.strftime('%m-%d %H:%M:%S')}")
    print(f"Current Price: ${current_price:.0f}")
    print(f"Funding Rate: {funding_rate:.6f} ({funding_rate*100:.4f}%)")
    print(f"Mark-Spot Basis: {live_obs['basis_distance']*100:.4f}%")
    print(f"Liquidity Ratio: {live_obs['liquidity_ratio']:.3f}")
    print()
    
    for interval, _, _, radius, funding_amp, liquidity_bias in timeframes:
        pred = aryabhata_eclipse_geometry(live_obs, radius, funding_amp, liquidity_bias)
        move_pct = (pred - live_obs['current_price']) / live_obs['current_price']
        
        # Calculate target time
        if interval == '5m':
            target_time = current_time.replace(second=0, microsecond=0)
            target_time = target_time.replace(minute=(target_time.minute // 5 + 1) * 5)
        elif interval == '15m':
            target_time = current_time.replace(second=0, microsecond=0)
            target_time = target_time.replace(minute=(target_time.minute // 15 + 1) * 15)
        else:  # 1h
            target_time = current_time.replace(minute=0, second=0, microsecond=0)
            target_time = target_time.replace(hour=target_time.hour + 1)
        
        print(f"{interval.upper()} prediction for {target_time.strftime('%m-%d %H:%M')}: ${pred:.0f} ({move_pct*100:+.2f}%)")
        
except Exception as e:
    print(f"Error: {e}")

print(f"\n💫 Geometric astronomy with ACTUAL market data!")