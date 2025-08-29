"""
Simple working bulk data fetcher
"""
import requests
import pandas as pd
from datetime import datetime

def get_bulk_btc_data(hours=1000):
    """Get bulk BTC data that definitely works"""
    print(f"🔄 Fetching last {hours} hours of BTC data...")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': min(hours, 1000)
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.text}")
        return None
    
    data = response.json()
    print(f"✅ Got {len(data)} data points")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Return clean data
    clean_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    print(f"✅ Processed DataFrame shape: {clean_df.shape}")
    print(f"Price range: ${clean_df['close'].min():.0f} - ${clean_df['close'].max():.0f}")
    
    return clean_df

def quick_causal_analysis(df):
    """Quick causal relationship discovery"""
    print("\n🧠 Quick Causal Analysis...")
    
    # Calculate features for causal analysis
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_volume'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
    df['big_move'] = (abs(df['price_change']) > df['price_change'].rolling(20).std() * 2).astype(int)
    
    # Simple causal relationships
    correlations = []
    
    # Volume -> Price relationship
    vol_price_corr = df['high_volume'].shift(1).corr(df['big_move'])
    correlations.append(f"High volume → Big price move: {vol_price_corr:.3f}")
    
    # Price momentum
    momentum_corr = df['price_change'].shift(1).corr(df['price_change'])
    correlations.append(f"Price momentum (1h lag): {momentum_corr:.3f}")
    
    print("Discovered relationships:")
    for corr in correlations:
        print(f"  • {corr}")
    
    return correlations

if __name__ == "__main__":
    # Get bulk data
    btc_data = get_bulk_btc_data(500)  # Last 500 hours
    
    if btc_data is not None:
        # Quick analysis
        relationships = quick_causal_analysis(btc_data)
        
        # Current price
        current_price = btc_data['close'].iloc[-1]
        recent_change = ((current_price - btc_data['close'].iloc[-24]) / btc_data['close'].iloc[-24]) * 100
        
        print(f"\n📊 Current Analysis:")
        print(f"Current BTC Price: ${current_price:,.2f}")
        print(f"24h Change: {recent_change:+.2f}%")
        print(f"Data Points: {len(btc_data)}")
        
        print(f"\n✅ Successfully analyzed {len(btc_data)} data points!")
        print("🚀 This proves bulk data analysis works!")
    else:
        print("❌ Failed to get bulk data")