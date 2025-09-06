import requests

def test_binance_api():
    """Test Binance Futures API directly"""
    url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
    
    print("Testing Binance Futures API...")
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"API Error: {response.status_code}")
        return
        
    data = response.json()
    print(f"Total futures pairs received: {len(data)}")
    
    # Filter USDT pairs
    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
    print(f"Total USDT pairs: {len(usdt_pairs)}")
    
    # Filter active pairs (volume > 0)
    active_usdt = [item for item in usdt_pairs if float(item['quoteVolume']) > 0]
    print(f"Active USDT pairs (volume > 0): {len(active_usdt)}")
    
    # Sort by volume
    sorted_pairs = sorted(active_usdt, key=lambda x: float(x['quoteVolume']), reverse=True)
    
    print(f"\nTop 20 USDT pairs by volume:")
    for i, pair in enumerate(sorted_pairs[:20], 1):
        print(f"{i:2d}. {pair['symbol']:15} Volume: {float(pair['quoteVolume']):,.0f}")
    
    print(f"\nAll {len(sorted_pairs)} USDT symbols:")
    symbols = [pair['symbol'] for pair in sorted_pairs]
    print(symbols)
    
    return symbols

if __name__ == "__main__":
    symbols = test_binance_api()
    print(f"\nTotal symbols that would be analyzed: {len(symbols) if symbols else 0}")