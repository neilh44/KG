"""
Data Ingestion Module - Binance API Integration
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class BinanceDataIngestion:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.futures_url = "https://fapi.binance.com/fapi/v1"
    
    def get_kline_data(self, symbol="BTCUSDT", interval="1h", limit=100):
        """Get OHLCV data"""
        endpoint = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        print(f"Fetching data from: {endpoint}")
        response = requests.get(endpoint, params=params)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API Error: {response.text}")
            return pd.DataFrame()
        
        data = response.json()
        print(f"Received {len(data)} data points")
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to proper data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_funding_rate(self, symbol="BTCUSDT"):
        """Get current funding rate"""
        endpoint = f"{self.futures_url}/premiumIndex"
        params = {'symbol': symbol}
        
        print(f"Fetching funding rate from: {endpoint}")
        response = requests.get(endpoint, params=params)
        print(f"Funding response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Funding API Error: {response.text}")
            return {'funding_rate': 0.0, 'mark_price': 0.0, 'index_price': 0.0}
        
        data = response.json()
        print(f"Funding data: {data}")
        
        return {
            'funding_rate': float(data['lastFundingRate']),
            'mark_price': float(data['markPrice']),
            'index_price': float(data['indexPrice'])
        }
    
    def get_order_book(self, symbol="BTCUSDT", limit=100):
        """Get order book depth"""
        endpoint = f"{self.base_url}/depth"
        params = {'symbol': symbol, 'limit': limit}
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        bids = [[float(x[0]), float(x[1])] for x in data['bids']]
        asks = [[float(x[0]), float(x[1])] for x in data['asks']]
        
        return {'bids': bids, 'asks': asks}
    
    def get_open_interest(self, symbol="BTCUSDT"):
        """Get futures open interest"""
        endpoint = f"{self.futures_url}/openInterest"
        params = {'symbol': symbol}
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        return {
            'open_interest': float(data['openInterest']),
            'timestamp': datetime.now()
        }
    
    def get_market_data_snapshot(self):
        """Get comprehensive market snapshot"""
        print("Getting market snapshot...")
        
        try:
            ohlcv = self.get_kline_data(limit=24)
            if ohlcv.empty:
                print("ERROR: No OHLCV data received")
                return None
            
            funding = self.get_funding_rate()
            order_book = self.get_order_book(limit=50)
            open_interest = self.get_open_interest()
            
            snapshot = {
                'timestamp': datetime.now(),
                'ohlcv': ohlcv,
                'funding': funding,
                'order_book': order_book,
                'open_interest': open_interest
            }
            
            print("Market snapshot created successfully")
            return snapshot
            
        except Exception as e:
            print(f"ERROR in get_market_data_snapshot: {e}")
            import traceback
            print(traceback.format_exc())
            return None

if __name__ == "__main__":
    ingestion = BinanceDataIngestion()
    data = ingestion.get_market_data_snapshot()
    print("Market data snapshot retrieved successfully")
    print(f"Latest BTC price: {data['ohlcv']['close'].iloc[-1]}")