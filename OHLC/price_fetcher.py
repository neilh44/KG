import requests
import websocket
import json
import threading
from datetime import datetime

class PriceFetcher:
    def __init__(self):
        self.base_url = 'https://fapi.binance.com/fapi/v1'
        self.ws_url = 'wss://fstream.binance.com/ws'
        self.price_cache = {}
        self.ws_connections = {}
        
    def get_market_data(self, symbol):
        """Get comprehensive market data for symbol"""
        try:
            # Current price
            ticker_url = f"{self.base_url}/ticker/price"
            price_response = requests.get(ticker_url, params={'symbol': symbol})
            current_price = float(price_response.json()['price'])
            
            # 24hr stats
            stats_url = f"{self.base_url}/ticker/24hr"
            stats_response = requests.get(stats_url, params={'symbol': symbol})
            stats = stats_response.json()
            
            # Current candle open
            kline_url = f"{self.base_url}/klines"
            kline_response = requests.get(kline_url, params={
                'symbol': symbol,
                'interval': '1h',
                'limit': 1
            })
            candle_data = kline_response.json()[0]
            
            return {
                'symbol': symbol,
                'price': current_price,
                'open_24h': float(stats['openPrice']),
                'high_24h': float(stats['highPrice']),
                'low_24h': float(stats['lowPrice']),
                'volume': float(stats['volume']),
                'change_24h': float(stats['priceChangePercent']),
                'candle_open': float(candle_data[1]),
                'candle_high': float(candle_data[2]),
                'candle_low': float(candle_data[3]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_prices(self, symbols):
        """Get prices for multiple symbols efficiently"""
        try:
            # Batch request for all symbols
            ticker_url = f"{self.base_url}/ticker/price"
            response = requests.get(ticker_url)
            all_prices = response.json()
            
            # Filter for requested symbols
            symbol_set = set(symbols)
            filtered_prices = {
                item['symbol']: float(item['price']) 
                for item in all_prices 
                if item['symbol'] in symbol_set
            }
            
            return filtered_prices
            
        except Exception as e:
            return {symbol: None for symbol in symbols}
    
    def start_price_stream(self, symbols, callback=None):
        """Start WebSocket stream for real-time prices"""
        if not symbols:
            return
            
        # Create stream string
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
        stream_url = f"{self.ws_url}/stream?streams={'/'.join(streams)}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    ticker_data = data['data']
                    symbol = ticker_data['s']
                    price = float(ticker_data['c'])
                    
                    self.price_cache[symbol] = {
                        'price': price,
                        'change': float(ticker_data['P']),
                        'volume': float(ticker_data['v']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if callback:
                        callback(symbol, self.price_cache[symbol])
                        
            except Exception as e:
                print(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_open(ws):
            print(f"WebSocket connected for {len(symbols)} symbols")
        
        # Start WebSocket in separate thread
        ws = websocket.WebSocketApp(
            stream_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        def run_ws():
            ws.run_forever()
            
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        
        return ws
    
    def get_cached_price(self, symbol):
        """Get price from cache if available"""
        return self.price_cache.get(symbol, {}).get('price')
    
    def get_orderbook_depth(self, symbol, limit=10):
        """Get order book depth"""
        try:
            url = f"{self.base_url}/depth"
            response = requests.get(url, params={'symbol': symbol, 'limit': limit})
            data = response.json()
            
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in data['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in data['asks']],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}