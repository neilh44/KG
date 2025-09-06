import requests
import json

class DataAnalyzer:
    def __init__(self):
        self.base_url = 'https://fapi.binance.com/fapi/v1/klines'
    
    def get_top_futures_symbols(self):
        """Get top 20 futures trading pairs by volume"""
        url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
        response = requests.get(url)
        data = response.json()
        
        # Filter USDT pairs and sort by volume
        usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
        top_20 = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:20]
        
        return [pair['symbol'] for pair in top_20]
    
    def fetch_ohlc_data(self, symbol, interval='1h', limit=1000):
        """Fetch OHLC data for futures analysis"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        return {
            'opens': [float(x[1]) for x in data],
            'highs': [float(x[2]) for x in data],
            'lows': [float(x[3]) for x in data],
            'closes': [float(x[4]) for x in data]
        }
    
    def calculate_percentile(self, data, percentile):
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[index]
    
    def analyze_symbol_stats(self, symbol):
        """Analyze VaR/VaG statistics for a symbol"""
        ohlc = self.fetch_ohlc_data(symbol)
        
        # Calculate relationships
        high_to_low = [((ohlc['lows'][i] - ohlc['highs'][i]) / ohlc['highs'][i]) * 100 
                       for i in range(len(ohlc['opens']))]
        open_to_high = [((ohlc['highs'][i] - ohlc['opens'][i]) / ohlc['opens'][i]) * 100 
                        for i in range(len(ohlc['opens']))]
        open_to_close = [((ohlc['closes'][i] - ohlc['opens'][i]) / ohlc['opens'][i]) * 100 
                         for i in range(len(ohlc['opens']))]
        
        stats = {}
        for name, data in [('high_to_low', high_to_low), ('open_to_high', open_to_high), ('open_to_close', open_to_close)]:
            var_5 = self.calculate_percentile(data, 5)
            vag_95 = self.calculate_percentile(data, 95)
            mean_val = sum(data) / len(data)
            
            stats[name] = {
                'var_5': var_5,
                'vag_95': vag_95,
                'mean': mean_val,
                'ratio': abs(vag_95 / var_5) if var_5 != 0 else 0
            }
        
        return stats
    
    def scan_top_coins(self):
        """Scan top 20 highest volume USDT futures coins"""
        symbols = self.get_top_futures_symbols()
        results = {}
        
        print(f"Starting analysis of {len(symbols)} top volume symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                results[symbol] = self.analyze_symbol_stats(symbol)
                print(f"Analyzed {i}/{len(symbols)}: {symbol}")
                
                # Rate limiting
                import time
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
                
        print(f"Analysis complete for {len(results)} symbols")
        return results