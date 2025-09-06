import requests
import time
from datetime import datetime

class SignalGenerator:
    def __init__(self):
        self.ticker_url = 'https://fapi.binance.com/fapi/v1/ticker/price'
        self.kline_url = 'https://fapi.binance.com/fapi/v1/klines'
        
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        params = {'symbol': symbol}
        response = requests.get(self.ticker_url, params=params)
        return float(response.json()['price'])
    
    def get_current_candle_open(self, symbol, interval='1h'):
        """Get current candle's open price"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1
        }
        response = requests.get(self.kline_url, params=params)
        data = response.json()[0]
        return float(data[1])  # Open price
    
    def generate_signal(self, symbol, analysis_stats=None):
        """Generate trading signal based on price movement from open"""
        try:
            current_price = self.get_current_price(symbol)
            open_price = self.get_current_candle_open(symbol)
            
            from_open = ((current_price - open_price) / open_price) * 100
            
            # Dynamic thresholds based on symbol volatility (use BTC defaults if no analysis)
            if analysis_stats and 'open_to_close' in analysis_stats:
                volatility = abs(analysis_stats['open_to_close']['vag_95'] - analysis_stats['open_to_close']['var_5'])
                neutral_zone = volatility * 0.15  # 15% of typical range
                trend_threshold = volatility * 0.30  # 30% of typical range
                fade_threshold = volatility * 0.60   # 60% of typical range
            else:
                # BTC-like defaults
                neutral_zone = 0.30
                trend_threshold = 0.60
                fade_threshold = 1.20
            
            signal = self._calculate_signal_logic(from_open, current_price, open_price, 
                                                 neutral_zone, trend_threshold, fade_threshold)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'open_price': open_price,
                'from_open_pct': from_open,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}
    
    def _calculate_signal_logic(self, from_open, current_price, open_price, 
                               neutral_zone, trend_threshold, fade_threshold):
        """Core signal logic based on movement from open"""
        
        if abs(from_open) < neutral_zone:
            return {
                'action': 'NEUTRAL',
                'reason': 'Within neutral zone',
                'target': None,
                'stop': None
            }
        
        elif neutral_zone <= from_open < trend_threshold:
            # Trending up
            target = open_price * 1.0064  # +0.64% target
            stop = current_price * 0.996   # -0.4% stop
            return {
                'action': 'LONG',
                'reason': 'Trending up',
                'target': target,
                'stop': stop
            }
        
        elif from_open >= trend_threshold:
            # Fade zone - price too high
            target = current_price * 0.994  # -0.6% target
            stop = current_price * 1.004    # +0.4% stop
            return {
                'action': 'SHORT',
                'reason': 'Fade zone - overextended',
                'target': target,
                'stop': stop
            }
        
        elif -trend_threshold < from_open <= -neutral_zone:
            # Trending down
            target = open_price * 0.994
            stop = current_price * 1.004
            return {
                'action': 'SHORT',
                'reason': 'Trending down',
                'target': target,
                'stop': stop
            }
        
        elif from_open <= -trend_threshold:
            # Recovery zone - price too low
            recovery_target = current_price * 1.0067  # +0.67% recovery
            stop = current_price * 0.998              # -0.2% stop
            return {
                'action': 'LONG',
                'reason': 'Recovery zone - oversold',
                'target': recovery_target,
                'stop': stop
            }
    
    def scan_signals(self, symbols, analysis_results=None):
        """Generate signals for multiple symbols"""
        signals = []
        
        print(f"Generating signals for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols, 1):
            stats = analysis_results.get(symbol) if analysis_results else None
            signal = self.generate_signal(symbol, stats)
            signals.append(signal)
            
            if i % 20 == 0:  # Progress update every 20 symbols
                print(f"Generated signals for {i}/{len(symbols)} symbols...")
            
            time.sleep(0.1)  # Rate limiting
            
        print(f"Signal generation complete for {len(signals)} symbols")
        return signals