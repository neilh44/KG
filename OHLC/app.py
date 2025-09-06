from flask import Flask, jsonify, request
from data_analyzer import DataAnalyzer
from signal_generator import SignalGenerator  
from price_fetcher import PriceFetcher
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Initialize modules
analyzer = DataAnalyzer()
signal_gen = SignalGenerator()
price_fetcher = PriceFetcher()

# Cache for analysis results
analysis_cache = {}
signals_cache = {}
last_analysis_time = None

def background_analysis():
    """Background task to update analysis periodically"""
    global analysis_cache, last_analysis_time
    
    while True:
        try:
            print(f"[{datetime.now()}] Running background analysis...")
            
            # Debug: Check how many symbols we're getting
            symbols = analyzer.get_top_futures_symbols()
            print(f"[DEBUG] Retrieved {len(symbols)} symbols for analysis")
            
            analysis_cache = analyzer.scan_top_coins()
            last_analysis_time = datetime.now()
            print(f"[{datetime.now()}] Analysis complete for {len(analysis_cache)} symbols")
            
            # Wait 2 hours before next analysis (longer due to more symbols)
            time.sleep(7200)
            
        except Exception as e:
            print(f"Background analysis error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(600)  # Wait 10 minutes on error

def background_signals():
    """Background task to update signals every 5 minutes"""
    global signals_cache
    
    while True:
        try:
            if analysis_cache:
                symbols = list(analysis_cache.keys())
                signals_cache = signal_gen.scan_signals(symbols, analysis_cache)
                print(f"[{datetime.now()}] Updated signals for {len(signals_cache)} symbols")
            
            time.sleep(60)  # Update every 5 minutes
            
        except Exception as e:
            print(f"Background signals error: {e}")
            time.sleep(60)

@app.route('/api/top-coins')
def get_top_coins():
    """Get top 20 futures coins"""
    try:
        symbols = analyzer.get_top_futures_symbols()
        return jsonify({
            'success': True,
            'symbols': symbols,
            'count': len(symbols),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analysis')
def get_analysis():
    """Get statistical analysis for all coins"""
    if not analysis_cache:
        return jsonify({'success': False, 'error': 'Analysis not ready, please wait...'}), 404
    
    return jsonify({
        'success': True,
        'analysis': analysis_cache,
        'last_updated': last_analysis_time.isoformat() if last_analysis_time else None,
        'symbols_count': len(analysis_cache)
    })

@app.route('/api/analysis/<symbol>')
def get_symbol_analysis(symbol):
    """Get analysis for specific symbol"""
    symbol = symbol.upper()
    
    if symbol in analysis_cache:
        return jsonify({
            'success': True,
            'symbol': symbol,
            'analysis': analysis_cache[symbol],
            'last_updated': last_analysis_time.isoformat() if last_analysis_time else None
        })
    else:
        # Generate on-demand analysis
        try:
            stats = analyzer.analyze_symbol_stats(symbol)
            return jsonify({
                'success': True,
                'symbol': symbol,
                'analysis': stats,
                'generated': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals')
def get_signals():
    """Get all current trading signals"""
    if not signals_cache:
        return jsonify({'success': False, 'error': 'Signals not ready, please wait...'}), 404
    
    # Filter for actionable signals only
    actionable = [s for s in signals_cache if s.get('signal', {}).get('action') != 'NEUTRAL']
    
    return jsonify({
        'success': True,
        'signals': signals_cache,
        'actionable_signals': actionable,
        'total_count': len(signals_cache),
        'actionable_count': len(actionable),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/signals/<symbol>')
def get_symbol_signal(symbol):
    """Get signal for specific symbol"""
    symbol = symbol.upper()
    
    # Try from cache first
    cached_signal = next((s for s in signals_cache if s.get('symbol') == symbol), None)
    
    if cached_signal:
        return jsonify({
            'success': True,
            'signal': cached_signal
        })
    else:
        # Generate live signal
        try:
            stats = analysis_cache.get(symbol)
            signal = signal_gen.generate_signal(symbol, stats)
            return jsonify({
                'success': True,
                'signal': signal,
                'generated': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/price/<symbol>')
def get_price(symbol):
    """Get current price and market data"""
    symbol = symbol.upper()
    
    try:
        market_data = price_fetcher.get_market_data(symbol)
        return jsonify({
            'success': True,
            'data': market_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/prices')
def get_multiple_prices():
    """Get prices for multiple symbols"""
    symbols_param = request.args.get('symbols', '')
    
    if not symbols_param:
        # Use top 20 by default
        symbols = list(analysis_cache.keys()) if analysis_cache else analyzer.get_top_futures_symbols()
    else:
        symbols = [s.strip().upper() for s in symbols_param.split(',')]
    
    try:
        prices = price_fetcher.get_multiple_prices(symbols)
        return jsonify({
            'success': True,
            'prices': prices,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard data with signals and prices"""
    try:
        # Get actionable signals
        actionable_signals = [s for s in signals_cache if s.get('signal', {}).get('action') != 'NEUTRAL'] if signals_cache else []
        
        # Get current prices for signal symbols
        signal_symbols = [s['symbol'] for s in actionable_signals]
        prices = price_fetcher.get_multiple_prices(signal_symbols) if signal_symbols else {}
        
        return jsonify({
            'success': True,
            'dashboard': {
                'total_symbols': len(analysis_cache),
                'actionable_signals': len(actionable_signals),
                'signals': actionable_signals,
                'prices': prices,
                'last_analysis': last_analysis_time.isoformat() if last_analysis_time else None,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'success': True,
        'status': {
            'analysis_ready': bool(analysis_cache),
            'signals_ready': bool(signals_cache),
            'symbols_count': len(analysis_cache),
            'signals_count': len(signals_cache) if signals_cache else 0,
            'actionable_signals': len([s for s in signals_cache if s.get('signal', {}).get('action') != 'NEUTRAL']) if signals_cache else 0,
            'last_analysis': last_analysis_time.isoformat() if last_analysis_time else None,
            'server_time': datetime.now().isoformat(),
            'next_signal_update': 'Every 5 minutes',
            'next_analysis_update': 'Every hour'
        }
    })

if __name__ == '__main__':
    # Start background tasks
    analysis_thread = threading.Thread(target=background_analysis, daemon=True)
    signals_thread = threading.Thread(target=background_signals, daemon=True)
    
    analysis_thread.start()
    signals_thread.start()
    
    print("Starting Binance Futures Signal System...")
    print("Available endpoints:")
    print("- GET /api/top-coins - Get top 20 highest volume USDT futures")
    print("- GET /api/analysis - Get full analysis")
    print("- GET /api/analysis/<symbol> - Get symbol analysis")  
    print("- GET /api/signals - Get all signals")
    print("- GET /api/signals/<symbol> - Get symbol signal")
    print("- GET /api/price/<symbol> - Get market data")
    print("- GET /api/prices?symbols=BTC,ETH - Get multiple prices")
    print("- GET /api/dashboard - Get dashboard data")
    print("- GET /api/status - Get system status")
    
    app.run(host='0.0.0.0', port=5000, debug=False)