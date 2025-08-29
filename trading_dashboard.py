"""
Real-time Trading Dashboard with Multi-Timeframe Signals
"""
import time
from datetime import datetime
from multi_timeframe_predictor import MultiTimeframePredictor, SignalType

class TradingDashboard:
    def __init__(self):
        self.predictor = MultiTimeframePredictor()
        self.last_signals = []
        self.signal_history = []
    
    def run_dashboard(self, update_interval_minutes=30):
        """Run continuous trading dashboard"""
        print("🚀 Starting Trading Dashboard...")
        print("=" * 60)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                print(f"\n📊 UPDATE #{iteration} | {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 40)
                
                # Quick analysis (faster updates)
                if iteration == 1 or iteration % 4 == 0:  # Full analysis every 4 iterations
                    success = self.predictor.fetch_multi_timeframe_data()
                    if success:
                        self.predictor.analyze_all_timeframes()
                        signals = self.predictor.generate_trading_signals()
                    else:
                        print("❌ Data fetch failed, using cached analysis")
                        signals = self.last_signals
                else:
                    # Use cached analysis with current price update
                    signals = self.last_signals
                
                # Display compact update
                self.display_compact_update()
                
                # Check for signal changes
                self.check_signal_changes(signals)
                
                # Store signals
                self.last_signals = signals
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'signals': len(signals),
                    'primary_action': signals[0].signal_type.value if signals else 'HOLD'
                })
                
                # Wait for next update
                print(f"⏰ Next update in {update_interval_minutes} minutes...")
                time.sleep(update_interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in dashboard: {e}")
                print("🔄 Retrying in 1 minute...")
                time.sleep(60)
    
    def display_compact_update(self):
        """Display compact trading update"""
        if not self.predictor.timeframe_analyses:
            print("📊 Initializing analysis...")
            return
        
        current_price = self.predictor._get_current_price()
        
        # Timeframe summary
        print(f"💰 BTC: ${current_price:,.0f}")
        
        tf_summary = []
        for tf, analysis in self.predictor.timeframe_analyses.items():
            emoji = {"bullish": "📈", "bearish": "📉"}.get(analysis.trend_direction, "➡️")
            tf_summary.append(f"{tf}:{emoji}{analysis.predicted_change:+.1%}")
        
        print(f"📊 " + " | ".join(tf_summary))
        
        # Current signals
        if self.last_signals:
            primary_signal = self.last_signals[0]
            signal_emoji = {
                SignalType.STRONG_BUY: "🟢🟢",
                SignalType.BUY: "🟢",
                SignalType.SELL: "🔴",
                SignalType.STRONG_SELL: "🔴🔴",
                SignalType.HOLD: "🟡"
            }.get(primary_signal.signal_type, "❓")
            
            print(f"🎯 {signal_emoji} {primary_signal.signal_type.value} "
                  f"(${primary_signal.target_price:,.0f} target, "
                  f"{primary_signal.confidence:.0%} conf)")
    
    def check_signal_changes(self, new_signals):
        """Check for important signal changes"""
        if not self.last_signals or not new_signals:
            return
        
        old_primary = self.last_signals[0].signal_type
        new_primary = new_signals[0].signal_type
        
        # Alert on signal change
        if old_primary != new_primary:
            print(f"🚨 SIGNAL CHANGE: {old_primary.value} → {new_primary.value}")
            
        # Alert on high confidence signals
        for signal in new_signals:
            if signal.confidence > 0.85:
                print(f"⚡ HIGH CONFIDENCE: {signal.signal_type.value} at {signal.confidence:.0%}")

def quick_signal_check():
    """Quick one-time signal check"""
    print("⚡ Quick Signal Check...")
    
    predictor = MultiTimeframePredictor()
    
    # Quick data fetch (reduced for speed)
    print("📊 Fetching current market data...")
    success = predictor.fetch_multi_timeframe_data()
    
    if success:
        predictor.analyze_all_timeframes()
        signals = predictor.generate_trading_signals()
        summary = predictor.get_signal_summary()
        
        current_price = predictor._get_current_price()
        
        print(f"\n💰 BTC: ${current_price:,.2f}")
        print(f"🎯 Signal: {summary['primary_action']} (Conf: {summary['confidence']:.0%})")
        print(f"📈 Target: ${summary['target_price']:,.0f} | 🛑 Stop: ${summary['stop_loss']:,.0f}")
        print(f"💡 Reason: {summary['reason']}")
        
        return summary
    else:
        print("❌ Failed to fetch data")
        return None

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Quick Signal Check (30 seconds)")
    print("2. Full Dashboard (continuous)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_signal_check()
    else:
        dashboard = TradingDashboard()
        dashboard.run_dashboard(update_interval_minutes=15)  # Update every 15 minutes