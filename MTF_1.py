"""
Multi-Timeframe Predictor with Trading Signals
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from llm_processor import LLMProcessor
from knowledge_graph import CryptoKnowledgeGraph  
from causal_ai import CausalAI

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    signal_type: SignalType
    confidence: float
    reason: str
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str
    timestamp: datetime

@dataclass 
class TimeframeAnalysis:
    timeframe: str
    data_points: int
    causal_relationships: int
    predicted_change: float
    confidence: float
    trend_direction: str
    key_insights: List[str]

class MultiTimeframePredictor:
    def __init__(self):
        self.llm_processor = LLMProcessor()
        self.knowledge_graph = CryptoKnowledgeGraph()
        self.causal_ai = CausalAI()
        
        self.timeframe_data = {}
        self.timeframe_analyses = {}
        self.trading_signals = []
        
    def fetch_multi_timeframe_data(self):
        """Fetch data across multiple timeframes"""
        timeframes = {
            '1m': {'hours': 17, 'desc': 'Short-term (1m intervals)'},
            '5m': {'hours': 83, 'desc': 'Medium-term (5m intervals)'}, 
            '15m': {'hours': 250, 'desc': 'Long-term (15m intervals)'}
        }
        
        print("🔄 Fetching multi-timeframe data...")
        
        for tf, config in timeframes.items():
            print(f"\n📊 {config['desc']}")
            data = self._fetch_timeframe_data(tf, config['hours'])
            
            if data is not None:
                self.timeframe_data[tf] = data
                print(f"✅ {tf}: {len(data)} data points")
            else:
                print(f"❌ {tf}: Failed to fetch data")
        
        return len(self.timeframe_data) > 0
    
    def _fetch_timeframe_data(self, interval: str, hours: int) -> Optional[pd.DataFrame]:
        """Fetch data for specific timeframe"""
        if interval == '1m':
            limit = min(hours * 60, 1000)
        elif interval == '5m':
            limit = min(hours * 12, 1000)
        elif interval == '15m':
            limit = min(hours * 4, 1000)
        else:
            limit = 100
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
        except Exception as e:
            print(f"Error fetching {interval} data: {e}")
            return None
    
    def analyze_all_timeframes(self):
        """Analyze causal relationships across all timeframes"""
        print("\n🧠 Multi-Timeframe Causal Analysis...")
        
        for tf, data in self.timeframe_data.items():
            print(f"\n📊 Analyzing {tf} timeframe...")
            
            # Create market snapshots
            snapshots = self._create_snapshots(data, tf)
            
            # Discover causal relationships for this timeframe
            causal_ai = CausalAI()  # Fresh instance for each timeframe
            relationships = causal_ai.discover_causal_relationships(snapshots)
            
            # Analyze current market structure
            current_data = self._create_current_snapshot(data)
            insights = self.llm_processor.analyze_market_structure(current_data)
            
            # Create timeframe analysis
            analysis = TimeframeAnalysis(
                timeframe=tf,
                data_points=len(data),
                causal_relationships=len(relationships),
                predicted_change=self._calculate_predicted_change(insights, relationships),
                confidence=self._calculate_timeframe_confidence(insights, relationships),
                trend_direction=insights['trend_strength']['direction'],
                key_insights=self._extract_key_insights(relationships, insights, tf)
            )
            
            self.timeframe_analyses[tf] = analysis
            print(f"✅ {tf}: {len(relationships)} causal relationships, {analysis.confidence:.1%} confidence")
            
            # Display causal relationship details
            if relationships:
                print(f"   🔗 Causal Relationships in {tf}:")
                for i, rel in enumerate(relationships[:3], 1):  # Show top 3
                    print(f"      {i}. {rel['cause']} → {rel['effect']} (Strength: {rel['strength']:.3f})")
    
    def _create_snapshots(self, data: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Create market snapshots from timeframe data"""
        chunk_size = {'1m': 1440, '5m': 144, '15m': 48}.get(timeframe, 24)
        snapshots = []
        
        for i in range(chunk_size, len(data), max(1, chunk_size//4)):
            chunk = data.iloc[max(0, i-chunk_size):i]
            if len(chunk) >= chunk_size//2:
                snapshot = {
                    'timestamp': chunk.iloc[-1]['timestamp'],
                    'ohlcv': chunk,
                    'funding': {'funding_rate': np.random.normal(0.0001, 0.0002)},
                    'order_book': {'bids': [[0, 0]], 'asks': [[0, 0]]},
                    'open_interest': {'open_interest': 1000000}
                }
                snapshots.append(snapshot)
        
        return snapshots
    
    def _create_current_snapshot(self, data: pd.DataFrame) -> Dict:
        """Create current market snapshot"""
        recent_data = data.tail(24)
        return {
            'timestamp': datetime.now(),
            'ohlcv': recent_data,
            'funding': {'funding_rate': 0.0001},
            'order_book': {'bids': [[0, 0]], 'asks': [[0, 0]]},
            'open_interest': {'open_interest': 1000000}
        }
    
    def _calculate_predicted_change(self, insights: Dict, relationships: List) -> float:
        """Calculate predicted price change for timeframe"""
        base_change = {'bullish': 0.02, 'bearish': -0.02, 'neutral': 0}.get(
            insights['trend_strength']['direction'], 0
        )
        
        # Adjust based on causal relationships (with realistic scaling)
        causal_adjustment = 0
        for rel in relationships:
            if abs(rel['strength']) > 0.1:
                if 'price' in rel['effect']:
                    # Scale causal effect to realistic range
                    scaled_effect = np.tanh(rel['strength']) * 0.01  # Max 1% adjustment
                    causal_adjustment += scaled_effect
        
        # Clamp total change to realistic range
        total_change = base_change + causal_adjustment
        return np.clip(total_change, -0.10, 0.10)  # Max 10% change
    
    def _calculate_timeframe_confidence(self, insights: Dict, relationships: List) -> float:
        """Calculate confidence for timeframe analysis"""
        base_confidence = 0.6
        
        # Boost confidence with strong relationships
        strong_rels = [r for r in relationships if abs(r['strength']) > 0.1]
        base_confidence += len(strong_rels) * 0.05
        
        # Boost confidence with strong trends
        if insights['trend_strength']['strength'] == 'strong':
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _extract_key_insights(self, relationships: List, insights: Dict, timeframe: str) -> List[str]:
        """Extract key insights for timeframe"""
        insights_list = []
        
        # Strong causal relationships
        strong_rels = [r for r in relationships if abs(r['strength']) > 0.1]
        if strong_rels:
            insights_list.append(f"{len(strong_rels)} strong causal relationships in {timeframe}")
        
        # Trend insights
        trend = insights['trend_strength']
        insights_list.append(f"{trend['strength']} {trend['direction']} trend on {timeframe}")
        
        # Market regime
        regime = insights.get('market_regime', 'unknown')
        insights_list.append(f"Market regime: {regime} ({timeframe})")
        
        return insights_list
    
    def generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on multi-timeframe causal analysis"""
        print("\n🎯 Generating Trading Signals...")
        
        signals = []
        current_price = self._get_current_price()
        
        # Multi-timeframe confluence analysis
        confluence_signal = self._analyze_timeframe_confluence()
        
        # Causal-based signals
        causal_signals = self._generate_causal_signals(current_price)
        
        # Combine signals
        all_signals = [confluence_signal] + causal_signals
        signals.extend([s for s in all_signals if s is not None])
        
        self.trading_signals = signals
        return signals
    
    def _get_current_price(self) -> float:
        """Get current BTC price"""
        if '1m' in self.timeframe_data:
            return float(self.timeframe_data['1m']['close'].iloc[-1])
        return 0.0
    
    def _analyze_timeframe_confluence(self) -> Optional[TradingSignal]:
        """Analyze confluence across timeframes"""
        if len(self.timeframe_analyses) < 2:
            return None
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for a in self.timeframe_analyses.values() 
                           if a.trend_direction == 'bullish')
        bearish_count = sum(1 for a in self.timeframe_analyses.values()
                           if a.trend_direction == 'bearish')
        
        total_confidence = np.mean([a.confidence for a in self.timeframe_analyses.values()])
        current_price = self._get_current_price()
        
        # Determine signal based on confluence
        if bullish_count > bearish_count and bullish_count >= 2:
            signal_type = SignalType.STRONG_BUY if bullish_count == 3 else SignalType.BUY
            target_price = current_price * 1.03
            stop_loss = current_price * 0.98
            reason = f"Multi-timeframe confluence: {bullish_count}/3 timeframes bullish"
            
        elif bearish_count > bullish_count and bearish_count >= 2:
            signal_type = SignalType.STRONG_SELL if bearish_count == 3 else SignalType.SELL
            target_price = current_price * 0.97
            stop_loss = current_price * 1.02
            reason = f"Multi-timeframe confluence: {bearish_count}/3 timeframes bearish"
            
        else:
            signal_type = SignalType.HOLD
            target_price = current_price
            stop_loss = current_price
            reason = "Mixed signals across timeframes - hold position"
        
        return TradingSignal(
            signal_type=signal_type,
            confidence=total_confidence,
            reason=reason,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe="Multi-TF",
            timestamp=datetime.now()
        )
    
    def _generate_causal_signals(self, current_price: float) -> List[TradingSignal]:
        """Generate signals based on specific causal relationships"""
        signals = []
        
        for tf, analysis in self.timeframe_analyses.items():
            # Use realistic thresholds for signal generation
            if analysis.causal_relationships > 0 and abs(analysis.predicted_change) > 0.005:  # 0.5% threshold
                
                if analysis.predicted_change > 0.005:
                    signal_type = SignalType.BUY
                    target_price = current_price * (1 + min(analysis.predicted_change, 0.05))  # Max 5% target
                    stop_loss = current_price * 0.985
                    reason = f"Causal analysis on {tf}: {analysis.predicted_change:.2%} upside expected"
                    
                elif analysis.predicted_change < -0.005:
                    signal_type = SignalType.SELL  
                    target_price = current_price * (1 + max(analysis.predicted_change, -0.05))  # Max 5% downside
                    stop_loss = current_price * 1.015
                    reason = f"Causal analysis on {tf}: {analysis.predicted_change:.2%} downside expected"
                    
                else:
                    continue
                
                signal = TradingSignal(
                    signal_type=signal_type,
                    confidence=analysis.confidence,
                    reason=reason,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    timeframe=tf,
                    timestamp=datetime.now()
                )
                
                signals.append(signal)
        
        return signals
    
    def display_multi_timeframe_analysis(self):
        """Display comprehensive multi-timeframe analysis"""
        print("\n" + "="*80)
        print("📊 MULTI-TIMEFRAME CAUSAL ANALYSIS")
        print("="*80)
        
        current_price = self._get_current_price()
        print(f"💰 Current BTC Price: ${current_price:,.2f}")
        
        # Timeframe analysis summary
        print(f"\n📈 Timeframe Analysis:")
        for tf, analysis in self.timeframe_analyses.items():
            trend_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(analysis.trend_direction, "❓")
            print(f"   {trend_emoji} {tf.upper()}: {analysis.trend_direction} | "
                  f"Change: {analysis.predicted_change:+.2%} | "
                  f"Conf: {analysis.confidence:.0%} | "
                  f"Causal: {analysis.causal_relationships}")
        
        # Trading signals
        print(f"\n🎯 TRADING SIGNALS ({len(self.trading_signals)} signals)")
        print("-" * 80)
        
        for i, signal in enumerate(self.trading_signals, 1):
            signal_emoji = {
                SignalType.STRONG_BUY: "🟢🟢",
                SignalType.BUY: "🟢",
                SignalType.SELL: "🔴", 
                SignalType.STRONG_SELL: "🔴🔴",
                SignalType.HOLD: "🟡"
            }.get(signal.signal_type, "❓")
            
            print(f"{i}. {signal_emoji} {signal.signal_type.value} | {signal.timeframe}")
            print(f"   💡 {signal.reason}")
            print(f"   📊 Entry: ${signal.entry_price:,.0f} | Target: ${signal.target_price:,.0f} | "
                  f"Stop: ${signal.stop_loss:,.0f}")
            print(f"   ✅ Confidence: {signal.confidence:.0%}")
            
            if i < len(self.trading_signals):
                print()
        
        # Key insights across timeframes
        print(f"\n🧠 KEY INSIGHTS:")
        all_insights = []
        for analysis in self.timeframe_analyses.values():
            all_insights.extend(analysis.key_insights)
        
        for insight in all_insights[:5]:  # Top 5 insights
            print(f"   • {insight}")
        
        # Display detailed causal relationships
        print(f"\n🔗 CAUSAL RELATIONSHIPS SUMMARY:")
        for tf, analysis in self.timeframe_analyses.items():
            if analysis.causal_relationships > 0:
                print(f"   {tf.upper()}: {analysis.causal_relationships} relationships found")
        
        print("="*80)
    
    def get_signal_summary(self) -> Dict:
        """Get actionable signal summary"""
        if not self.trading_signals:
            return {"action": "NO_SIGNALS", "confidence": 0}
        
        # Find strongest signal
        strongest_signal = max(self.trading_signals, key=lambda x: x.confidence)
        
        # Calculate overall sentiment
        buy_signals = len([s for s in self.trading_signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]])
        sell_signals = len([s for s in self.trading_signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]])
        
        return {
            "primary_action": strongest_signal.signal_type.value,
            "confidence": strongest_signal.confidence,
            "entry_price": strongest_signal.entry_price,
            "target_price": strongest_signal.target_price,
            "stop_loss": strongest_signal.stop_loss,
            "reason": strongest_signal.reason,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_signals": len(self.trading_signals)
        }

def main():
    """Main execution for multi-timeframe analysis"""
    predictor = MultiTimeframePredictor()
    
    # Fetch multi-timeframe data
    print("🚀 Starting Multi-Timeframe Analysis...")
    
    success = predictor.fetch_multi_timeframe_data()
    if not success:
        print("❌ Failed to fetch multi-timeframe data")
        return
    
    # Analyze all timeframes
    predictor.analyze_all_timeframes()
    
    # Generate trading signals
    signals = predictor.generate_trading_signals()
    
    # Display comprehensive analysis
    predictor.display_multi_timeframe_analysis()
    
    # Get actionable summary
    summary = predictor.get_signal_summary()
    
    print(f"\n🎯 ACTIONABLE SUMMARY:")
    print(f"Primary Action: {summary['primary_action']} (Confidence: {summary['confidence']:.0%})")
    print(f"Entry: ${summary['entry_price']:,.0f} | Target: ${summary['target_price']:,.0f}")
    
    return predictor, signals

if __name__ == "__main__":
    predictor, signals = main()