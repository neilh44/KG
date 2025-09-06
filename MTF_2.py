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
import time

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

@dataclass
class RelationshipTracker:
    relationship: str
    timeframe: str
    iterations_seen: int
    avg_strength: float
    last_strength: float
    first_seen: int
    last_seen: int
    strength_history: List[float]

class MultiTimeframePredictor:
    def __init__(self):
        self.llm_processor = LLMProcessor()
        self.knowledge_graph = CryptoKnowledgeGraph()
        self.causal_ai = CausalAI()
        
        self.timeframe_data = {}
        self.timeframe_analyses = {}
        self.trading_signals = []
        self.causal_history = []
        self.relationship_trackers = {}  # Track relationship persistence
        self.iteration = 0
        self.market_regime = "unknown"
        
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
            
            # Track and display causal relationships
            if relationships:
                print(f"   🔗 Causal Relationships in {tf}:")
                for i, rel in enumerate(relationships[:3], 1):
                    print(f"      {i}. {rel['cause']} → {rel['effect']} (Strength: {rel['strength']:.3f})")
                    # Store in history and update trackers
                    self.causal_history.append({
                        'iteration': self.iteration,
                        'timeframe': tf,
                        'relationship': f"{rel['cause']} → {rel['effect']}",
                        'strength': rel['strength'],
                        'timestamp': datetime.now()
                    })
                    self._update_relationship_tracker(tf, f"{rel['cause']} → {rel['effect']}", rel['strength'])
    
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
        """Generate signals based on relationship persistence and strength"""
        signals = []
        
        for tf, analysis in self.timeframe_analyses.items():
            if analysis.causal_relationships > 0:
                # Get relationship persistence score
                persistence_score = self._get_persistence_score(tf)
                normalized_strength = self._normalize_strength(tf, analysis.predicted_change)
                
                if abs(normalized_strength) > 0.005 and persistence_score >= 2:  # 2+ iterations
                    
                    # Dynamic targets based on persistence and confluence
                    confluence_bonus = self._get_confluence_bonus()
                    target_adj = min(abs(normalized_strength) * (1 + persistence_score * 0.1), 0.05)
                    stop_adj = 0.015 - (persistence_score * 0.002)  # Tighter stops for persistent signals
                    
                    if normalized_strength > 0:
                        signal_type = SignalType.STRONG_BUY if persistence_score >= 3 else SignalType.BUY
                        target_price = current_price * (1 + target_adj + confluence_bonus)
                        stop_loss = current_price * (1 - stop_adj)
                        
                    else:
                        signal_type = SignalType.STRONG_SELL if persistence_score >= 3 else SignalType.SELL
                        target_price = current_price * (1 - target_adj - confluence_bonus)
                        stop_loss = current_price * (1 + stop_adj)
                    
                    reason = f"{tf} causal (persist:{persistence_score}, strength:{normalized_strength:.2%})"
                    
                    signal = TradingSignal(
                        signal_type=signal_type,
                        confidence=min(analysis.confidence + (persistence_score * 0.05), 0.95),
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
        
        # Show causal history if available
        if len(self.causal_history) > 1:
            print(f"\n📈 CAUSAL EVOLUTION (Last 3 iterations):")
            recent_history = self.causal_history[-6:]  # Last 6 entries (2 per iteration max)
            for entry in recent_history:
                print(f"   [{entry['iteration']}] {entry['timeframe']}: {entry['relationship']} ({entry['strength']:+.3f})")
        
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
            "total_signals": len(self.trading_signals),
            "market_regime": self.market_regime,
            "persistence_score": self._get_max_persistence_score()
        }
    
    def _update_relationship_tracker(self, timeframe: str, relationship: str, strength: float):
        """Update relationship persistence tracking"""
        key = f"{timeframe}_{relationship}"
        
        if key in self.relationship_trackers:
            tracker = self.relationship_trackers[key]
            tracker.iterations_seen += 1
            tracker.last_strength = strength
            tracker.last_seen = self.iteration
            tracker.strength_history.append(strength)
            tracker.avg_strength = np.mean(tracker.strength_history[-3:])  # Rolling avg
        else:
            self.relationship_trackers[key] = RelationshipTracker(
                relationship=relationship,
                timeframe=timeframe,
                iterations_seen=1,
                avg_strength=strength,
                last_strength=strength,
                first_seen=self.iteration,
                last_seen=self.iteration,
                strength_history=[strength]
            )
    
    def _get_persistence_score(self, timeframe: str) -> int:
        """Get max persistence score for timeframe"""
        tf_trackers = [t for k, t in self.relationship_trackers.items() if k.startswith(timeframe)]
        return max([t.iterations_seen for t in tf_trackers], default=0)
    
    def _get_max_persistence_score(self) -> int:
        """Get maximum persistence score across all relationships"""
        return max([t.iterations_seen for t in self.relationship_trackers.values()], default=0)
    
    def _normalize_strength(self, timeframe: str, predicted_change: float) -> float:
        """Normalize strength by timeframe characteristics"""
        tf_multipliers = {'1m': 1.0, '5m': 0.8, '15m': 0.6}
        return predicted_change * tf_multipliers.get(timeframe, 1.0)
    
    def _get_confluence_bonus(self) -> float:
        """Calculate confluence bonus for multi-relationship alignment"""
        active_relationships = [t for t in self.relationship_trackers.values() 
                              if t.last_seen == self.iteration]
        
        if len(active_relationships) >= 2:
            # Check if relationships align in direction
            signs = [1 if t.last_strength > 0 else -1 for t in active_relationships]
            if len(set(signs)) == 1:  # All same direction
                return 0.005  # 0.5% bonus for confluence
        return 0
    
    def _classify_market_regime(self):
        """Classify current market regime based on active relationships"""
        active_rels = [t for t in self.relationship_trackers.values() if t.last_seen == self.iteration]
        
        if not active_rels:
            self.market_regime = "neutral"
            return
        
        funding_driven = any("funding_rate" in t.relationship for t in active_rels)
        volume_driven = any("volume" in t.relationship for t in active_rels)
        
        if funding_driven and volume_driven:
            self.market_regime = "mixed_leadership"
        elif funding_driven:
            self.market_regime = "derivatives_led"
        elif volume_driven:
            self.market_regime = "spot_led"
        else:
            self.market_regime = "other_factors"
    
    def run_dynamic_analysis(self, iterations: int = float('inf')):
        """Run dynamic analysis with 60-second intervals until stopped"""
        print("🔄 Starting Continuous Dynamic Causal Analysis...")
        print(f"⏰ Running continuously with 60-second intervals")
        print("📝 Press Ctrl+C to stop\n")
        
        try:
            while self.iteration < iterations:
                self.iteration += 1
                print(f"\n🔄 ITERATION {self.iteration} | {datetime.now().strftime('%H:%M:%S')}")
                print("-" * 50)
                
                # Fetch and analyze
                if self.fetch_multi_timeframe_data():
                    self.analyze_all_timeframes()
                    self._classify_market_regime()
                    signals = self.generate_trading_signals()
                    
                    # Feed signals back into system for learning
                    self._process_signal_feedback(signals)
                    
                    # Enhanced summary with regime and persistence
                    summary = self.get_signal_summary()
                    print(f"\n⚡ QUICK UPDATE:")
                    print(f"   Action: {summary['primary_action']} | Confidence: {summary['confidence']:.0%}")
                    print(f"   Target: ${summary['target_price']:,.0f} | Stop: ${summary['stop_loss']:,.0f}")
                    print(f"   Regime: {summary['market_regime']} | Max Persist: {summary['persistence_score']}")
                    
                    # Show relationship rotations
                    self._display_relationship_status()
                    
                    print(f"\n⏳ Waiting 60 seconds for next iteration...")
                    time.sleep(60)
                else:
                    print("❌ Data fetch failed, retrying in 60 seconds...")
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            print(f"\n\n🛑 Analysis stopped by user after {self.iteration} iterations")
            print(f"\n🏁 FINAL ANALYSIS:")
            self.display_multi_timeframe_analysis()
            self._display_persistence_summary()
    
    def _display_relationship_status(self):
        """Display current relationship persistence status"""
        persistent_rels = [t for t in self.relationship_trackers.values() if t.iterations_seen >= 2]
        if persistent_rels:
            print(f"   🔄 Persistent: {', '.join([f'{t.relationship}({t.iterations_seen})' for t in persistent_rels])}")
    
    def _display_persistence_summary(self):
        """Display final persistence analysis"""
        print(f"\n🏆 PERSISTENCE CHAMPIONS:")
        sorted_trackers = sorted(self.relationship_trackers.values(), 
                               key=lambda x: x.iterations_seen, reverse=True)
        
        for tracker in sorted_trackers[:5]:
            consistency = "📈" if len(set(np.sign(tracker.strength_history))) == 1 else "📊"
            print(f"   {consistency} {tracker.relationship} ({tracker.timeframe}): "
                  f"{tracker.iterations_seen} iterations | Avg: {tracker.avg_strength:+.3f}")
        
        print(f"\n🎯 REGIME EVOLUTION: {self.market_regime}")
        print(f"💡 RECOMMENDATION: Focus on {sorted_trackers[0].timeframe} timeframe" if sorted_trackers else "💡 No persistent patterns found")
    
    def _process_signal_feedback(self, signals: List[TradingSignal]):
        """Process trading signals through simplified feedback loop"""
        if not signals:
            return
            
        # Create signal data for analysis
        signal_data = self._prepare_signal_data(signals)
        
        # Simple signal analysis (replace complex LLM call)
        signal_insights = self._analyze_signals_simple(signal_data)
        
        # Update knowledge graph equivalent (track signal patterns)
        self._update_signal_patterns(signal_data, signal_insights)
        
        # Simple causal prediction (replace complex causal AI call)
        outcome_prediction = self._predict_signal_outcome(signal_data)
        
        # Display feedback results
        self._display_feedback_results(outcome_prediction)
    
    def _analyze_signals_simple(self, signal_data: Dict) -> Dict:
        """Simple signal analysis replacement for LLM"""
        signals = signal_data['signals']
        context = signal_data['market_context']
        
        # Calculate signal quality metrics
        avg_confidence = np.mean([s['confidence'] for s in signals])
        signal_consistency = len(set([s['type'] for s in signals])) == 1  # All same direction
        
        return {
            'signal_quality': avg_confidence,
            'signal_consistency': signal_consistency,
            'regime_alignment': context['regime'] in ['spot_led', 'derivatives_led'],
            'persistence_factor': context['persistent_relationships']
        }
    
    def _update_signal_patterns(self, signal_data: Dict, insights: Dict):
        """Update signal pattern tracking (Knowledge Graph equivalent)"""
        # Store signal outcome predictions for learning
        if not hasattr(self, 'signal_patterns'):
            self.signal_patterns = []
        
        pattern = {
            'iteration': self.iteration,
            'regime': signal_data['market_context']['regime'],
            'primary_signal': signal_data['signals'][0]['type'] if signal_data['signals'] else 'NONE',
            'confidence': signal_data['signals'][0]['confidence'] if signal_data['signals'] else 0,
            'quality_score': insights['signal_quality'],
            'consistency': insights['signal_consistency'],
            'causal_count': len(signal_data['causal_context'])
        }
        
        self.signal_patterns.append(pattern)
    
    def _predict_signal_outcome(self, signal_data: Dict) -> Dict:
        """Predict signal outcome using historical patterns (Causal AI equivalent)"""
        if not hasattr(self, 'signal_patterns') or len(self.signal_patterns) < 3:
            return {'predicted_outcome': 'Learning...', 'success_prob': 50}
        
        # Analyze historical patterns
        recent_patterns = self.signal_patterns[-10:]  # Last 10 iterations
        current_regime = signal_data['market_context']['regime']
        
        # Find similar regime patterns
        similar_patterns = [p for p in recent_patterns if p['regime'] == current_regime]
        
        if similar_patterns:
            avg_quality = np.mean([p['quality_score'] for p in similar_patterns])
            consistency_rate = np.mean([p['consistency'] for p in similar_patterns])
            
            # Predict based on historical performance
            success_prob = (avg_quality * 0.7 + consistency_rate * 0.3) * 100
            
            if success_prob > 75:
                outcome = "High probability success"
            elif success_prob > 60:
                outcome = "Moderate success expected"
            else:
                outcome = "Caution - mixed signals"
        else:
            # New regime - use current metrics
            signals = signal_data['signals']
            if signals:
                success_prob = signals[0]['confidence'] * 100
                outcome = "New regime pattern"
            else:
                success_prob = 50
                outcome = "No clear pattern"
        
        return {
            'predicted_outcome': outcome,
            'success_prob': min(success_prob, 95),
            'recommended_adjustment': self._get_signal_recommendation(success_prob)
        }
    
    def _get_signal_recommendation(self, success_prob: float) -> str:
        """Get recommendation based on success probability"""
        if success_prob > 80:
            return "Scale up position size"
        elif success_prob > 65:
            return "Standard position size"
        elif success_prob > 50:
            return "Reduce position size"
        else:
            return "Consider waiting for better setup"
    
    def _prepare_signal_data(self, signals: List[TradingSignal]) -> Dict:
        """Prepare signal data for feedback processing"""
        current_price = self._get_current_price()
        
        return {
            'signals': [
                {
                    'type': s.signal_type.value,
                    'confidence': s.confidence,
                    'timeframe': s.timeframe,
                    'entry_price': s.entry_price,
                    'target_price': s.target_price,
                    'stop_loss': s.stop_loss,
                    'expected_return': (s.target_price - s.entry_price) / s.entry_price
                } for s in signals
            ],
            'market_context': {
                'current_price': current_price,
                'regime': self.market_regime,
                'iteration': self.iteration,
                'persistent_relationships': len([t for t in self.relationship_trackers.values() if t.iterations_seen >= 3])
            },
            'causal_context': [
                {
                    'relationship': t.relationship,
                    'timeframe': t.timeframe,
                    'persistence': t.iterations_seen,
                    'avg_strength': t.avg_strength
                } for t in self.relationship_trackers.values() if t.last_seen == self.iteration
            ]
        }
    
    def _display_feedback_results(self, causal_results: Dict):
        """Display results from feedback loop"""
        if causal_results and 'predicted_outcome' in causal_results:
            print(f"   🧠 AI Prediction: {causal_results['predicted_outcome']}")
            print(f"   📊 Success Probability: {causal_results.get('success_prob', 0):.0%}")
            
            if 'recommended_adjustment' in causal_results:
                print(f"   ⚙️ Adjustment: {causal_results['recommended_adjustment']}")

def main():
    """Main execution for continuous dynamic analysis"""
    predictor = MultiTimeframePredictor()
    
    # Run continuous analysis (until Ctrl+C)
    predictor.run_dynamic_analysis()
    
    return predictor

if __name__ == "__main__":
    predictor = main()