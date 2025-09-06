"""
Trading Signal Validation and Feedback System
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

@dataclass
class SignalOutcome:
    signal_id: str
    predicted_change: float
    actual_change: float
    accuracy: float
    profit_loss: float
    hit_target: bool
    hit_stop: bool
    duration_minutes: int

@dataclass
class ValidationResult:
    total_signals: int
    successful_signals: int
    accuracy_rate: float
    avg_profit_loss: float
    best_timeframe: str
    insights: List[str]

class SignalValidator:
    def __init__(self, predictor):
        self.predictor = predictor
        self.signal_history = []
        self.outcomes = []
        self.validation_window = 60  # minutes to wait for outcome
        
    def validate_signals(self, signals: List) -> ValidationResult:
        """Compare trading signals with actual price movements"""
        print("\n🔍 SIGNAL VALIDATION")
        print("-" * 40)
        
        current_price = self.predictor._get_current_price()
        outcomes = []
        
        # Store signals for future validation
        for signal in signals:
            signal_record = {
                'id': f"{signal.timeframe}_{self.predictor.iteration}",
                'signal': signal,
                'start_price': current_price,
                'start_time': datetime.now(),
                'validated': False
            }
            self.signal_history.append(signal_record)
        
        # Validate older signals that have had time to play out
        for record in self.signal_history:
            if not record['validated'] and self._should_validate(record):
                outcome = self._calculate_outcome(record, current_price)
                if outcome:
                    outcomes.append(outcome)
                    record['validated'] = True
        
        self.outcomes.extend(outcomes)
        
        # Generate validation summary
        if outcomes:
            return self._create_validation_result(outcomes)
        else:
            return ValidationResult(0, 0, 0.0, 0.0, "", ["No signals ready for validation"])
    
    def _should_validate(self, record: Dict) -> bool:
        """Check if signal is ready for validation"""
        elapsed = (datetime.now() - record['start_time']).total_seconds() / 60
        return elapsed >= self.validation_window
    
    def _calculate_outcome(self, record: Dict, current_price: float) -> Optional[SignalOutcome]:
        """Calculate actual outcome vs predicted"""
        signal = record['signal']
        start_price = record['start_price']
        
        if start_price == 0:
            return None
        
        actual_change = (current_price - start_price) / start_price
        predicted_change = (signal.target_price - signal.entry_price) / signal.entry_price
        
        # Check if target/stop was hit
        if signal.signal_type.value in ['BUY', 'STRONG_BUY']:
            hit_target = current_price >= signal.target_price
            hit_stop = current_price <= signal.stop_loss
            profit_loss = actual_change
        else:
            hit_target = current_price <= signal.target_price
            hit_stop = current_price >= signal.stop_loss
            profit_loss = -actual_change
        
        accuracy = 1.0 - abs(actual_change - predicted_change) / abs(predicted_change) if predicted_change != 0 else 0.5
        
        return SignalOutcome(
            signal_id=record['id'],
            predicted_change=predicted_change,
            actual_change=actual_change,
            accuracy=max(0, accuracy),
            profit_loss=profit_loss,
            hit_target=hit_target,
            hit_stop=hit_stop,
            duration_minutes=self.validation_window
        )
    
    def _create_validation_result(self, outcomes: List[SignalOutcome]) -> ValidationResult:
        """Create validation summary"""
        successful = sum(1 for o in outcomes if o.hit_target or o.profit_loss > 0)
        accuracy_rate = successful / len(outcomes)
        avg_pnl = np.mean([o.profit_loss for o in outcomes])
        
        # Find best performing timeframe
        tf_performance = {}
        for outcome in outcomes:
            tf = outcome.signal_id.split('_')[0]
            if tf not in tf_performance:
                tf_performance[tf] = []
            tf_performance[tf].append(outcome.profit_loss)
        
        best_tf = max(tf_performance.keys(), 
                     key=lambda x: np.mean(tf_performance[x])) if tf_performance else ""
        
        insights = [
            f"Success rate: {accuracy_rate:.1%}",
            f"Avg P/L: {avg_pnl:+.2%}",
            f"Best timeframe: {best_tf}" if best_tf else "Insufficient data"
        ]
        
        return ValidationResult(
            total_signals=len(outcomes),
            successful_signals=successful,
            accuracy_rate=accuracy_rate,
            avg_profit_loss=avg_pnl,
            best_timeframe=best_tf,
            insights=insights
        )

class FeedbackProcessor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.learning_data = []
        
    def process_feedback(self, signals: List, validation: ValidationResult):
        """Process validation feedback to improve model"""
        print("\n🧠 FEEDBACK PROCESSING")
        print("-" * 40)
        
        # Prepare feedback data
        feedback_data = self._prepare_feedback_data(signals, validation)
        
        # Update knowledge graph with performance data
        self._update_knowledge_graph(feedback_data)
        
        # Run causal analysis on signal performance
        causal_insights = self._analyze_signal_causality(feedback_data)
        
        return self._generate_improved_signals(causal_insights)
    
    def _prepare_feedback_data(self, signals: List, validation: ValidationResult) -> Dict:
        """Prepare comprehensive feedback dataset"""
        return {
            'current_signals': [
                {
                    'timeframe': s.timeframe,
                    'type': s.signal_type.value,
                    'confidence': s.confidence,
                    'predicted_change': (s.target_price - s.entry_price) / s.entry_price
                } for s in signals
            ],
            'validation_results': {
                'accuracy_rate': validation.accuracy_rate,
                'avg_pnl': validation.avg_profit_loss,
                'best_timeframe': validation.best_timeframe,
                'total_validated': validation.total_signals
            },
            'market_context': {
                'regime': self.predictor.market_regime,
                'iteration': self.predictor.iteration,
                'persistent_relationships': len([t for t in self.predictor.relationship_trackers.values() 
                                               if t.iterations_seen >= 3])
            }
        }
    
    def _update_knowledge_graph(self, feedback_data: Dict):
        """Update knowledge graph with performance insights"""
        # Store learning data for pattern recognition
        learning_entry = {
            'iteration': self.predictor.iteration,
            'timestamp': datetime.now(),
            'accuracy': feedback_data['validation_results']['accuracy_rate'],
            'best_timeframe': feedback_data['validation_results']['best_timeframe'],
            'market_regime': feedback_data['market_context']['regime'],
            'signal_count': len(feedback_data['current_signals'])
        }
        
        self.learning_data.append(learning_entry)
        
        # Identify patterns in performance
        if len(self.learning_data) >= 3:
            recent_accuracy = np.mean([d['accuracy'] for d in self.learning_data[-3:]])
            print(f"   📈 Learning Progress: {recent_accuracy:.1%} (3-iteration avg)")
    
    def _analyze_signal_causality(self, feedback_data: Dict) -> Dict:
        """Analyze what causes good/bad signals"""
        causal_insights = {
            'regime_performance': {},
            'timeframe_performance': {},
            'confidence_correlation': 0,
            'recommendations': []
        }
        
        if len(self.learning_data) >= 3:
            # Analyze regime performance
            regimes = {}
            for entry in self.learning_data:
                regime = entry['market_regime']
                if regime not in regimes:
                    regimes[regime] = []
                regimes[regime].append(entry['accuracy'])
            
            for regime, accuracies in regimes.items():
                causal_insights['regime_performance'][regime] = np.mean(accuracies)
            
            # Find best performing regime
            best_regime = max(regimes.keys(), key=lambda x: np.mean(regimes[x]))
            causal_insights['recommendations'].append(f"Best regime: {best_regime}")
        
        return causal_insights
    
    def _generate_improved_signals(self, causal_insights: Dict) -> List:
        """Generate improved signals based on causal analysis"""
        print(f"   🎯 Signal Improvement:")
        
        improved_signals = []
        current_price = self.predictor._get_current_price()
        
        # Use best performing patterns
        best_regime = causal_insights.get('regime_performance', {})
        if best_regime:
            best_regime_name = max(best_regime.keys(), key=lambda x: best_regime[x])
            if self.predictor.market_regime == best_regime_name:
                print(f"   ✅ Current regime ({best_regime_name}) performing well")
                
                # Generate enhanced signal based on learning
                signal = self._create_enhanced_signal(current_price, causal_insights)
                if signal:
                    improved_signals.append(signal)
        
        return improved_signals
    
    def _create_enhanced_signal(self, current_price: float, insights: Dict) -> Optional:
        """Create enhanced signal based on learning"""
        # Find strongest timeframe analysis
        strongest_tf = max(self.predictor.timeframe_analyses.keys(),
                          key=lambda x: self.predictor.timeframe_analyses[x].confidence)
        
        analysis = self.predictor.timeframe_analyses[strongest_tf]
        
        if analysis.confidence > 0.7:  # High confidence threshold
            # Enhanced signal with learned adjustments
            if analysis.predicted_change > 0:
                signal_type = SignalType.STRONG_BUY
                target = current_price * (1 + min(abs(analysis.predicted_change) * 1.2, 0.04))
                stop = current_price * 0.985
            else:
                signal_type = SignalType.STRONG_SELL  
                target = current_price * (1 - min(abs(analysis.predicted_change) * 1.2, 0.04))
                stop = current_price * 1.015
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=min(analysis.confidence + 0.1, 0.95),  # Learning boost
                reason=f"Enhanced {strongest_tf} signal (learned pattern)",
                entry_price=current_price,
                target_price=target,
                stop_loss=stop,
                timeframe=f"{strongest_tf}_enhanced",
                timestamp=datetime.now()
            )
        
        return None

# Enhanced MultiTimeframePredictor with validation
class EnhancedPredictor(MultiTimeframePredictor):
    def __init__(self):
        super().__init__()
        self.validator = SignalValidator(self)
        self.feedback_processor = FeedbackProcessor(self)
        
    def run_enhanced_analysis(self, iterations: int = 5):
        """Run analysis with signal validation and feedback"""
        print("🚀 ENHANCED TRADING SYSTEM WITH VALIDATION")
        print("="*60)
        
        try:
            while self.iteration < iterations:
                self.iteration += 1
                print(f"\n🔄 ITERATION {self.iteration}")
                print("-" * 30)
                
                # 1. Fetch data and generate initial signals
                if self.fetch_multi_timeframe_data():
                    self.analyze_all_timeframes()
                    initial_signals = self.generate_trading_signals()
                    
                    # 2. Validate previous signals against actual price
                    validation = self.validator.validate_signals(initial_signals)
                    self._display_validation_results(validation)
                    
                    # 3. Process feedback and generate improved signals
                    if validation.total_signals > 0:
                        final_signals = self.feedback_processor.process_feedback(
                            initial_signals, validation)
                        self.trading_signals.extend(final_signals)
                    
                    # 4. Display final signals
                    self._display_final_signals()
                    
                    # Wait for next iteration
                    if self.iteration < iterations:
                        print(f"\n⏳ Waiting 60 seconds...")
                        time.sleep(60)
                else:
                    print("❌ Data fetch failed")
                    
        except KeyboardInterrupt:
            print("\n🛑 Analysis stopped")
            self._display_final_summary()
    
    def _display_validation_results(self, validation: ValidationResult):
        """Display validation results"""
        if validation.total_signals > 0:
            print(f"\n📊 VALIDATION RESULTS:")
            print(f"   ✅ Success Rate: {validation.accuracy_rate:.1%}")
            print(f"   💰 Avg P/L: {validation.avg_profit_loss:+.2%}")
            print(f"   🎯 Best Timeframe: {validation.best_timeframe}")
        else:
            print(f"   ⏳ No signals ready for validation yet")
    
    def _display_final_signals(self):
        """Display final trading signals after feedback"""
        if not self.trading_signals:
            return
            
        print(f"\n🎯 FINAL SIGNALS ({len(self.trading_signals)})")
        print("-" * 40)
        
        for i, signal in enumerate(self.trading_signals[-3:], 1):  # Show last 3
            emoji = {"BUY": "🟢", "SELL": "🔴", "STRONG_BUY": "🟢🟢", 
                    "STRONG_SELL": "🔴🔴", "HOLD": "🟡"}.get(signal.signal_type.value, "❓")
            
            print(f"{i}. {emoji} {signal.signal_type.value} ({signal.timeframe})")
            print(f"   💡 {signal.reason}")
            print(f"   💰 Entry: ${signal.entry_price:,.0f} → Target: ${signal.target_price:,.0f}")
            print(f"   ✅ Confidence: {signal.confidence:.0%}")
    
    def _display_final_summary(self):
        """Display comprehensive final summary"""
        print(f"\n🏁 FINAL PERFORMANCE SUMMARY")
        print("="*50)
        
        if self.outcomes:
            successful = sum(1 for o in self.outcomes if o.hit_target)
            total = len(self.outcomes)
            avg_pnl = np.mean([o.profit_loss for o in self.outcomes])
            
            print(f"📊 Total Signals Validated: {total}")
            print(f"✅ Success Rate: {successful/total:.1%}")
            print(f"💰 Average P/L: {avg_pnl:+.2%}")
            print(f"🎯 Best Strategy: Focus on {self._get_best_strategy()}")
        else:
            print("📊 No signals were validated during this run")
            print("💡 Increase iteration count or wait longer for validation")
    
    def _get_best_strategy(self) -> str:
        """Identify best performing strategy"""
        if not self.feedback_processor.learning_data:
            return "Insufficient data"
        
        recent_data = self.feedback_processor.learning_data[-3:]
        best_timeframe = max(set([d['best_timeframe'] for d in recent_data if d['best_timeframe']]),
                           key=lambda x: sum(1 for d in recent_data if d['best_timeframe'] == x))
        
        return f"{best_timeframe} timeframe with regime awareness"

def main():
    """Main execution with validation and feedback"""
    predictor = EnhancedPredictor()
    
    # Run enhanced analysis with validation
    predictor.run_enhanced_analysis(iterations=3)  # Short run for testing
    
    return predictor

if __name__ == "__main__":
    predictor = main()