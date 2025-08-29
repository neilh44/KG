"""
Main Predictor Module - Hybrid LLM + KG + Causal AI Integration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

from data_ingestion import BinanceDataIngestion
from llm_processor import LLMProcessor
from knowledge_graph import CryptoKnowledgeGraph
from causal_ai import CausalAI

class BTCPricePredictor:
    def __init__(self):
        self.data_ingestion = BinanceDataIngestion()
        self.llm_processor = LLMProcessor()
        self.knowledge_graph = CryptoKnowledgeGraph()
        self.causal_ai = CausalAI()
        
        self.market_history = []
        self.prediction_history = []
        
    def initialize_system_with_bulk_data(self, timeframe="1h", days=30):
        """Initialize system with bulk historical data"""
        print(f"🚀 Initializing with bulk historical data...")
        print(f"📊 Timeframe: {timeframe}, Days: {days}")
        
        # Get bulk historical OHLCV data
        historical_ohlcv = self.data_ingestion.get_historical_data_bulk(
            interval=timeframe, days=days
        )
        
        if historical_ohlcv is None or historical_ohlcv.empty:
            print("❌ Failed to fetch historical OHLCV data")
            return False
        
        # Get historical funding rates
        funding_limit = min(days * 3, 500)  # 3 funding rates per day max
        historical_funding = self.data_ingestion.get_historical_funding_rates(
            limit=funding_limit
        )
        
        # Create market data snapshots from historical data
        print(f"📈 Processing {len(historical_ohlcv)} historical data points...")
        
        self.market_history = []
        
        # Process in chunks to simulate different time periods
        chunk_size = max(1, len(historical_ohlcv) // 100)  # Create ~100 data points
        
        for i in range(0, len(historical_ohlcv), chunk_size):
            chunk = historical_ohlcv.iloc[i:i+chunk_size+24]  # Include 24h window
            
            if len(chunk) >= 24:
                # Get corresponding funding rate
                chunk_timestamp = chunk.iloc[-1]['timestamp']
                funding_data = self._find_closest_funding(historical_funding, chunk_timestamp)
                
                # Create market snapshot
                snapshot = {
                    'timestamp': chunk_timestamp,
                    'ohlcv': chunk.tail(24),  # Last 24 hours
                    'funding': funding_data,
                    'order_book': {'bids': [[0, 0]], 'asks': [[0, 0]]},  # Placeholder
                    'open_interest': {'open_interest': 0, 'timestamp': chunk_timestamp}
                }
                
                self.market_history.append(snapshot)
        
        print(f"✅ Created {len(self.market_history)} market snapshots")
        
        # Discover causal relationships
        print("🧠 Discovering causal relationships...")
        try:
            relationships = self.causal_ai.discover_causal_relationships(self.market_history)
            print(f"✅ Discovered {len(relationships)} causal relationships")
            
            # Print discovered relationships
            if relationships:
                print("\n🔍 Discovered Causal Relationships:")
                for rel in relationships[:5]:  # Show top 5
                    print(f"   • {rel['cause']} → {rel['effect']} (strength: {rel['strength']:.3f})")
            
        except Exception as e:
            print(f"⚠️  Causal discovery warning: {e}")
        
        return True
    
    def _find_closest_funding(self, funding_history, target_timestamp):
        """Find funding rate closest to target timestamp"""
        if not funding_history:
            return {'funding_rate': 0.0, 'mark_price': 0.0, 'index_price': 0.0}
        
        # Convert target to timestamp if needed
        if hasattr(target_timestamp, 'timestamp'):
            target_ts = target_timestamp.timestamp() * 1000
        else:
            target_ts = target_timestamp
        
        closest_funding = min(funding_history, 
                            key=lambda x: abs(x['fundingTime'] - target_ts))
        
        return {
            'funding_rate': float(closest_funding['fundingRate']),
            'mark_price': 0.0,  # Not available in historical data
            'index_price': 0.0
        }
        
    def generate_prediction(self, prediction_horizon_hours=24):
        """Generate comprehensive price prediction"""
        print(f"\n--- BTC Price Prediction ({datetime.now()}) ---")
        
        # 1. Get current market data
        current_data = self.data_ingestion.get_market_data_snapshot()
        
        # 2. LLM analysis
        insights = self.llm_processor.analyze_market_structure(current_data)
        narrative = self.llm_processor.generate_narrative(insights)
        
        # 3. Update knowledge graph
        entities = self.knowledge_graph.add_market_entities(current_data, insights)
        self.knowledge_graph.add_relationships(entities, insights)
        
        # 4. Find patterns in knowledge graph
        kg_patterns = self.knowledge_graph.find_patterns(lookback_hours=24)
        
        # 5. Causal analysis
        causal_insights = self._perform_causal_analysis(current_data, insights)
        
        # 6. Generate integrated prediction
        prediction = self._integrate_predictions(current_data, insights, kg_patterns, causal_insights)
        
        # 7. Store prediction for tracking
        prediction['timestamp'] = datetime.now()
        prediction['actual_price'] = current_data['ohlcv']['close'].iloc[-1]
        self.prediction_history.append(prediction)
        
        return prediction
    
    def _perform_causal_analysis(self, current_data, insights):
        """Perform causal analysis on current market conditions"""
        causal_insights = {}
        
        # Analyze potential cascade effects
        if insights['funding_sentiment']['sentiment'] == 'extreme_greed':
            cascade = self.causal_ai.analyze_cascade_effects(
                {'type': 'funding_rate', 'magnitude': 1.5}, current_data
            )
            causal_insights['funding_cascade'] = cascade
        
        # Counterfactual analysis
        counterfactuals = self.causal_ai.generate_counterfactual(
            current_data,
            {'funding_rate': 0.001}  # What if funding was neutral?
        )
        causal_insights['counterfactuals'] = counterfactuals
        
        # Intervention effects
        if insights['volume_profile']['profile'] == 'low_volume':
            interventions = self.causal_ai.predict_intervention_effects(
                current_data,
                {'volume_spike': 1}  # What if volume spiked?
            )
            causal_insights['interventions'] = interventions
        
        return causal_insights
    
    def _integrate_predictions(self, current_data, insights, kg_patterns, causal_insights):
        """Integrate all analysis into final prediction"""
        current_price = current_data['ohlcv']['close'].iloc[-1]
        
        # Base prediction from technical analysis
        base_prediction = self._technical_prediction(current_data, insights)
        
        # Knowledge graph adjustments
        kg_adjustment = self._kg_pattern_adjustment(kg_patterns)
        
        # Causal adjustments
        causal_adjustment = self._causal_adjustment(causal_insights)
        
        # Combine predictions
        final_price_target = base_prediction * (1 + kg_adjustment + causal_adjustment)
        
        # Calculate confidence
        confidence = self._calculate_confidence(insights, kg_patterns, causal_insights)
        
        # Generate risk assessment
        risk_factors = self._assess_risks(current_data, insights, causal_insights)
        
        # Time-based targets
        targets = {
            '1h': current_price * (1 + (final_price_target/current_price - 1) * 0.1),
            '4h': current_price * (1 + (final_price_target/current_price - 1) * 0.4),
            '24h': final_price_target
        }
        
        return {
            'current_price': current_price,
            'predicted_price_24h': final_price_target,
            'price_targets': targets,
            'price_change_pct': ((final_price_target - current_price) / current_price) * 100,
            'confidence': confidence,
            'market_narrative': self.llm_processor.generate_narrative(insights),
            'risk_factors': risk_factors,
            'causal_insights': self._format_causal_insights(causal_insights),
            'key_levels': insights['support_resistance'],
            'market_regime': insights['market_regime']
        }
    
    def _technical_prediction(self, current_data, insights):
        """Base technical analysis prediction"""
        current_price = current_data['ohlcv']['close'].iloc[-1]
        
        # Simple momentum-based prediction
        if insights['trend_strength']['direction'] == 'bullish':
            if insights['trend_strength']['strength'] == 'strong':
                return current_price * 1.03  # 3% up
            else:
                return current_price * 1.015  # 1.5% up
        elif insights['trend_strength']['direction'] == 'bearish':
            if insights['trend_strength']['strength'] == 'strong':
                return current_price * 0.97  # 3% down
            else:
                return current_price * 0.985  # 1.5% down
        else:
            return current_price  # Sideways
    
    def _kg_pattern_adjustment(self, patterns):
        """Adjust prediction based on knowledge graph patterns"""
        adjustment = 0
        
        for pattern in patterns:
            if pattern['pattern'] == 'high_funding_low_volume':
                adjustment -= 0.02  # Bearish adjustment
            # Add more pattern-based adjustments
        
        return adjustment
    
    def _causal_adjustment(self, causal_insights):
        """Adjust prediction based on causal analysis"""
        adjustment = 0
        
        # Funding cascade effects
        if 'funding_cascade' in causal_insights:
            cascades = causal_insights['funding_cascade']
            for cascade in cascades:
                if cascade['effect'] == 'price_change':
                    adjustment += cascade['magnitude'] * 0.5  # Weight the effect
        
        return adjustment
    
    def _calculate_confidence(self, insights, kg_patterns, causal_insights):
        """Calculate prediction confidence"""
        base_confidence = 0.6
        
        # Boost confidence with strong trends
        if insights['trend_strength']['strength'] == 'strong':
            base_confidence += 0.15
        
        # Boost with high volume confirmation
        if insights['volume_profile']['profile'] == 'high_volume':
            base_confidence += 0.1
        
        # Reduce confidence with conflicting signals
        if len(kg_patterns) == 0:
            base_confidence -= 0.05
        
        return min(base_confidence, 0.95)
    
    def _assess_risks(self, current_data, insights, causal_insights):
        """Assess key risk factors"""
        risks = []
        
        # Funding risk
        if insights['funding_sentiment']['sentiment'] == 'extreme_greed':
            risks.append("High funding rate suggests overleveraged longs - liquidation risk")
        
        # Volume risk
        if insights['volume_profile']['profile'] == 'low_volume':
            risks.append("Low volume suggests weak conviction - reversal risk")
        
        # Cascade risk
        if 'funding_cascade' in causal_insights and causal_insights['funding_cascade']:
            risks.append("Potential cascade effects from funding rate pressure")
        
        return risks
    
    def _format_causal_insights(self, causal_insights):
        """Format causal insights for readable output"""
        formatted = []
        
        if 'counterfactuals' in causal_insights:
            formatted.append("Counterfactual: If funding were neutral, price impact would be reduced")
        
        if 'funding_cascade' in causal_insights:
            cascade_count = len(causal_insights['funding_cascade'])
            formatted.append(f"Identified {cascade_count} potential cascade effects")
        
        return formatted
    
    def display_prediction(self, prediction):
        """Display prediction in readable format"""
        print("\n" + "="*60)
        print("🔮 BTC/USDT PRICE PREDICTION")
        print("="*60)
        
        print(f"📊 Current Price: ${prediction['current_price']:,.2f}")
        print(f"🎯 24h Target: ${prediction['predicted_price_24h']:,.2f}")
        print(f"📈 Expected Change: {prediction['price_change_pct']:+.2f}%")
        print(f"✅ Confidence: {prediction['confidence']*100:.1f}%")
        
        print(f"\n🕐 Time-based Targets:")
        for timeframe, target in prediction['price_targets'].items():
            print(f"   {timeframe}: ${target:,.2f}")
        
        print(f"\n📰 Market Narrative:")
        print(f"   {prediction['market_narrative']}")
        
        if prediction['risk_factors']:
            print(f"\n⚠️  Risk Factors:")
            for risk in prediction['risk_factors']:
                print(f"   • {risk}")
        
        if prediction['causal_insights']:
            print(f"\n🧠 Causal Insights:")
            for insight in prediction['causal_insights']:
                print(f"   • {insight}")
        
        print(f"\n🎚️  Key Levels:")
        levels = prediction['key_levels']
        if levels['support']:
            print(f"   Support: {levels['support']}")
        if levels['resistance']:
            print(f"   Resistance: {levels['resistance']}")
        
        print("="*60)

def main():
    """Main execution function"""
    predictor = BTCPricePredictor()
    
    # Initialize system
    predictor.initialize_system(history_hours=24)
    
    # Generate prediction
    prediction = predictor.generate_prediction()
    
    # Display results
    predictor.display_prediction(prediction)
    
    return prediction

if __name__ == "__main__":
    prediction = main()