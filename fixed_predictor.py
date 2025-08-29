"""
Fixed Predictor using working bulk data approach
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

from llm_processor import LLMProcessor
from knowledge_graph import CryptoKnowledgeGraph
from causal_ai import CausalAI

class FixedBTCPredictor:
    def __init__(self):
        self.llm_processor = LLMProcessor()
        self.knowledge_graph = CryptoKnowledgeGraph()
        self.causal_ai = CausalAI()
        
        self.market_history = []
        self.prediction_history = []
    
    def fetch_bulk_historical_data(self, hours=500):
        """Working bulk data fetcher"""
        print(f"📊 Fetching {hours} hours of historical data...")
        
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'limit': min(hours, 1000)
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.text}")
            return None
        
        data = response.json()
        print(f"✅ Retrieved {len(data)} data points")
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    def create_market_snapshots(self, historical_df, chunk_size=20):
        """Convert historical data to market snapshots for causal analysis"""
        print(f"🔄 Creating market snapshots (chunk size: {chunk_size})...")
        
        snapshots = []
        
        for i in range(chunk_size, len(historical_df), chunk_size):
            chunk = historical_df.iloc[max(0, i-24):i]  # 24h window
            
            if len(chunk) >= 24:
                snapshot = {
                    'timestamp': chunk.iloc[-1]['timestamp'],
                    'ohlcv': chunk,
                    'funding': {'funding_rate': 0.0001, 'mark_price': chunk.iloc[-1]['close']},
                    'order_book': {'bids': [[0, 0]], 'asks': [[0, 0]]},
                    'open_interest': {'open_interest': 1000000}
                }
                snapshots.append(snapshot)
        
        print(f"✅ Created {len(snapshots)} market snapshots")
        return snapshots
    
    def initialize_with_bulk_data(self, hours=500):
        """Initialize system with working bulk data approach"""
        print("🚀 Initializing Enhanced Prediction System...")
        
        # Fetch bulk historical data
        historical_df = self.fetch_bulk_historical_data(hours)
        if historical_df is None:
            return False
        
        # Create market snapshots for causal analysis
        self.market_history = self.create_market_snapshots(historical_df)
        
        # Discover causal relationships
        print("🧠 Discovering causal relationships...")
        relationships = self.causal_ai.discover_causal_relationships(self.market_history)
        
        print(f"✅ System initialized successfully!")
        print(f"📊 Data points: {len(historical_df)}")
        print(f"🔗 Market snapshots: {len(self.market_history)}")
        print(f"🧠 Causal relationships: {len(relationships)}")
        
        # Show discovered relationships
        if relationships:
            print("\n🔍 Discovered Causal Relationships:")
            for rel in relationships[:3]:
                print(f"   • {rel['cause']} → {rel['effect']} (strength: {rel['strength']:.3f})")
        
        return True
    
    def get_current_market_data(self):
        """Get current market snapshot"""
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': 'BTCUSDT', 'interval': '1h', 'limit': 24}
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return {
            'timestamp': datetime.now(),
            'ohlcv': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
            'funding': {'funding_rate': 0.0001},
            'order_book': {'bids': [[0, 0]], 'asks': [[0, 0]]},
            'open_interest': {'open_interest': 1000000}
        }
    
    def generate_enhanced_prediction(self):
        """Generate prediction with bulk data insights"""
        print("\n🔮 Generating Enhanced Prediction...")
        
        # Get current market data
        current_data = self.get_current_market_data()
        if not current_data:
            print("❌ Failed to get current market data")
            return None
        
        # LLM analysis
        insights = self.llm_processor.analyze_market_structure(current_data)
        
        # Knowledge graph analysis
        entities = self.knowledge_graph.add_market_entities(current_data, insights)
        self.knowledge_graph.add_relationships(entities, insights)
        kg_patterns = self.knowledge_graph.find_patterns(lookback_hours=24)
        
        # Enhanced causal analysis
        causal_insights = self._enhanced_causal_analysis(current_data)
        
        # Generate integrated prediction
        prediction = self._create_enhanced_prediction(current_data, insights, causal_insights)
        
        return prediction
    
    def _enhanced_causal_analysis(self, current_data):
        """Enhanced causal analysis using bulk data insights"""
        insights = {}
        
        # Use discovered causal relationships for prediction
        if self.causal_ai.causal_relationships:
            insights['strong_relationships'] = [
                rel for rel in self.causal_ai.causal_relationships 
                if abs(rel['strength']) > 0.1
            ]
        
        # Current market conditions
        current_price = current_data['ohlcv']['close'].iloc[-1]
        price_change = current_data['ohlcv']['close'].pct_change().iloc[-1]
        volume_ratio = (current_data['ohlcv']['volume'].iloc[-1] / 
                       current_data['ohlcv']['volume'].mean())
        
        insights['current_conditions'] = {
            'price_change': price_change,
            'volume_ratio': volume_ratio,
            'price_level': current_price
        }
        
        return insights
    
    def _create_enhanced_prediction(self, current_data, insights, causal_insights):
        """Create enhanced prediction using all analysis"""
        current_price = current_data['ohlcv']['close'].iloc[-1]
        
        # Base prediction from trend analysis
        if insights['trend_strength']['direction'] == 'bullish':
            base_change = 0.02 if insights['trend_strength']['strength'] == 'strong' else 0.01
        elif insights['trend_strength']['direction'] == 'bearish':
            base_change = -0.02 if insights['trend_strength']['strength'] == 'strong' else -0.01
        else:
            base_change = 0
        
        # Causal adjustments
        causal_adjustment = 0
        strong_relationships = causal_insights.get('strong_relationships', [])
        
        for rel in strong_relationships:
            if 'volume' in rel['cause'] and 'price' in rel['effect']:
                volume_ratio = causal_insights['current_conditions']['volume_ratio']
                if volume_ratio > 1.5:  # High volume
                    causal_adjustment += rel['strength'] * 0.5
        
        # Final prediction
        total_change = base_change + causal_adjustment
        predicted_price = current_price * (1 + total_change)
        
        # Calculate confidence based on data quality
        confidence = 0.7
        if len(self.causal_ai.causal_relationships) > 0:
            confidence += 0.1
        if len(strong_relationships) > 0:
            confidence += 0.1
        
        return {
            'current_price': current_price,
            'predicted_price_24h': predicted_price,
            'price_change_pct': total_change * 100,
            'confidence': min(confidence, 0.95),
            'causal_relationships_used': len(strong_relationships),
            'data_points_analyzed': len(self.market_history),
            'market_narrative': self.llm_processor.generate_narrative(insights),
            'key_insights': [
                f"Analyzed {len(self.market_history)} historical periods",
                f"Found {len(self.causal_ai.causal_relationships)} causal relationships",
                f"Using {len(strong_relationships)} strong relationships for prediction"
            ]
        }
    
    def display_enhanced_prediction(self, prediction):
        """Display enhanced prediction results"""
        print("\n" + "="*70)
        print("🔮 ENHANCED BTC/USDT PRICE PREDICTION")
        print("="*70)
        
        print(f"📊 Current Price: ${prediction['current_price']:,.2f}")
        print(f"🎯 24h Prediction: ${prediction['predicted_price_24h']:,.2f}")
        print(f"📈 Expected Change: {prediction['price_change_pct']:+.2f}%")
        print(f"✅ Confidence: {prediction['confidence']*100:.1f}%")
        
        print(f"\n📊 Analysis Summary:")
        print(f"   • Historical data points: {prediction['data_points_analyzed']}")
        print(f"   • Causal relationships: {prediction['causal_relationships_used']}")
        
        print(f"\n🧠 Key Insights:")
        for insight in prediction['key_insights']:
            print(f"   • {insight}")
        
        print(f"\n📰 Market Narrative:")
        print(f"   {prediction['market_narrative']}")
        
        print("="*70)

def main():
    """Main execution with enhanced bulk data analysis"""
    predictor = FixedBTCPredictor()
    
    # Initialize with bulk data
    success = predictor.initialize_with_bulk_data(hours=500)
    
    if not success:
        print("❌ Failed to initialize system")
        return
    
    # Generate enhanced prediction
    prediction = predictor.generate_enhanced_prediction()
    
    if prediction:
        predictor.display_enhanced_prediction(prediction)
    else:
        print("❌ Failed to generate prediction")

if __name__ == "__main__":
    main()