"""
Knowledge Graph Module - Relationship Mapping
"""
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class CryptoKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.relationships = {}
        self.entity_data = {}
        
    def add_market_entities(self, market_data, insights):
        """Add market entities and their current state"""
        timestamp = market_data['timestamp']
        
        # Price entity
        price_node = f"BTC_price_{timestamp.strftime('%Y%m%d_%H')}"
        current_price = market_data['ohlcv']['close'].iloc[-1]
        
        self.graph.add_node(price_node, 
                           type='price',
                           value=current_price,
                           timestamp=timestamp)
        
        # Funding rate entity
        funding_node = f"funding_rate_{timestamp.strftime('%Y%m%d_%H')}"
        self.graph.add_node(funding_node,
                           type='funding_rate',
                           value=market_data['funding']['funding_rate'],
                           timestamp=timestamp)
        
        # Volume entity
        volume_node = f"volume_{timestamp.strftime('%Y%m%d_%H')}"
        current_volume = market_data['ohlcv']['volume'].iloc[-1]
        self.graph.add_node(volume_node,
                           type='volume',
                           value=current_volume,
                           timestamp=timestamp)
        
        # Open Interest entity
        oi_node = f"open_interest_{timestamp.strftime('%Y%m%d_%H')}"
        self.graph.add_node(oi_node,
                           type='open_interest',
                           value=market_data['open_interest']['open_interest'],
                           timestamp=timestamp)
        
        # Market regime entity
        regime_node = f"market_regime_{timestamp.strftime('%Y%m%d_%H')}"
        self.graph.add_node(regime_node,
                           type='market_regime',
                           value=insights['market_regime'],
                           timestamp=timestamp)
        
        return {
            'price': price_node,
            'funding': funding_node,
            'volume': volume_node,
            'oi': oi_node,
            'regime': regime_node
        }
    
    def add_relationships(self, entities, insights):
        """Add relationships between entities"""
        timestamp = datetime.now()
        
        # Funding -> Price relationship
        if insights['funding_sentiment']['sentiment'] in ['greed', 'extreme_greed']:
            self.graph.add_edge(entities['funding'], entities['price'],
                              relationship='negative_pressure',
                              strength=0.7,
                              timestamp=timestamp)
        
        # Volume -> Price relationship  
        if insights['volume_profile']['profile'] == 'high_volume':
            self.graph.add_edge(entities['volume'], entities['price'],
                              relationship='momentum_confirmation',
                              strength=0.8,
                              timestamp=timestamp)
        
        # Open Interest -> Volatility relationship
        self.graph.add_edge(entities['oi'], entities['regime'],
                          relationship='volatility_influence',
                          strength=0.6,
                          timestamp=timestamp)
        
        # Market regime -> Price direction
        if insights['market_regime'] == 'bull_market':
            self.graph.add_edge(entities['regime'], entities['price'],
                              relationship='directional_bias',
                              strength=0.75,
                              timestamp=timestamp)
    
    def find_patterns(self, lookback_hours=24):
        """Find recurring patterns in the knowledge graph"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=lookback_hours)
        
        patterns = []
        
        # Find nodes within time window
        recent_nodes = [n for n, d in self.graph.nodes(data=True) 
                       if d.get('timestamp', datetime.min) > cutoff_time]
        
        # Pattern 1: High funding + Low volume -> Price drop
        funding_nodes = [n for n in recent_nodes if 'funding_rate' in n]
        volume_nodes = [n for n in recent_nodes if 'volume' in n]
        price_nodes = [n for n in recent_nodes if 'BTC_price' in n]
        
        for f_node in funding_nodes:
            funding_val = self.graph.nodes[f_node].get('value', 0)
            if funding_val > 0.005:  # High funding
                # Find corresponding volume and price
                timestamp_str = f_node.split('_')[-1]
                vol_node = f"volume_{timestamp_str}"
                price_node = f"BTC_price_{timestamp_str}"
                
                if vol_node in self.graph.nodes and price_node in self.graph.nodes:
                    vol_val = self.graph.nodes[vol_node].get('value', 0)
                    # Look for pattern
                    if vol_val < np.percentile([self.graph.nodes[v]['value'] for v in volume_nodes], 25):
                        patterns.append({
                            'pattern': 'high_funding_low_volume',
                            'prediction': 'price_drop_risk',
                            'confidence': 0.65,
                            'timestamp': self.graph.nodes[f_node]['timestamp']
                        })
        
        return patterns
    
    def get_entity_influence_score(self, entity_type):
        """Calculate influence score of entity type on price"""
        price_nodes = [n for n, d in self.graph.nodes(data=True) 
                      if d.get('type') == 'price']
        entity_nodes = [n for n, d in self.graph.nodes(data=True) 
                       if d.get('type') == entity_type]
        
        influence_scores = []
        
        for entity_node in entity_nodes:
            # Check edges to price nodes
            edges_to_price = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                             if u == entity_node and v in price_nodes]
            
            total_strength = sum([d.get('strength', 0) for _, _, d in edges_to_price])
            influence_scores.append(total_strength)
        
        return np.mean(influence_scores) if influence_scores else 0
    
    def get_market_snapshot(self):
        """Get current market state from knowledge graph"""
        latest_nodes = {}
        current_time = datetime.now()
        
        # Get most recent node of each type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type')
            if node_type:
                if (node_type not in latest_nodes or 
                    data.get('timestamp', datetime.min) > 
                    latest_nodes[node_type].get('timestamp', datetime.min)):
                    latest_nodes[node_type] = {
                        'node': node,
                        'value': data.get('value'),
                        'timestamp': data.get('timestamp')
                    }
        
        return latest_nodes
    
    def export_relationships(self):
        """Export all relationships for analysis"""
        relationships = []
        for u, v, d in self.graph.edges(data=True):
            relationships.append({
                'source': u,
                'target': v,
                'relationship': d.get('relationship'),
                'strength': d.get('strength'),
                'timestamp': d.get('timestamp')
            })
        
        return pd.DataFrame(relationships)

if __name__ == "__main__":
    from data_ingestion import BinanceDataIngestion
    from llm_processor import LLMProcessor
    
    # Initialize components
    ingestion = BinanceDataIngestion()
    processor = LLMProcessor()
    kg = CryptoKnowledgeGraph()
    
    # Get data and insights
    market_data = ingestion.get_market_data_snapshot()
    insights = processor.analyze_market_structure(market_data)
    
    # Build knowledge graph
    entities = kg.add_market_entities(market_data, insights)
    kg.add_relationships(entities, insights)
    
    # Find patterns
    patterns = kg.find_patterns()
    snapshot = kg.get_market_snapshot()
    
    print("Knowledge Graph built successfully")
    print(f"Found {len(patterns)} patterns")
    print("Current market snapshot:", {k: v['value'] for k, v in snapshot.items()})