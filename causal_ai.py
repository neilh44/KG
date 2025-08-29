"""
Causal AI Module - Causal Discovery and Inference
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CausalAI:
    def __init__(self):
        self.causal_models = {}
        self.causal_relationships = []
        self.treatment_effects = {}
        
    def discover_causal_relationships(self, market_data_history):
        """Discover causal relationships from historical data"""
        if not market_data_history:
            return []
            
        # Convert to DataFrame for analysis
        df = self._prepare_causal_dataframe(market_data_history)
        
        # Discover key causal relationships
        causal_pairs = [
            ('funding_rate', 'price_change', 'funding_price_causality'),
            ('volume_spike', 'price_volatility', 'volume_volatility_causality'),
            ('open_interest_change', 'price_direction', 'oi_direction_causality'),
            ('funding_rate', 'liquidation_cascade', 'funding_liquidation_causality')
        ]
        
        discovered_relationships = []
        
        for cause, effect, name in causal_pairs:
            if cause in df.columns and effect in df.columns:
                causal_strength = self._estimate_causal_effect(df, cause, effect)
                
                if abs(causal_strength) > 0.1:  # Significant threshold
                    discovered_relationships.append({
                        'cause': cause,
                        'effect': effect,
                        'strength': causal_strength,
                        'name': name,
                        'confidence': min(abs(causal_strength) * 2, 0.95)
                    })
        
        self.causal_relationships = discovered_relationships
        return discovered_relationships
    
    def _prepare_causal_dataframe(self, market_data_history):
        """Prepare data for causal analysis"""
        records = []
        
        for data_point in market_data_history:
            if 'ohlcv' in data_point and 'funding' in data_point:
                ohlcv = data_point['ohlcv']
                
                # Calculate derived features
                price_change = ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-2] if len(ohlcv) > 1 else 0
                price_volatility = ohlcv['close'].pct_change().std()
                volume_spike = 1 if ohlcv['volume'].iloc[-1] > ohlcv['volume'].mean() * 1.5 else 0
                
                funding_rate = data_point['funding']['funding_rate']
                oi_change = (data_point.get('open_interest', {}).get('open_interest', 0) - 
                            records[-1].get('open_interest', 0)) if records else 0
                
                record = {
                    'timestamp': data_point['timestamp'],
                    'price': ohlcv['close'].iloc[-1],
                    'price_change': price_change,
                    'price_volatility': price_volatility,
                    'volume_spike': volume_spike,
                    'funding_rate': funding_rate,
                    'open_interest': data_point.get('open_interest', {}).get('open_interest', 0),
                    'oi_change': oi_change,
                    'price_direction': 1 if price_change > 0 else 0,
                    'liquidation_cascade': 1 if abs(price_change) > ohlcv['close'].iloc[-1] * 0.02 else 0
                }
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _estimate_causal_effect(self, df, cause, effect):
        """Estimate causal effect using instrumental variables approach"""
        if len(df) < 10:  # Need sufficient data
            return 0
        
        # Simple causal estimation - in production, use more sophisticated methods
        X = df[[cause]].values
        y = df[effect].values
        
        # Add time lags for causal inference
        if len(df) > 5:
            X_lagged = np.column_stack([X[:-1], X[:-2] if len(X) > 2 else X[:-1]])
            y_lagged = y[1:]
            
            if len(X_lagged) == len(y_lagged):
                model = LinearRegression()
                model.fit(X_lagged, y_lagged)
                return model.coef_[0]  # Return primary causal coefficient
        
        # Fallback to correlation-based estimation
        correlation = np.corrcoef(X.flatten(), y.flatten())[0, 1]
        return correlation * 0.7  # Discount for causality vs correlation
    
    def predict_intervention_effects(self, current_market_data, interventions):
        """Predict effects of market interventions"""
        predictions = {}
        
        for intervention_name, intervention_value in interventions.items():
            effect_predictions = []
            
            # Find causal relationships involving this intervention
            relevant_relationships = [r for r in self.causal_relationships 
                                    if r['cause'] == intervention_name]
            
            for relationship in relevant_relationships:
                # Calculate predicted effect
                effect_size = relationship['strength'] * intervention_value
                confidence = relationship['confidence']
                
                effect_predictions.append({
                    'effect_variable': relationship['effect'],
                    'predicted_change': effect_size,
                    'confidence': confidence
                })
            
            predictions[intervention_name] = effect_predictions
        
        return predictions
    
    def analyze_cascade_effects(self, trigger_event, current_market_data):
        """Analyze potential cascade effects from trigger event"""
        cascade_chain = [trigger_event]
        cascade_effects = []
        
        # Find direct effects
        direct_effects = [r for r in self.causal_relationships 
                         if r['cause'] == trigger_event['type']]
        
        for effect_rel in direct_effects:
            effect_magnitude = trigger_event['magnitude'] * effect_rel['strength']
            
            cascade_effects.append({
                'stage': 1,
                'cause': trigger_event['type'],
                'effect': effect_rel['effect'],
                'magnitude': effect_magnitude,
                'confidence': effect_rel['confidence']
            })
            
            # Look for second-order effects
            second_order = [r for r in self.causal_relationships 
                           if r['cause'] == effect_rel['effect']]
            
            for second_rel in second_order:
                second_magnitude = effect_magnitude * second_rel['strength']
                
                if abs(second_magnitude) > 0.05:  # Significant second-order effect
                    cascade_effects.append({
                        'stage': 2,
                        'cause': effect_rel['effect'],
                        'effect': second_rel['effect'],
                        'magnitude': second_magnitude,
                        'confidence': second_rel['confidence'] * 0.8  # Reduce confidence
                    })
        
        return cascade_effects
    
    def generate_counterfactual(self, current_market_data, counterfactual_scenario):
        """Generate counterfactual predictions"""
        counterfactuals = {}
        
        for changed_variable, new_value in counterfactual_scenario.items():
            # Find all effects of this variable
            affected_relationships = [r for r in self.causal_relationships 
                                    if r['cause'] == changed_variable]
            
            current_value = self._get_current_value(current_market_data, changed_variable)
            change_magnitude = new_value - current_value
            
            predicted_effects = {}
            
            for relationship in affected_relationships:
                effect_change = change_magnitude * relationship['strength']
                current_effect_value = self._get_current_value(current_market_data, 
                                                              relationship['effect'])
                
                predicted_effects[relationship['effect']] = {
                    'current_value': current_effect_value,
                    'predicted_value': current_effect_value + effect_change,
                    'change': effect_change,
                    'confidence': relationship['confidence']
                }
            
            counterfactuals[changed_variable] = predicted_effects
        
        return counterfactuals
    
    def _get_current_value(self, market_data, variable):
        """Extract current value of variable from market data"""
        if variable == 'funding_rate':
            return market_data.get('funding', {}).get('funding_rate', 0)
        elif variable == 'price_change':
            ohlcv = market_data.get('ohlcv', pd.DataFrame())
            if len(ohlcv) > 1:
                return ohlcv['close'].iloc[-1] - ohlcv['close'].iloc[-2]
        elif variable == 'volume_spike':
            ohlcv = market_data.get('ohlcv', pd.DataFrame())
            if not ohlcv.empty:
                return 1 if ohlcv['volume'].iloc[-1] > ohlcv['volume'].mean() * 1.5 else 0
        
        return 0
    
    def get_causal_insights_summary(self):
        """Get summary of discovered causal insights"""
        if not self.causal_relationships:
            return "No causal relationships discovered yet."
        
        summary = f"Discovered {len(self.causal_relationships)} causal relationships:\n"
        
        for rel in self.causal_relationships:
            direction = "increases" if rel['strength'] > 0 else "decreases"
            summary += f"- {rel['cause']} {direction} {rel['effect']} "
            summary += f"(strength: {rel['strength']:.3f}, confidence: {rel['confidence']:.2f})\n"
        
        return summary

if __name__ == "__main__":
    from data_ingestion import BinanceDataIngestion
    
    # Initialize
    causal_ai = CausalAI()
    ingestion = BinanceDataIngestion()
    
    # Simulate historical data for causal discovery
    market_history = []
    for i in range(10):
        data = ingestion.get_market_data_snapshot()
        market_history.append(data)
    
    # Discover causal relationships
    relationships = causal_ai.discover_causal_relationships(market_history)
    print("Causal Analysis Complete")
    print(causal_ai.get_causal_insights_summary())