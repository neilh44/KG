"""
LLM Processing Module - Market Intelligence Extraction
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
import re

class LLMProcessor:
    def __init__(self):
        self.sentiment_cache = {}
    
    def calculate_technical_indicators(self, df):
        """Generate technical analysis insights"""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else df['close'].mean()
        
        # Bollinger Bands
        df['bb_upper'] = df['ma_20'] + (df['close'].rolling(20).std() * 2)
        df['bb_lower'] = df['ma_20'] - (df['close'].rolling(20).std() * 2)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def analyze_market_structure(self, market_data):
        """Extract market structure insights"""
        ohlcv = market_data['ohlcv']
        funding = market_data['funding']
        oi = market_data['open_interest']
        
        # Technical analysis
        tech_df = self.calculate_technical_indicators(ohlcv)
        latest = tech_df.iloc[-1]
        
        insights = {
            'trend_strength': self._assess_trend_strength(tech_df),
            'volatility_regime': self._assess_volatility(ohlcv),
            'funding_sentiment': self._assess_funding_sentiment(funding),
            'volume_profile': self._assess_volume_profile(tech_df),
            'support_resistance': self._find_support_resistance(ohlcv),
            'market_regime': self._determine_market_regime(tech_df, funding)
        }
        
        return insights
    
    def _assess_trend_strength(self, df):
        """Assess trend strength and direction"""
        latest = df.iloc[-1]
        
        if latest['close'] > latest['ma_20'] > latest['ma_50']:
            return {'direction': 'bullish', 'strength': 'strong'}
        elif latest['close'] > latest['ma_20']:
            return {'direction': 'bullish', 'strength': 'weak'}
        elif latest['close'] < latest['ma_20'] < latest['ma_50']:
            return {'direction': 'bearish', 'strength': 'strong'}
        else:
            return {'direction': 'bearish', 'strength': 'weak'}
    
    def _assess_volatility(self, df):
        """Assess current volatility regime"""
        returns = df['close'].pct_change().dropna()
        current_vol = returns.rolling(24).std().iloc[-1] * np.sqrt(24)  # 24h volatility
        avg_vol = returns.std() * np.sqrt(24)
        
        if current_vol > avg_vol * 1.5:
            return 'high'
        elif current_vol < avg_vol * 0.7:
            return 'low'
        else:
            return 'normal'
    
    def _assess_funding_sentiment(self, funding_data):
        """Analyze funding rate sentiment"""
        funding_rate = funding_data['funding_rate']
        
        if funding_rate > 0.01:  # 1%
            return {'sentiment': 'extreme_greed', 'signal': 'potential_short_squeeze'}
        elif funding_rate > 0.003:  # 0.3%
            return {'sentiment': 'greed', 'signal': 'long_heavy'}
        elif funding_rate < -0.003:
            return {'sentiment': 'fear', 'signal': 'short_heavy'}
        else:
            return {'sentiment': 'neutral', 'signal': 'balanced'}
    
    def _assess_volume_profile(self, df):
        """Analyze volume patterns"""
        latest_vol_ratio = df['volume_ratio'].iloc[-1]
        
        if latest_vol_ratio > 2:
            return {'profile': 'high_volume', 'signal': 'strong_conviction'}
        elif latest_vol_ratio < 0.5:
            return {'profile': 'low_volume', 'signal': 'weak_conviction'}
        else:
            return {'profile': 'normal_volume', 'signal': 'average_conviction'}
    
    def _find_support_resistance(self, df):
        """Identify key support and resistance levels"""
        highs = df['high'].rolling(5).max()
        lows = df['low'].rolling(5).min()
        current_price = df['close'].iloc[-1]
        
        # Find recent significant levels
        resistance_levels = highs[highs > current_price].tail(3).tolist()
        support_levels = lows[lows < current_price].tail(3).tolist()
        
        return {
            'resistance': resistance_levels,
            'support': support_levels,
            'current_price': current_price
        }
    
    def _determine_market_regime(self, df, funding_data):
        """Determine overall market regime"""
        trend = self._assess_trend_strength(df)
        funding_sentiment = self._assess_funding_sentiment(funding_data)
        volatility = self._assess_volatility(df)
        
        if trend['direction'] == 'bullish' and funding_sentiment['sentiment'] in ['greed', 'extreme_greed']:
            return 'bull_market'
        elif trend['direction'] == 'bearish' and funding_sentiment['sentiment'] == 'fear':
            return 'bear_market'
        elif volatility == 'low':
            return 'crab_market'
        else:
            return 'transitional'
    
    def generate_narrative(self, insights):
        """Generate human-readable market narrative"""
        trend = insights['trend_strength']
        funding = insights['funding_sentiment']
        volume = insights['volume_profile']
        regime = insights['market_regime']
        
        narrative = f"Market shows {trend['strength']} {trend['direction']} trend. "
        narrative += f"Funding indicates {funding['sentiment']} with {funding['signal']}. "
        narrative += f"Volume profile: {volume['profile']} suggesting {volume['signal']}. "
        narrative += f"Overall regime: {regime}."
        
        return narrative

if __name__ == "__main__":
    from data_ingestion import BinanceDataIngestion
    
    ingestion = BinanceDataIngestion()
    processor = LLMProcessor()
    
    market_data = ingestion.get_market_data_snapshot()
    insights = processor.analyze_market_structure(market_data)
    narrative = processor.generate_narrative(insights)
    
    print("Market Analysis:", narrative)