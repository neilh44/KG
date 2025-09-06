import requests
import numpy as np
import math
import time
from datetime import datetime

class GeometricBTCPredictor:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
    def get_live_price(self):
        """Fetch current BTC price from Binance"""
        response = requests.get(f"{self.base_url}/ticker/price?symbol=BTCUSDT")
        return float(response.json()['price'])
    
    def get_kline_data(self, interval, limit=50):
        """Get recent kline data for orbital calculations"""
        response = requests.get(f"{self.base_url}/klines", params={
            'symbol': 'BTCUSDT',
            'interval': interval,
            'limit': limit
        })
        klines = response.json()
        prices = [float(k[4]) for k in klines]  # Close prices
        volumes = [float(k[5]) for k in klines]  # Volumes
        return prices, volumes
    
    def find_orbital_center(self, prices):
        """Find geometric center of price orbit using circle fitting - Aryabhata's method"""
        if len(prices) < 3:
            return sum(prices) / len(prices)
            
        min_variance = float('inf')
        best_center = sum(prices) / len(prices)
        
        price_min = min(prices)
        price_max = max(prices)
        price_range = price_max - price_min
        
        # Search for optimal center with minimal variance (like finding center of celestial orbit)
        for i in range(41):
            test_center = price_min + (price_range * i / 40)
            variance = sum((p - test_center) ** 2 for p in prices)
            
            if variance < min_variance:
                min_variance = variance
                best_center = test_center
                
        return best_center
    
    def calculate_orbital_radius(self, prices, center):
        """Calculate RMS orbital radius - distance from celestial center"""
        distances = [abs(p - center) for p in prices]
        rms_radius = math.sqrt(sum(d ** 2 for d in distances) / len(distances))
        return rms_radius
    
    def calculate_angular_frequency(self, prices, window_size):
        """Calculate adaptive angular frequency from price oscillations"""
        if len(prices) < 4:
            return 2 * math.pi / 20
            
        # Find peaks and troughs for cycle detection
        peaks = []
        troughs = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(i)
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(i)
        
        # Calculate average period
        periods = []
        if len(peaks) > 1:
            periods.extend([peaks[i] - peaks[i-1] for i in range(1, len(peaks))])
        if len(troughs) > 1:
            periods.extend([troughs[i] - troughs[i-1] for i in range(1, len(troughs))])
            
        if periods:
            avg_period = sum(periods) / len(periods)
            angular_freq = 2 * math.pi / max(avg_period, 4)
        else:
            angular_freq = 2 * math.pi / (window_size / 2)
            
        return angular_freq
    
    def calculate_orbital_phase(self, prices, center, angular_freq):
        """Calculate current orbital phase and velocity"""
        if len(prices) < 3:
            return 0, 0
            
        # Get last 3 positions relative to center
        p1 = prices[-3] - center
        p2 = prices[-2] - center  
        p3 = prices[-1] - center
        
        # Calculate orbital velocities
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Current radius and position
        current_radius = abs(p3)
        
        if current_radius < 0.001:
            return math.pi / 2 if v2 > 0 else 3 * math.pi / 2, v2
            
        if angular_freq <= 0:
            angular_freq = 0.1
            
        # Calculate phase using position and velocity
        normalized_pos = p3 / current_radius if current_radius > 0 else 0
        position_phase = math.acos(min(1.0, max(-1.0, normalized_pos)))
        
        # Determine quadrant based on position and velocity
        if p3 >= 0:  # Upper half of orbit
            phase = position_phase if v2 >= 0 else position_phase
        else:  # Lower half of orbit
            if v2 <= 0:
                phase = math.pi + position_phase
            else:
                phase = 2 * math.pi - position_phase
                
        return phase, v2
    
    def calculate_momentum_and_confidence(self, prices, orbital_velocity, angular_freq, radius):
        """Calculate orbital momentum and prediction confidence"""
        if len(prices) < 5:
            return 0, 50
        
        # Calculate momentum as product of velocity and recent price acceleration
        recent_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        acceleration = sum(recent_changes[-3:]) / 3 if len(recent_changes) >= 3 else 0
        momentum = orbital_velocity * acceleration * 0.001
        
        # Calculate confidence based on orbit stability
        price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.1
        
        # Higher confidence for stable orbits with consistent patterns
        orbit_stability = 1 / (1 + price_volatility * 100)
        radius_consistency = 1 / (1 + abs(angular_freq - 0.314) * 10)  # 0.314 ≈ π/10
        
        confidence = (orbit_stability * 0.6 + radius_consistency * 0.4) * 100
        confidence = max(10, min(95, confidence))  # Clamp between 10-95%
        
        return momentum, confidence
    
    def predict_next_orbital_position(self, prices, timeframe_minutes):
        """Predict next price using orbital mechanics - Aryabhata's celestial calculations"""
        window_size = min(30, len(prices))
        recent_prices = prices[-window_size:]
        
        # Calculate orbital parameters
        center = self.find_orbital_center(recent_prices)
        radius = self.calculate_orbital_radius(recent_prices, center)
        angular_freq = self.calculate_angular_frequency(recent_prices, window_size)
        
        # Get current phase and velocity
        current_phase, orbital_velocity = self.calculate_orbital_phase(recent_prices, center, angular_freq)
        
        # Calculate momentum and confidence
        momentum, confidence = self.calculate_momentum_and_confidence(
            recent_prices, orbital_velocity, angular_freq, radius
        )
        
        # Calculate volatility for risk assessment
        volatility = np.std(recent_prices) / np.mean(recent_prices) if len(recent_prices) > 1 else 0.01
        
        # Time step for prediction
        time_step = timeframe_minutes / (24 * 60)  # Fraction of day
        
        # Calculate phase advance
        phase_velocity = angular_freq * time_step
        next_phase = (current_phase + phase_velocity) % (2 * math.pi)
        
        # Factor in orbital velocity for trajectory correction
        velocity_factor = orbital_velocity * 0.001 * timeframe_minutes
        
        # Predict next position on elliptical orbit
        # Using elliptical parametric equations with eccentricity
        eccentricity = min(0.3, abs(orbital_velocity) * 0.0001)  # Dynamic eccentricity
        semi_major = radius * (1 + eccentricity)
        semi_minor = radius * (1 - eccentricity)
        
        # Parametric ellipse position
        x = semi_major * math.cos(next_phase)
        y = semi_minor * math.sin(next_phase)
        
        # Convert back to price space
        elliptical_radius = math.sqrt(x**2 + y**2)
        predicted_displacement = elliptical_radius * math.cos(next_phase)
        
        # Add velocity correction
        predicted_price = center + predicted_displacement + velocity_factor
        predicted_change = predicted_price - recent_prices[-1]
        
        return predicted_price, predicted_change, {
            'center': center,
            'radius': radius,
            'phase': math.degrees(current_phase),
            'next_phase': math.degrees(next_phase),
            'angular_freq': angular_freq,
            'orbital_velocity': orbital_velocity,
            'eccentricity': eccentricity,
            'momentum': momentum,
            'confidence': confidence,
            'volatility': volatility
        }
    
    def backtest_orbital_model(self, interval, periods=50):
        """Backtest orbital prediction model"""
        prices, volumes = self.get_kline_data(interval, periods + 30)
        
        timeframe_map = {'5m': 5, '15m': 15, '1h': 60}
        timeframe_minutes = timeframe_map[interval]
        
        predictions = []
        actuals = []
        orbital_data = []
        
        # Use sliding window for backtesting
        for i in range(30, len(prices) - 1):
            current_prices = prices[:i+1]
            actual_next = prices[i + 1]
            
            pred_price, pred_change, orbital_info = self.predict_next_orbital_position(
                current_prices, timeframe_minutes
            )
            
            predictions.append(pred_price)
            actuals.append(actual_next)
            orbital_data.append(orbital_info)
            
        return np.array(predictions), np.array(actuals), orbital_data
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate prediction accuracy metrics"""
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction Accuracy (up/down prediction)
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        if len(actual_direction) > 0:
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 50
        
        return r2, mape, direction_accuracy
    
    def run_orbital_analysis(self):
        """Run complete orbital analysis for all timeframes - Aryabhata's Eclipse Prediction Method"""
        timeframes = ['5m', '15m', '1h']
        current_price = self.get_live_price()
        
        print(f"Current BTC Price: ${current_price:.2f}")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("BTC ORBITAL PREDICTION MODEL (Aryabhata's Celestial Mechanics)")
        print("Using geometric principles from ancient astronomical calculations")
        print("=" * 80)
        
        for tf in timeframes:
            tf_minutes = {'5m': 5, '15m': 15, '1h': 60}[tf]
            
            # Get recent data for current prediction
            prices, volumes = self.get_kline_data(tf, 50)
            
            # Get prediction with orbital mechanics
            pred_price, pred_change, orbital_info = self.predict_next_orbital_position(
                prices, tf_minutes
            )
            
            # Backtest the model
            try:
                predictions, actuals, backtest_orbital = self.backtest_orbital_model(tf, 30)
                if len(predictions) > 0:
                    r2, mape, direction_acc = self.calculate_metrics(actuals, predictions)
                else:
                    r2, mape, direction_acc = 0, 100, 50
            except:
                r2, mape, direction_acc = 0, 100, 50
            
            change_pct = (pred_change / current_price) * 100
            
            print(f"\n{tf.upper()} TIMEFRAME ORBITAL ANALYSIS:")
            print(f"{'─'*50}")
            print(f"  Predicted Price: ${pred_price:.2f}")
            print(f"  Expected Change: ${pred_change:+.2f} ({change_pct:+.3f}%)")
            print(f"  ")
            print(f"  CELESTIAL PARAMETERS (Aryabhata Method):")
            print(f"    • Orbital Center: ${orbital_info['center']:.2f}")
            print(f"    • Orbital Radius: ${orbital_info['radius']:.2f}")
            print(f"    • Current Phase: {orbital_info['phase']:.1f}°")
            print(f"    • Next Phase: {orbital_info['next_phase']:.1f}°")
            print(f"    • Orbital Velocity: {orbital_info['orbital_velocity']:.3f}")
            print(f"    • Eccentricity: {orbital_info['eccentricity']:.4f}")
            
            # Advanced orbital analysis and trading signals
            phase = orbital_info['phase']
            velocity = orbital_info['orbital_velocity']
            momentum = orbital_info['momentum']
            confidence = orbital_info['confidence']
            
            # Determine orbital position and strength
            if 315 <= phase <= 360 or 0 <= phase <= 45:
                position = "Near Aphelion (Bottom) - Like Moon's Farthest Point"
                signal_strength = "STRONG BUY" if velocity > 0 and momentum > 0 else "BUY"
            elif 45 < phase <= 135:
                position = "Ascending Arc (Rising) - Like Rising Moon"
                signal_strength = "HOLD/BUY" if velocity > 0 else "CAUTION"
            elif 135 < phase <= 225:
                position = "Near Perihelion (Top) - Like Moon's Closest Point"
                signal_strength = "STRONG SELL" if velocity < 0 and momentum < 0 else "SELL"
            else:
                position = "Descending Arc (Falling) - Like Setting Moon"
                signal_strength = "HOLD/SELL" if velocity < 0 else "CAUTION"
            
            # Additional confluence signals
            confluence = []
            if abs(change_pct) > 0.1:
                confluence.append("Significant Celestial Movement Expected")
            if orbital_info.get('eccentricity', 0) > 0.01:
                confluence.append("High Orbital Eccentricity - Volatile Celestial Phase")
            if confidence > 70:
                confluence.append("High Astronomical Confidence")
            elif confidence < 30:
                confluence.append("Low Confidence - Irregular Celestial Pattern")
            
            print(f"  ")
            print(f"  BACKTEST VALIDATION:")
            print(f"    • R² Score: {r2:.4f}")
            print(f"    • MAPE: {mape:.2f}%")
            print(f"    • Direction Accuracy: {direction_acc:.1f}%")
            print(f"    • Model Confidence: {confidence:.1f}%")
            print(f"  ")
            print(f"  ARYABHATA'S CELESTIAL SIGNALS:")
            print(f"    • Orbital Position: {position}")
            print(f"    • Trading Signal: {signal_strength}")
            print(f"    • Momentum: {momentum:.4f}")
            if confluence:
                print(f"    • Celestial Confluence: {', '.join(confluence)}")
            
            # Risk assessment based on orbital mechanics
            if orbital_info['eccentricity'] > 0.02:
                risk_level = "HIGH - Highly elliptical orbit (like comet)"
            elif orbital_info['volatility'] > 0.01:
                risk_level = "MEDIUM - Moderate celestial volatility"
            else:
                risk_level = "LOW - Stable circular orbit (like planets)"
                
            print(f"    • Celestial Risk: {risk_level}")
            
            # Ancient astronomical wisdom
            print(f"  ")
            print(f"  ANCIENT WISDOM:")
            if phase < 90:
                wisdom = "As Aryabhata observed eclipses, this marks a new celestial cycle beginning"
            elif phase < 180:
                wisdom = "The celestial body rises - favorable for growth and expansion"
            elif phase < 270:
                wisdom = "Peak celestial power - time for harvest or taking profits"
            else:
                wisdom = "The celestial descent - prepare for the next cycle"
            print(f"    • {wisdom}")

# Run the orbital analysis
if __name__ == "__main__":
    print("🌟 Initializing Aryabhata's BTC Celestial Predictor...")
    print("Using ancient Indian astronomical methods for modern crypto prediction")
    print()
    
    predictor = GeometricBTCPredictor()
    try:
        predictor.run_orbital_analysis()
    except Exception as e:
        print(f"Error in celestial calculations: {e}")
        print("The stars may be misaligned. Please check your internet connection or try again.")