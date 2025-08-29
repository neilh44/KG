"""
Continuous BTC Price Monitoring and Prediction
"""
import time
from datetime import datetime
from predictor import BTCPricePredictor

def continuous_monitoring():
    """Run continuous price monitoring with periodic predictions"""
    print("🔄 Starting Continuous BTC Monitoring...")
    
    predictor = BTCPricePredictor()
    
    # Initialize with substantial history
    print("📚 Building comprehensive market knowledge...")
    success = predictor.initialize_system(history_hours=25)
    
    if not success:
        print("❌ Failed to initialize. Exiting.")
        return
    
    prediction_count = 0
    
    while True:
        try:
            prediction_count += 1
            print(f"\n🔮 Generating Prediction #{prediction_count} at {datetime.now()}")
            
            # Generate prediction
            prediction = predictor.generate_prediction()
            
            if prediction:
                print(f"\n⚡ QUICK UPDATE:")
                print(f"   Current: ${prediction['current_price']:,.2f}")
                print(f"   24h Target: ${prediction['predicted_price_24h']:,.2f}")
                print(f"   Change: {prediction['price_change_pct']:+.2f}%")
                print(f"   Confidence: {prediction['confidence']*100:.1f}%")
                
                # Show full prediction every 3rd time
                if prediction_count % 3 == 0:
                    predictor.display_prediction(prediction)
                
                # Alert on high confidence predictions
                if prediction['confidence'] > 0.8:
                    print("🚨 HIGH CONFIDENCE PREDICTION!")
                
                # Alert on significant price moves
                if abs(prediction['price_change_pct']) > 5:
                    print(f"🚨 SIGNIFICANT MOVE PREDICTED: {prediction['price_change_pct']:+.2f}%")
            
            # Wait before next prediction (adjustable)
            wait_minutes = 10
            print(f"⏰ Next prediction in {wait_minutes} minutes...")
            time.sleep(wait_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            print("🔄 Retrying in 30 seconds...")
            time.sleep(30)

if __name__ == "__main__":
    continuous_monitoring()