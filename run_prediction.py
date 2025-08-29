"""
Simple script to run BTC price prediction
"""
from predictor import BTCPricePredictor

def quick_prediction():
    """Run a quick prediction with minimal setup"""
    print("🚀 Starting BTC Price Prediction...")
    
    try:
        predictor = BTCPricePredictor()
        
        # Initialize with more history for better causal discovery
        success = predictor.initialize_system(history_hours=15)  # More data for causal patterns
        
        if not success:
            print("❌ Failed to initialize system. Exiting.")
            return None
        
        # Generate and display prediction
        print("\nGenerating prediction...")
        prediction = predictor.generate_prediction()
        
        if prediction:
            predictor.display_prediction(prediction)
        else:
            print("❌ Failed to generate prediction")
            return None
        
        return prediction
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    prediction = quick_prediction()
    
    # Optional: Save prediction to file
    import json
    import datetime
    
    filename = f"btc_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        prediction_copy = prediction.copy()
        prediction_copy['timestamp'] = str(prediction_copy['timestamp'])
        json.dump(prediction_copy, f, indent=2, default=str)
    
    print(f"\n💾 Prediction saved to {filename}")