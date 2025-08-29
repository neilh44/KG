"""
Bulk Historical Data Analysis for Better Causal Discovery
"""
from predictor import BTCPricePredictor

def analyze_with_bulk_data():
    """Run analysis with bulk historical data"""
    print("🔬 Starting Bulk Historical Analysis...")
    
    predictor = BTCPricePredictor()
    
    # Choose your timeframe and history
    timeframes = {
        "1h": {"days": 41, "desc": "1-hour intervals, ~1000 data points, 41 days"},
        "4h": {"days": 167, "desc": "4-hour intervals, ~1000 data points, 167 days"}, 
        "1d": {"days": 1000, "desc": "Daily intervals, 1000 data points, ~3 years"}
    }
    
    print("\n📊 Available Timeframes:")
    for tf, info in timeframes.items():
        print(f"   {tf}: {info['desc']}")
    
    # Use 1-hour data for good granularity
    selected_timeframe = "1h"
    config = timeframes[selected_timeframe]
    
    print(f"\n🎯 Selected: {config['desc']}")
    print("🔄 This will fetch ~1000 data points in one API call...")
    
    # Initialize with bulk data
    success = predictor.initialize_system_with_bulk_data(
        timeframe=selected_timeframe, 
        days=config['days']
    )
    
    if not success:
        print("❌ Failed to initialize with bulk data")
        return
    
    # Generate enhanced prediction
    print("\n🔮 Generating Enhanced Prediction...")
    prediction = predictor.generate_prediction()
    
    if prediction:
        predictor.display_prediction(prediction)
        
        # Show causal insights summary
        print("\n" + "="*60)
        print("🧠 CAUSAL ANALYSIS SUMMARY")
        print("="*60)
        print(predictor.causal_ai.get_causal_insights_summary())
        
        # Show knowledge graph insights
        kg_snapshot = predictor.knowledge_graph.get_market_snapshot()
        if kg_snapshot:
            print("\n📊 KNOWLEDGE GRAPH SNAPSHOT:")
            for entity_type, data in kg_snapshot.items():
                print(f"   {entity_type}: {data.get('value')}")
    
    return prediction

def compare_timeframes():
    """Compare predictions across different timeframes"""
    print("⚖️  Comparing Timeframe Analysis...")
    
    timeframes = ["1h", "4h"]
    predictions = {}
    
    for tf in timeframes:
        print(f"\n📊 Analyzing {tf} timeframe...")
        predictor = BTCPricePredictor()
        
        days = 30 if tf == "1h" else 120
        success = predictor.initialize_system_with_bulk_data(
            timeframe=tf, days=days
        )
        
        if success:
            prediction = predictor.generate_prediction()
            predictions[tf] = prediction
            
            print(f"   {tf}: ${prediction['predicted_price_24h']:,.0f} "
                  f"({prediction['price_change_pct']:+.1f}%) "
                  f"conf: {prediction['confidence']*100:.0f}%")
    
    # Compare results
    if len(predictions) > 1:
        print(f"\n🔍 TIMEFRAME COMPARISON:")
        for tf, pred in predictions.items():
            print(f"   {tf}: Target ${pred['predicted_price_24h']:,.0f} "
                  f"({pred['price_change_pct']:+.2f}%)")
    
    return predictions

if __name__ == "__main__":
    # Choose your analysis type
    analysis_type = input("Choose analysis: (1) Bulk Analysis (2) Compare Timeframes: ")
    
    if analysis_type == "2":
        results = compare_timeframes()
    else:
        results = analyze_with_bulk_data()
    
    print(f"\n✅ Analysis complete!")