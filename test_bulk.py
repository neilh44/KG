"""
Test script to isolate bulk data fetching issue
"""
from data_ingestion import BinanceDataIngestion
import pandas as pd

def test_bulk_data():
    print("🔍 Testing bulk data fetching...")
    
    ingestion = BinanceDataIngestion()
    
    # Test the basic function
    print("📊 Testing get_historical_data_bulk...")
    
    try:
        result = ingestion.get_historical_data_bulk(interval="1h", days=2)
        
        print(f"Result type: {type(result)}")
        
        if result is None:
            print("❌ Result is None")
            return
            
        if isinstance(result, pd.DataFrame):
            print(f"✅ Got DataFrame with shape: {result.shape}")
            print(f"Empty: {result.empty}")
            
            if not result.empty:
                print("✅ Sample data:")
                print(result.head(2))
                print(f"Columns: {list(result.columns)}")
            else:
                print("❌ DataFrame is empty")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_bulk_data()