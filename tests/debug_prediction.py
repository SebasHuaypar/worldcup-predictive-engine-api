import sys
import os
import pandas as pd
import joblib

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simulator import predict_match_expectations, load_model
from src.data.database import get_connection, load_all_matches
from src.features.builder import get_rolling_stats

def debug():
    df_matches = load_all_matches()
    home_gf_l5, home_ga_l5 = get_rolling_stats("Argentina", '9999-12-31', df_matches, 5)
    print(f"Argentina L5 GF: {home_gf_l5}, GA: {home_ga_l5}")
    
    exp = predict_match_expectations("Argentina", "France", neutral=True)
    print("\nExpectations:")
    for k, v in exp.items():
        print(f" - {k}: {v}")
        
    model = load_model()
    print("\nModel Features Importance:")
    print("Unified Model Feature importances:", model.feature_importances_)

if __name__ == "__main__":
    debug()
