import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.models.sklearn_model import StockPredictionModel
from src.agent.trading_agent import TradingAgent
from src.data.preprocess import preprocess_data

def test_model_predictions():
    """Test model predictions and direction classification"""
    # Load data
    data_path = os.path.join(project_root, "data", "raw", "TSLA.csv")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(data_path, sequence_length)
    
    # Load model
    model_path = os.path.join(project_root, "models", "advanced_model.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model = StockPredictionModel(sequence_length, X_train.shape[2])
    model.load(model_path)
    
    # Make predictions
    print("Testing model predictions...")
    pred = model.predict(X_test[:5])
    print(f"First 5 predictions: {pred.flatten()}")
    
    # Test direction prediction if available
    if hasattr(model, 'predict_direction') and model.direction_classifier is not None:
        dir_pred = model.predict_direction(X_test[:5])
        print(f"First 5 direction predictions: {dir_pred}")
    else:
        print("Direction classifier not available")
    
    # Test trading agent
    print("\nTesting trading agent...")
    agent = TradingAgent(model, scaler)
    
    # Prepare test data
    test_idx = len(X_test) // 2
    test_data = X_test[test_idx:test_idx+10]  # Use 10 days
    
    # Get unscaled prices for verification
    feature_columns = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 
                      'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volatility', 'Return']
    last_rows = processed_data.iloc[-len(X_test)+test_idx:-(len(X_test)-test_idx-10)]
    
    print("\nModel and Agent Test Results:")
    print(f"{'Day':4} {'Act Price':10} {'Pred Price':10} {'Direction':8} {'Decision':8} {'Amount':10}")
    print("-" * 60)
    
    # Make decisions for each day
    for i, (X, row) in enumerate(zip(test_data, last_rows.itertuples())):
        X_batch = X.reshape(1, *X.shape)  # Add batch dimension
        
        # Get current price
        current_price = row.Close
        
        # Predicted price
        pred_price = model.predict(X_batch)[0][0]
        # Unscale prediction
        price_min = scaler.data_min_[0]
        price_max = scaler.data_max_[0]
        pred_price_unscaled = pred_price * (price_max - price_min) + price_min
        
        # Predict direction (if available)
        if hasattr(model, 'predict_direction') and model.direction_classifier is not None:
            direction = model.predict_direction(X_batch)[0]
            direction_str = "UP" if direction == 1 else "DOWN"
        else:
            direction_str = "N/A"
        
        # Make trading decision
        decision, amount, _ = agent.make_decision(X_batch, current_price, row.Date, processed_data)
        
        print(f"{i:4d} ${current_price:9.2f} ${pred_price_unscaled:9.2f} {direction_str:8} {decision:8} {amount:10.2f}")
    
    print("\nAgent final state:")
    print(f"Shares: {agent.shares}")
    print(f"Balance: ${agent.balance:.2f}")
    print(f"Transactions: {len(agent.transaction_history)}")
    
    return agent.transaction_history

if __name__ == "__main__":
    transactions = test_model_predictions()
    
    # If there are transactions, display them
    if transactions:
        print("\nTransaction History:")
        for i, tx in enumerate(transactions):
            print(f"{i+1}. {tx['action']} {tx['shares']} shares @ ${tx['price']:.2f} on {tx['date']}")
    else:
        print("\nNo transactions recorded")
