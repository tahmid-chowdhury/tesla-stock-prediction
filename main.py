import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.data.preprocess import preprocess_data

# Import sklearn_model instead of lstm_model
from src.models.sklearn_model import StockPredictionModel, train_model
from src.agent.trading_agent import TradingAgent
from src.evaluation.backtesting import TradingSimulation

# Import new forecast utilities
from src.utils.date_utils import generate_future_trading_days, get_last_trading_day
from src.forecasting.price_forecaster import StockForecaster

# Add imports for new features
from src.evaluation.strategy_comparison import compare_strategies
from src.visualization.dashboard import create_dashboard
from src.debug.test_direction_prediction import test_direction_prediction_accuracy

def main():
    # Set paths
    data_path = "data/raw/TSLA.csv"
    model_path = "models/advanced_model.joblib"
    results_path = "results"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Step 1: Preprocess data
    print("Preprocessing data...")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(data_path, sequence_length)
    
    # Step 2: Train or load model
    print("Loading model...")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = StockPredictionModel(sequence_length, X_train.shape[2])
        model.load(model_path)
        
        # Check if model has direction classifier
        has_direction_classifier = hasattr(model, 'direction_classifier') and model.direction_classifier is not None
        if has_direction_classifier:
            print(f"Model includes direction classifier (version: {getattr(model, 'direction_classifier_version', 'unknown')})")
        else:
            print("Model does not have a direction classifier")
    else:
        print("Model not found. Please train a model first using train_advanced_model.py or train_direction_focused_model.py")
        print("Example: python src/models/train_direction_focused_model.py --epochs 20 --direction-focus")
        return
    
    # Test direction prediction accuracy (optional - uncomment to run)
    # test_direction_prediction_accuracy(model_path)
    
    # Step 3: Initialize trading agent
    print("Initializing trading agent...")
    agent = TradingAgent(model, scaler)
    
    # Step 4: Generate next 5 trading days dates
    print("Generating forecast for next 5 trading days...")
    last_trading_day = get_last_trading_day()
    future_dates = generate_future_trading_days(last_trading_day, 5)
    
    print(f"Last trading day: {last_trading_day.strftime('%Y-%m-%d')}")
    print(f"Future trading days: {[d.strftime('%Y-%m-%d') for d in future_dates]}")
    
    # Step 5: Create forecaster and generate future data
    feature_columns = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 
                      'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volatility', 'Return']
    
    forecaster = StockForecaster(model, scaler, sequence_length, feature_columns)
    
    # Add is_forecast flag to processed data
    processed_data['is_forecast'] = False
    
    # Generate forecast data
    forecast_data = forecaster.forecast_next_days(processed_data, future_dates)
    
    # Combine historical and forecast data
    combined_data = pd.concat([processed_data, forecast_data], ignore_index=True)
    
    # Step 6: Run simulation for future dates
    print("Running trading simulation for future dates...")
    start_date = future_dates[0]
    end_date = future_dates[-1]
    
    print(f"Simulation period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    simulation = TradingSimulation(agent, combined_data, sequence_length, start_date, end_date)
    results = simulation.run_simulation()
    
    # Make sure we have valid results before continuing
    if results.empty:
        print("Error: No data available for simulation in the selected date range.")
        return
    
    # Step 7: Visualize and save results
    print("Generating performance summary...")
    performance = simulation.get_performance_summary()
    
    print("\n===== Performance Summary (FORECAST) =====")
    for key, value in performance.items():
        if 'pct' in key:
            print(f"{key}: {value:.2f}%")
        elif 'value' in key or 'balance' in key or 'fees' in key:
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    results.to_csv(os.path.join(results_path, "forecast_results.csv"), index=False)
    
    # Plot results
    print("Plotting results...")
    simulation.plot_results(save_path=os.path.join(results_path, "forecast_plot.png"))
    
    # After running simulation and plotting results:
    
    # Create strategy comparison
    print("\nComparing trading strategies...")
    historical_data = processed_data[processed_data['is_forecast'] == False].copy()
    strategy_results, strategy_performance = compare_strategies(
        historical_data, 
        forecast_data,
        agent_results=results,
        save_path=os.path.join(results_path, "strategy_comparison.png")
    )
    
    # Create comprehensive dashboard
    print("\nGenerating dashboard...")
    create_dashboard(
        historical_data.tail(30),  # Last 30 days of historical data
        forecast_data,
        results,
        performance,
        save_path=os.path.join(results_path, "dashboard.png")
    )
    
    print("\nAll visualizations and comparisons complete!")
    
if __name__ == "__main__":
    main()
