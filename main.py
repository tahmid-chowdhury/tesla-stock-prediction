import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from src.data.preprocess import preprocess_data
from src.models.lstm_model import StockPredictionModel, train_model
from src.agent.trading_agent import TradingAgent
from src.evaluation.backtesting import TradingSimulation

def main():
    # Set paths
    data_path = "data/raw/TSLA.csv"
    model_path = "models/lstm_model.h5"
    results_path = "results"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    
    # Step 1: Preprocess data
    print("Preprocessing data...")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(data_path, sequence_length)
    
    # Step 2: Train or load model
    print("Training model...")
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = StockPredictionModel(sequence_length, X_train.shape[2])
        model.load(model_path)
    else:
        print("Training new model...")
        model = train_model(X_train, y_train, X_test, y_test, sequence_length, X_train.shape[2])
    
    # Step 3: Initialize trading agent
    print("Initializing trading agent...")
    agent = TradingAgent(model, scaler)
    
    # Step 4: Run simulation for March 24-28, 2025
    print("Running trading simulation...")
    start_date = datetime(2025, 3, 24)
    end_date = datetime(2025, 3, 28)
    simulation = TradingSimulation(agent, processed_data, sequence_length, start_date, end_date)
    
    results = simulation.run_simulation()
    
    # Step 5: Visualize and save results
    print("Generating performance summary...")
    performance = simulation.get_performance_summary()
    
    print("\n===== Performance Summary =====")
    for key, value in performance.items():
        if 'pct' in key:
            print(f"{key}: {value:.2f}%")
        elif 'value' in key or 'balance' in key or 'fees' in key:
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Save results
    results.to_csv(os.path.join(results_path, "simulation_results.csv"), index=False)
    
    # Plot results
    print("Plotting results...")
    simulation.plot_results(save_path=os.path.join(results_path, "simulation_plot.png"))
    
    print("\nSimulation complete!")
    
if __name__ == "__main__":
    main()
