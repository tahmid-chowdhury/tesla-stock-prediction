import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.data.preprocess import preprocess_data
from src.models.model_trainer import ModelTrainer
from src.visualization.performance_visualizer import PerformanceVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train advanced stock prediction model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data-path', type=str, default='data/raw/TSLA.csv', help='Path to raw data')
    parser.add_argument('--refresh-data', action='store_true', help='Download fresh data')
    parser.add_argument('--model-dir', type=str, default='models/advanced', help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='logs/training', help='Directory to save logs')
    parser.add_argument('--notes', type=str, default='', help='Notes about this training run')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Calculate absolute path for data
    data_path = os.path.join(project_root, args.data_path)
    
    print(f"Starting advanced model training with {args.epochs} epochs")
    start_time = datetime.now()
    
    # Preprocess data
    print("Preprocessing data...")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(
        data_path, sequence_length=sequence_length, refresh_data=args.refresh_data
    )
    
    # Split test data into validation and test sets
    test_idx = len(X_test) // 2
    X_val, X_test = X_test[:test_idx], X_test[test_idx:]
    y_val, y_test = y_test[:test_idx], y_test[test_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create model trainer
    trainer = ModelTrainer(
        sequence_length=sequence_length, 
        n_features=X_train.shape[2],
        model_dir=args.model_dir,
        logs_dir=args.logs_dir
    )
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 20, 50],
            'min_samples_split': [2, 5, 10]
        },
        'gbr': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0]
        }
    }
    
    # Train model with notes
    print(f"\nTraining model with {args.epochs} epochs...")
    training_results = trainer.train(
        X_train, y_train, X_val, y_val, 
        epochs=args.epochs, 
        param_grid=param_grid,
        notes=args.notes
    )
    
    # Evaluate on test set and update metrics
    print("\nEvaluating on test set...")
    X_test_2d = trainer.reshape_data(X_test)
    test_predictions = trainer.predict(X_test)
    
    # Update metrics with test results
    test_metrics = trainer.update_test_metrics(y_test, test_predictions)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R²: {test_r2:.6f}")
    
    # Export final model in format compatible with the rest of the system
    print("\nExporting final model...")
    final_model_path = os.path.join(project_root, "models", "advanced_model.joblib")
    trainer.export_stock_prediction_model(final_model_path)
    
    # Generate performance visualizations
    print("\nGenerating performance visualizations...")
    metrics_file = os.path.join(args.model_dir, "training_metrics.csv")
    visualizer = PerformanceVisualizer(metrics_file, os.path.join(args.logs_dir, "performance"))
    visualization_path = visualizer.visualize_metrics_over_time()
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(test_predictions, label='Predicted')
    plt.title('Test Set: Actual vs Predicted Stock Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(args.logs_dir, "test_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print training time
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time}")
    print(f"Best epoch: {training_results['best_epoch']+1}")
    print(f"Best validation R²: {training_results['best_score']:.6f}")
    print(f"Final model saved to {final_model_path}")
    print(f"Evaluation plot saved to {plot_path}")
    
    print("\nPerformance metrics:")
    print(f"  Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"  Test R²: {test_metrics['test_r2']:.6f}")
    print(f"  Direction Prediction - Accuracy: {test_metrics['classification_metrics']['accuracy']:.4f}")
    print(f"  Direction Prediction - F1: {test_metrics['classification_metrics']['f1']:.4f}")
    print(f"  Direction Prediction - Precision: {test_metrics['classification_metrics']['precision']:.4f}")
    print(f"  Direction Prediction - Recall: {test_metrics['classification_metrics']['recall']:.4f}")
    
    print(f"\nPerformance visualizations saved to {visualization_path}")
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
