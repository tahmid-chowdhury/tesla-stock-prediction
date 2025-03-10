import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import random

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.data.preprocess import preprocess_data
from src.models.model_trainer import ModelTrainer
from src.visualization.performance_visualizer import PerformanceVisualizer
from sklearn.utils import resample

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train direction-focused stock prediction model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--data-path', type=str, default='data/raw/TSLA.csv', help='Path to raw data')
    parser.add_argument('--refresh-data', action='store_true', help='Download fresh data')
    parser.add_argument('--model-dir', type=str, default='models/advanced', help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='logs/training', help='Directory to save logs')
    parser.add_argument('--notes', type=str, default='', help='Notes about this training run')
    parser.add_argument('--runs', type=int, default=1, help='Number of training runs to execute')
    parser.add_argument('--data-augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--direction-focus', action='store_true', help='Focus training on direction accuracy')
    
    return parser.parse_args()

def augment_data(X_train, y_train):
    """Generate augmented data samples for training"""
    print("Augmenting training data...")
    
    augmented_X = []
    augmented_y = []
    
    # First, add all original samples
    augmented_X.append(X_train)
    augmented_y.append(y_train)
    
    # Add noisy versions of the original data
    noise_levels = [0.01, 0.02, 0.03]
    for noise in noise_levels:
        # Add noise to the input features
        noisy_X = X_train + np.random.normal(0, noise, X_train.shape)
        augmented_X.append(noisy_X)
        augmented_y.append(y_train)  # Same targets
    
    # Add flipped samples (especially useful for down trends which may be underrepresented)
    # Get previous values to determine directions
    y_prev = np.roll(y_train, 1)
    y_prev[0] = y_train[0]
    
    # Identify up and down samples
    up_idx = np.where(y_train > y_prev)[0]
    down_idx = np.where(y_train <= y_prev)[0]
    
    # If there's significant imbalance, oversample the minority class
    if len(up_idx) > 2*len(down_idx) or len(down_idx) > 2*len(up_idx):
        minority_idx = down_idx if len(down_idx) < len(up_idx) else up_idx
        majority_idx = up_idx if len(down_idx) < len(up_idx) else down_idx
        
        # Oversample minority class
        minority_X = X_train[minority_idx]
        minority_y = y_train[minority_idx]
        
        # Determine how many samples to generate
        n_samples = len(majority_idx) - len(minority_idx)
        
        # Generate new samples with small variations
        for i in range(n_samples):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(minority_idx))
            
            # Add a small variation
            new_X = minority_X[idx] + np.random.normal(0, 0.02, minority_X[idx].shape)
            new_y = minority_y[idx]
            
            # Add to the dataset
            augmented_X.append(np.expand_dims(new_X, axis=0))
            augmented_y.append(np.array([new_y]))
    
    # Combine all augmented data
    combined_X = np.vstack(augmented_X)
    combined_y = np.concatenate(augmented_y)
    
    print(f"Original data shape: {X_train.shape}, Augmented data shape: {combined_X.shape}")
    return combined_X, combined_y

def modify_data_for_direction_focus(X_train, y_train):
    """
    Modify the dataset to better focus on direction prediction
    This helps the model learn direction changes rather than just price values
    """
    print("Modifying data for direction focus...")
    
    # Calculate previous values for direction detection
    y_prev = np.roll(y_train, 1)
    y_prev[0] = y_train[0]
    
    # Identify transitions (where direction changes) - these are more important for training
    direction_changes = np.where(np.sign(y_train - y_prev) != 
                                 np.sign(np.roll(y_train, -1) - y_train))[0]
    
    # Include some samples before and after each direction change
    important_indices = []
    for idx in direction_changes:
        # Add indices around the direction change point (3 before, 3 after if possible)
        for offset in range(-3, 4):
            check_idx = idx + offset
            if 0 <= check_idx < len(X_train):
                important_indices.append(check_idx)
    
    # Remove duplicates
    important_indices = np.unique(important_indices)
    
    # Get the direction change samples (oversample these)
    transition_X = X_train[important_indices]
    transition_y = y_train[important_indices]
    
    # Oversample transition points (3x)
    transition_X_repeated = np.repeat(transition_X, 3, axis=0)
    transition_y_repeated = np.repeat(transition_y, 3)
    
    # Combine with original data
    direction_focused_X = np.vstack([X_train, transition_X_repeated])
    direction_focused_y = np.concatenate([y_train, transition_y_repeated])
    
    print(f"Original data shape: {X_train.shape}, Direction-focused data shape: {direction_focused_X.shape}")
    print(f"Percentage of direction change samples: {len(important_indices)/len(X_train):.1%}")
    
    return direction_focused_X, direction_focused_y

def run_training(args, run_number=1):
    """Run a single training iteration with the given arguments"""
    # Calculate absolute path for data
    data_path = os.path.join(project_root, args.data_path)
    
    # Create run-specific directories if multiple runs
    if args.runs > 1:
        model_dir = os.path.join(args.model_dir, f"run_{run_number}")
        logs_dir = os.path.join(args.logs_dir, f"run_{run_number}")
    else:
        model_dir = args.model_dir
        logs_dir = args.logs_dir
        
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"Starting training run {run_number} of {args.runs} with {args.epochs} epochs")
    start_time = datetime.now()
    
    # Preprocess data
    print("Preprocessing data...")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(
        data_path, sequence_length=sequence_length, refresh_data=args.refresh_data
    )
    
    # Add data augmentation and variability based on the run number
    # Each run should use a slightly different version of the data
    if run_number > 1:
        # Add run-specific randomness
        np.random.seed(42 + run_number)
        random.seed(42 + run_number)
        
        # Slightly shuffle the order of training data for each run
        # Don't shuffle test data to keep evaluation consistent
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
    
    # Apply data augmentation if requested
    if args.data_augmentation:
        X_train, y_train = augment_data(X_train, y_train)
    
    # Modify data to focus on direction prediction
    if args.direction_focus:
        X_train, y_train = modify_data_for_direction_focus(X_train, y_train)
    
    # Split test data into validation and test sets
    test_idx = len(X_test) // 2
    X_val, X_test = X_test[:test_idx], X_test[test_idx:]
    y_val, y_test = y_test[:test_idx], y_test[test_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create model trainer with different random state for each run
    trainer = ModelTrainer(
        sequence_length=sequence_length, 
        n_features=X_train.shape[2],
        model_dir=model_dir,
        logs_dir=logs_dir
    )
    
    # Define parameter grid for hyperparameter tuning
    # Use a slightly different grid for each run to encourage diversity
    param_grid = {
        'rf': {
            'n_estimators': [50 + run_number*25, 100 + run_number*25, 200],
            'max_depth': [None, 20 + run_number*5, 50],
            'min_samples_split': [2, 5 + run_number, 10]
        },
        'gbr': {
            'n_estimators': [50, 100 + run_number*20, 200],
            'learning_rate': [0.01, 0.05, 0.1 + run_number*0.02],
            'max_depth': [3, 5 + run_number, 10]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0 + run_number]
        }
    }
    
    # Train model
    print(f"\nTraining model with {args.epochs} epochs...")
    run_notes = f"Run {run_number}/{args.runs}: {args.notes}"
    if args.data_augmentation:
        run_notes += " | With data augmentation"
    if args.direction_focus:
        run_notes += " | Direction-focused"
        
    training_results = trainer.train(
        X_train, y_train, X_val, y_val, 
        epochs=args.epochs, 
        param_grid=param_grid,
        notes=run_notes
    )
    
    # Evaluate on test set
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
    final_model_path = os.path.join(project_root, "models", f"advanced_model_run{run_number}.joblib")
    trainer.export_stock_prediction_model(final_model_path)
    
    # If it's the last run or only run, also save as the default model
    if run_number == args.runs:
        default_model_path = os.path.join(project_root, "models", "advanced_model.joblib")
        trainer.export_stock_prediction_model(default_model_path)
        print(f"Final run saved as default model: {default_model_path}")
    
    # Generate performance visualizations
    print("\nGenerating performance visualizations...")
    metrics_file = os.path.join(model_dir, "training_metrics.csv")
    visualizer = PerformanceVisualizer(metrics_file, os.path.join(logs_dir, "performance"))
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
    plot_path = os.path.join(logs_dir, "test_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print training time
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"\nTraining run {run_number} completed in {training_time}")
    print(f"Best epoch: {training_results['best_epoch']+1}")
    print(f"Best validation R²: {training_results['best_score']:.6f}")
    print(f"Final model saved to {final_model_path}")
    
    print("\nPerformance metrics:")
    print(f"  Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"  Test R²: {test_metrics['test_r2']:.6f}")
    print(f"  Direction Prediction - Accuracy: {test_metrics['classification_metrics']['accuracy']:.4f}")
    print(f"  Direction Prediction - F1: {test_metrics['classification_metrics']['f1']:.4f}")
    print(f"  Direction Prediction - Precision: {test_metrics['classification_metrics']['precision']:.4f}")
    print(f"  Direction Prediction - Recall: {test_metrics['classification_metrics']['recall']:.4f}")
    
    print(f"\nPerformance visualizations saved to {visualization_path}")
    
    return {
        'test_metrics': test_metrics,
        'training_time': training_time,
        'best_epoch': training_results['best_epoch'],
        'model_path': final_model_path
    }

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create main directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Run training for specified number of runs
    run_results = []
    for run_number in range(1, args.runs + 1):
        result = run_training(args, run_number)
        run_results.append(result)
    
    # If we ran multiple runs, combine the metrics for comparison
    if len(run_results) > 1:
        print("\n===== Multi-Run Summary =====")
        print(f"Completed {args.runs} training runs")
        
        # Create a table of results
        print("\nResults by Run:")
        print(f"{'Run':<5} {'F1 Score':<10} {'Accuracy':<10} {'Test R²':<10} {'Best Epoch':<12}")
        print("-" * 50)
        
        for i, result in enumerate(run_results):
            metrics = result['test_metrics']
            print(f"{i+1:<5} {metrics['classification_metrics']['f1']:.4f}      "
                  f"{metrics['classification_metrics']['accuracy']:.4f}      "
                  f"{metrics['test_r2']:.4f}      "
                  f"{result['best_epoch']+1}")
        
        # Calculate improvement from first to best run
        first_f1 = run_results[0]['test_metrics']['classification_metrics']['f1']
        best_f1 = max(r['test_metrics']['classification_metrics']['f1'] for r in run_results)
        best_f1_run = next(i+1 for i, r in enumerate(run_results) 
                         if r['test_metrics']['classification_metrics']['f1'] == best_f1)
        
        f1_improvement = (best_f1 - first_f1) / first_f1 * 100 if first_f1 > 0 else float('inf')
        
        print(f"\nBest F1 Score: {best_f1:.4f} (Run {best_f1_run})")
        print(f"Improvement over first run: {f1_improvement:.2f}%")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
