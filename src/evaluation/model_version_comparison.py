import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from src.models.sklearn_model import StockPredictionModel
from src.data.preprocess import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def compare_model_versions(model_paths, output_dir=None):
    """
    Compare different versions of trained models
    
    Parameters:
    - model_paths: List of paths to model files or a directory containing model files
    - output_dir: Directory to save comparison results
    
    Returns:
    - DataFrame with comparison metrics
    """
    # If model_paths is a directory, find all model files
    if isinstance(model_paths, str) and os.path.isdir(model_paths):
        model_files = []
        for root, dirs, files in os.walk(model_paths):
            for file in files:
                if file.endswith('.joblib') and not file.startswith('.'):
                    model_files.append(os.path.join(root, file))
        model_paths = model_files
    
    # Ensure we have models to compare
    if not model_paths:
        print("No model files found to compare")
        return None
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(project_root, "results", "model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Comparing {len(model_paths)} model versions")
    
    # Load test data once
    data_path = os.path.join(project_root, "data", "raw", "TSLA.csv")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(data_path, sequence_length)
    
    # Calculate true direction labels once
    y_test_prev = np.roll(y_test, 1)
    y_test_prev[0] = y_train[-1]  # Use last training value as previous for first test value
    y_test_dir = (y_test > y_test_prev).astype(int)
    
    # Prepare results container
    results = []
    
    # Evaluate each model
    for i, model_path in enumerate(model_paths):
        print(f"Evaluating model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        
        try:
            # Get model modified timestamp
            timestamp = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            # Load model
            model = StockPredictionModel(sequence_length, X_train.shape[2])
            model.load(model_path)
            
            # Check for direction classifier
            has_dir_classifier = hasattr(model, 'direction_classifier') and model.direction_classifier is not None
            dir_classifier_version = getattr(model, 'direction_classifier_version', 0)
            
            # Make price predictions
            X_test_2d = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
            price_pred = model.predict(X_test).flatten()
            
            # Regression metrics
            mse = mean_squared_error(y_test, price_pred)
            r2 = r2_score(y_test, price_pred)
            
            # Direction predictions
            if has_dir_classifier:
                dir_pred = model.predict_direction(X_test)
            else:
                price_pred_prev = np.roll(price_pred, 1)
                price_pred_prev[0] = y_train[-1]
                dir_pred = (price_pred > price_pred_prev).astype(int)
            
            # Direction metrics
            accuracy = accuracy_score(y_test_dir, dir_pred)
            precision = precision_score(y_test_dir, dir_pred, zero_division=0)
            recall = recall_score(y_test_dir, dir_pred, zero_division=0)
            f1 = f1_score(y_test_dir, dir_pred, zero_division=0)
            
            # Store results
            results.append({
                'model_path': model_path,
                'model_name': os.path.basename(model_path),
                'timestamp': timestamp,
                'has_direction_classifier': has_dir_classifier,
                'direction_classifier_version': dir_classifier_version,
                'mse': mse,
                'r2': r2,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
    
    # Convert to DataFrame and sort by timestamp
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('timestamp')
        
        # Save results
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Calculate improvements
        if len(results_df) > 1:
            # Add improvement columns
            first_r2 = results_df.iloc[0]['r2']
            first_f1 = results_df.iloc[0]['f1']
            results_df['r2_improvement'] = (results_df['r2'] - first_r2) / abs(first_r2) * 100
            results_df['f1_improvement'] = (results_df['f1'] - first_f1) / abs(first_f1) * 100
        
        # Plot results
        plot_comparison_results(results_df, output_dir)
        
        return results_df
    else:
        print("No valid models found to compare")
        return None

def plot_comparison_results(results_df, output_dir):
    """Plot model comparison results"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert timestamp to datetime for better plotting
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    
    # Plot regression metrics
    ax1.plot(results_df['timestamp'], results_df['r2'], 'o-', color='blue', label='R²')
    ax1.set_title('Regression Performance Over Model Versions', fontsize=14)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Format x-axis for better readability
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot classification metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, metric in enumerate(metrics):
        ax2.plot(results_df['timestamp'], results_df[metric], 'o-', 
                 color=colors[i], label=metric.capitalize())
    
    # Add direction classifier version if available
    if 'direction_classifier_version' in results_df.columns and results_df['direction_classifier_version'].nunique() > 1:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(results_df['timestamp'], results_df['direction_classifier_version'], 'o--', 
                      color='gray', label='Classifier Version')
        ax2_twin.set_ylabel('Classifier Version', fontsize=12)
        ax2_twin.legend(loc='upper right')
    
    ax2.set_title('Direction Prediction Performance Over Model Versions', fontsize=14)
    ax2.set_xlabel('Model Version Date', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_version_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    
    # Create an improvement summary plot if we have more than one model
    if len(results_df) > 1 and 'f1_improvement' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        plt.plot(results_df['timestamp'], results_df['r2_improvement'], 'o-', 
                 color='blue', label='R² Improvement')
        plt.plot(results_df['timestamp'], results_df['f1_improvement'], 'o-', 
                 color='purple', label='F1 Improvement')
        
        plt.title('Performance Improvement Over Model Versions', fontsize=14)
        plt.xlabel('Model Version Date', fontsize=12)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Format x-axis for better readability
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        improvement_path = os.path.join(output_dir, 'model_improvement.png')
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        print(f"Improvement plot saved to {improvement_path}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
        if os.path.isdir(model_path):
            print(f"Comparing all models in directory: {model_path}")
        else:
            print(f"Please provide a directory containing model files")
            sys.exit(1)
            
        compare_model_versions(model_path)
    else:
        print("Usage: python model_version_comparison.py <models_directory>")
        print("Example: python model_version_comparison.py models/advanced")
