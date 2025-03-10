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
from src.data.preprocess import preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def test_direction_prediction_accuracy(model_path=None):
    """Test direction prediction accuracy of the model"""
    print("Testing direction prediction accuracy...")
    
    # Load data
    data_path = os.path.join(project_root, "data", "raw", "TSLA.csv")
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler, processed_data = preprocess_data(data_path, sequence_length)
    
    # Default model path if not provided
    if model_path is None:
        model_path = os.path.join(project_root, "models", "advanced_model.joblib")
        
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}")
    model = StockPredictionModel(sequence_length, X_train.shape[2])
    model.load(model_path)
    
    # Check if model has direction classifier
    has_direction_classifier = hasattr(model, 'direction_classifier') and model.direction_classifier is not None
    print(f"Model has direction classifier: {has_direction_classifier}")
    if has_direction_classifier:
        print(f"Direction classifier version: {getattr(model, 'direction_classifier_version', 'unknown')}")
    
    # Calculate actual directions in test data
    y_test_prev = np.roll(y_test, 1)
    y_test_prev[0] = y_train[-1]  # Use last training value as previous for first test value
    y_test_dir = (y_test > y_test_prev).astype(int)
    
    # Make price predictions
    X_test_2d = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
    price_pred = model.predict(X_test).flatten()
    
    # Make direction predictions - two ways:
    # 1. Using dedicated direction classifier if available
    if hasattr(model, 'predict_direction') and has_direction_classifier:
        dir_pred = model.predict_direction(X_test)
        print("Made direction predictions using dedicated classifier")
    else:
        # 2. Deriving from price predictions
        price_pred_prev = np.roll(price_pred, 1)
        price_pred_prev[0] = y_train[-1]
        dir_pred = (price_pred > price_pred_prev).astype(int)
        print("Derived direction predictions from price predictions")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_dir, dir_pred)
    precision = precision_score(y_test_dir, dir_pred, zero_division=0)
    recall = recall_score(y_test_dir, dir_pred, zero_division=0)
    f1 = f1_score(y_test_dir, dir_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test_dir, dir_pred)
    
    # Print results
    print("\nDirection Prediction Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print("             Predicted")
    print("            Down   Up")
    print(f"Actual Down  {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
    print(f"       Up    {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
    
    # Calculate direction distribution
    actual_up = np.sum(y_test_dir)
    actual_down = len(y_test_dir) - actual_up
    predicted_up = np.sum(dir_pred)
    predicted_down = len(dir_pred) - predicted_up
    
    print("\nDirection Distribution:")
    print(f"Actual:    Up: {actual_up} ({actual_up/len(y_test_dir):.1%}), Down: {actual_down} ({actual_down/len(y_test_dir):.1%})")
    print(f"Predicted: Up: {predicted_up} ({predicted_up/len(dir_pred):.1%}), Down: {predicted_down} ({predicted_down/len(dir_pred):.1%})")
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Actual vs Predicted Directions
    plt.subplot(2, 1, 1)
    plt.plot(y_test_dir, 'b-', label='Actual Direction')
    plt.plot(dir_pred, 'r-', alpha=0.7, label='Predicted Direction')
    plt.title('Actual vs Predicted Price Direction')
    plt.xlabel('Time Step')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Highlight Incorrect Predictions
    plt.subplot(2, 1, 2)
    correct = y_test_dir == dir_pred
    plt.scatter(range(len(y_test_dir)), y_test_dir, c=['g' if c else 'r' for c in correct], alpha=0.7)
    plt.title('Direction Prediction Accuracy')
    plt.xlabel('Time Step')
    plt.ylabel('Actual Direction (Green=Correct, Red=Incorrect)')
    plt.yticks([0, 1], ['Down', 'Up'])
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(project_root, "results", "direction_prediction_test.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

if __name__ == "__main__":
    # Check if model path is provided as argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        test_direction_prediction_accuracy(model_path)
    else:
        test_direction_prediction_accuracy()
