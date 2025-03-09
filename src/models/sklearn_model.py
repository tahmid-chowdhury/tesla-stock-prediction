import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class StockPredictionModel:
    def __init__(self, sequence_length, n_features, model_type='random_forest'):
        """
        Initialize stock prediction model using scikit-learn
        
        Parameters:
        - sequence_length: Length of input sequences 
        - n_features: Number of features
        - model_type: 'random_forest' or 'ridge'
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = Ridge(alpha=1.0)
            
    def predict(self, X):
        """
        Make predictions with the model
        
        Parameters:
        - X: Input data with shape (batch_size, sequence_length, n_features)
        
        Returns:
        - Predictions with shape (batch_size, 1)
        """
        # Reshape for sklearn: (batch_size, sequence_length, n_features) -> (batch_size, sequence_length * n_features)
        batch_size = X.shape[0]
        X_reshaped = X.reshape(batch_size, -1)
        
        # Make predictions
        y_pred = self.model.predict(X_reshaped)
        
        # Return as 2D array to match TensorFlow model output format
        return y_pred.reshape(-1, 1)
    
    def save(self, filepath):
        """Save the model to disk"""
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a dictionary with model and metadata
        model_data = {
            'model': self.model,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'model_type': self.model_type
        }
        
        # Save with joblib
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model from disk"""
        # Load model data dictionary
        model_data = joblib.load(filepath)
        
        # Set attributes
        self.model = model_data['model']
        self.sequence_length = model_data['sequence_length']
        self.n_features = model_data['n_features']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")


def train_model(X_train, y_train, X_test, y_test, sequence_length, n_features, 
                model_type='random_forest', save_path=None):
    """
    Train the stock prediction model
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - sequence_length: Length of input sequences
    - n_features: Number of features
    - model_type: 'random_forest' or 'ridge'
    - save_path: Where to save the trained model
    
    Returns:
    - Trained model
    """
    # Initialize model
    model = StockPredictionModel(sequence_length, n_features, model_type)
    
    # Reshape data for sklearn: (samples, sequence_length, features) -> (samples, sequence_length * features)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    # Train model
    print(f"Training {model_type} model...")
    model.model.fit(X_train_reshaped, y_train)
    
    # Evaluate model
    y_pred_train = model.model.predict(X_train_reshaped)
    y_pred_test = model.model.predict(X_test_reshaped)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
    # Create evaluation plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.3)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Training Data')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Test Data')
    
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plot_dir = os.path.dirname(save_path)
        plt.savefig(os.path.join(plot_dir, "model_evaluation.png"))
    
    # Save model if save_path is provided
    if save_path:
        model.save(save_path)
    
    plt.show()
    
    return model
