import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel:
    def __init__(self, window_size=30, prediction_horizon=5, feature_dim=None):
        """
        Initialize LSTM model
        
        Args:
            window_size: Size of input sequence
            prediction_horizon: Number of days to predict
            feature_dim: Number of features
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.model = None
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def build_model(self):
        """
        Build LSTM model architecture
        """
        if self.feature_dim is None:
            raise ValueError("Feature dimension must be set before building the model")
            
        model = Sequential([
            # First LSTM layer with return sequences for stacking
            LSTM(units=64, return_sequences=True, 
                 input_shape=(self.window_size, self.feature_dim)),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=32),
            Dropout(0.2),
            
            # Output dense layer (prediction_horizon days forecast)
            Dense(self.prediction_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        logging.info("LSTM model compiled successfully")
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the LSTM model
        """
        if self.model is None:
            if self.feature_dim is None and X_train is not None:
                self.feature_dim = X_train.shape[2]
            self.build_model()
            
        # If validation data not provided, use a split from training data
        if X_val is None or y_val is None:
            val_split = 0.2
            validation_data = None
        else:
            val_split = 0.0
            validation_data = (X_val, y_val)
            
        # Callbacks for early stopping and model checkpointing
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, "lstm_best.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        self.model.save(os.path.join(self.models_dir, "lstm_final.h5"))
        logging.info("Model trained and saved successfully")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
        
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        if self.model is None:
            logging.error("Model not trained yet")
            return None
            
        predictions = self.model.predict(X)
        return predictions
        
    def load_saved_model(self, model_path=None):
        """
        Load a saved model
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, "lstm_best.h5")
            
        if not os.path.exists(model_path):
            logging.error(f"Model file {model_path} not found")
            return False
            
        try:
            # Custom objects to handle serialization issues
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError()
            }
            self.model = load_model(model_path, custom_objects=custom_objects)
            logging.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Try to rebuild and compile the model as a fallback
            try:
                logging.info("Attempting to rebuild and compile the model...")
                if self.feature_dim is None:
                    self.feature_dim = 20  # Default value based on processed data
                self.build_model()
                logging.info("Model rebuilt successfully. Note: The model will not have trained weights.")
                return True
            except Exception as rebuild_err:
                logging.error(f"Failed to rebuild model: {rebuild_err}")
                return False
    
    def evaluate(self, X_test, y_test, price_scaler=None):
        """
        Evaluate model performance on test data
        """
        if self.model is None:
            logging.error("Model not trained yet")
            return None
            
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # If a scaler is provided, inverse transform predictions and actual values
        if price_scaler is not None:
            predictions = price_scaler.inverse_transform(predictions)
            y_test = price_scaler.inverse_transform(y_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        
        # Calculate accuracy metrics for direction prediction
        direction_accuracy = self.calculate_direction_accuracy(y_test, predictions)
        
        # Plot predictions vs actual
        self.plot_predictions(y_test, predictions)
        
        # Log and return results
        logging.info(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
        logging.info(f"Direction Accuracy: {direction_accuracy}")
        
        return {
            'mse': mse,
            'rmse': rmse, 
            'mae': mae,
            'direction_accuracy': direction_accuracy
        }
        
    def calculate_direction_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy of predicting price movement direction
        """
        # Get price changes for actual and predicted
        y_true_dir = np.diff(y_true, axis=1) > 0
        y_pred_dir = np.diff(y_pred, axis=1) > 0
        
        # Calculate accuracy
        correct_dir = (y_true_dir == y_pred_dir).mean()
        return correct_dir
        
    def plot_training_history(self, history):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
        plt.close()
        
    def plot_predictions(self, y_true, y_pred, samples=5):
        """
        Plot predictions vs actual values
        """
        plt.figure(figsize=(15, 10))
        
        # Plot a sample of predictions vs actual
        for i in range(min(samples, len(y_true))):
            plt.subplot(samples, 1, i+1)
            plt.plot(y_true[-samples+i], label='Actual')
            plt.plot(y_pred[-samples+i], label='Predicted')
            plt.title(f'Sample {i+1}')
            plt.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'predictions.png'))
        plt.close()
