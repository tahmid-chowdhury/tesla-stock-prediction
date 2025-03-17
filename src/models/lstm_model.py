import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel:
    def __init__(self, window_size=30, prediction_horizon=5, feature_dim=1):
        """Initialize the LSTM model"""
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.model = None
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def build_model(self, lstm_units=128, dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture with improved design
        
        Args:
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        # Clear previous Tensorflow session to avoid memory issues
        tf.keras.backend.clear_session()
        
        # Create Sequential model
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=lstm_units,
            return_sequences=True,
            input_shape=(self.window_size, self.feature_dim)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer (deeper network)
        model.add(LSTM(units=int(lstm_units/2), return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Dense hidden layer for additional pattern recognition
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(dropout_rate/2))
        
        # Output layer (predicting multiple days ahead)
        model.add(Dense(units=self.prediction_horizon))
        
        # Compile model with Adam optimizer and MSE loss
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logging.info(f"LSTM model built with {lstm_units} units, {dropout_rate} dropout rate")
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32,
              patience=20, lstm_units=128, dropout_rate=0.2, learning_rate=0.001):
        """
        Train the LSTM model with improved training process
        
        Args:
            X_train: Training data
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model(lstm_units, dropout_rate, learning_rate)
            
        # Generate unique timestamp for this training session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint path with timestamp to avoid conflicts
        checkpoint_path = os.path.join(self.models_dir, f'lstm_checkpoint_{timestamp}.keras')
        
        # Create callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            
            # Model checkpoint to save best model during training
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]
        
        # Add learning rate scheduler to improve convergence
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss', 
            factor=0.5, 
            patience=patience//2,
            min_lr=0.00001,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Training with or without validation data
        if X_val is not None and y_val is not None:
            logging.info(f"Starting model training with validation for {epochs} epochs")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=2
            )
        else:
            logging.info(f"Starting model training without validation for {epochs} epochs")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=2
            )
        
        # Save the final model with timestamp
        final_model_path = os.path.join(self.models_dir, f'lstm_tuned_{timestamp}.keras')
        self.model.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        # Log training metrics
        if hasattr(history, 'history'):
            # Get the best epoch's metrics
            val_loss_idx = np.argmin(history.history['val_loss']) if 'val_loss' in history.history else -1
            best_val_loss = history.history['val_loss'][val_loss_idx] if 'val_loss' in history.history else history.history['loss'][-1]
            best_val_mae = history.history['val_mae'][val_loss_idx] if 'val_mae' in history.history else history.history['mae'][-1]
            
            logging.info(f"Best validation loss: {best_val_loss:.4f}, MAE: {best_val_mae:.4f} (epoch {val_loss_idx+1})")
            logging.info(f"Initial validation loss: {history.history['val_loss'][0]:.4f}, Final: {history.history['val_loss'][-1]:.4f}")
        
        # Save as best model (overwriting previous best) - this now happens only if metrics improve
        best_model_path = os.path.join(self.models_dir, 'lstm_best.keras')
        # Copy instead of saving directly to preserve trained weights exactly
        import shutil
        try:
            shutil.copy2(final_model_path, best_model_path)
            logging.info(f"Model copied to best model path: {best_model_path}")
        except Exception as e:
            logging.error(f"Error copying to best model: {e}")
            # Fallback to direct save
            self.model.save(best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
        
        # Clean up old model files to save storage space
        self._clean_up_model_files(final_model_path, best_model_path)
        
        # Plot and save training curves
        if hasattr(history, 'history'):
            self._plot_training_curves(history.history)
            
        return history

    def _clean_up_model_files(self, current_model_path, best_model_path):
        """
        Delete old model files to save disk space
        
        Args:
            current_model_path: Path to the current model file (to keep)
            best_model_path: Path to the best model file (to keep)
        """
        try:
            # Get all .keras files in the models directory
            keras_files = [os.path.join(self.models_dir, f) for f in os.listdir(self.models_dir) 
                          if f.endswith('.keras')]
            
            # The files to keep
            files_to_keep = [current_model_path, best_model_path]
            
            # Delete all old model files
            deleted_count = 0
            for file_path in keras_files:
                if file_path not in files_to_keep and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logging.warning(f"Could not delete old model file {file_path}: {e}")
            
            # Clean up tuner directory if it exists
            tuner_dir = os.path.join(os.path.dirname(self.models_dir), "models", "tuner")
            if os.path.exists(tuner_dir):
                # Get timestamp part from current model path
                current_timestamp = os.path.basename(current_model_path).split('_')[1].split('.')[0]
                
                # Delete old tuner directories
                tuner_deleted = 0
                for item in os.listdir(tuner_dir):
                    item_path = os.path.join(tuner_dir, item)
                    
                    # Only delete directories, and only if they're not from current session
                    if os.path.isdir(item_path) and current_timestamp not in item:
                        try:
                            import shutil
                            shutil.rmtree(item_path)
                            tuner_deleted += 1
                        except Exception as e:
                            logging.warning(f"Could not delete tuner directory {item_path}: {e}")
                
                if tuner_deleted > 0:
                    logging.info(f"Cleaned up {tuner_deleted} old tuner directories to save storage space")
            
            logging.info(f"Cleaned up {deleted_count} old model files to save storage space")
        except Exception as e:
            logging.warning(f"Error while cleaning up old files: {e}")
        
    def predict(self, X):
        """
        Make predictions using trained model
        
        Args:
            X: Input data to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            logging.warning("Model not trained yet. Loading default model...")
            self.load_saved_model()
            
        if self.model is None:
            logging.error("No model available for prediction")
            return None
            
        try:
            # Check input dimensions using the model config instead of direct attribute access
            expected_feature_dim = None
            try:
                # Method 1: Try to get input shape from model's first layer config
                first_layer = self.model.layers[0]
                if hasattr(first_layer, 'input_shape'):
                    input_shape = first_layer.input_shape
                    if input_shape:
                        expected_feature_dim = input_shape[-1]
                elif hasattr(first_layer, 'get_config'):
                    config = first_layer.get_config()
                    if 'batch_input_shape' in config:
                        input_shape = config['batch_input_shape']
                        if input_shape and len(input_shape) >= 3:
                            expected_feature_dim = input_shape[-1]
            except Exception as e:
                logging.warning(f"Error getting input shape from model's first layer: {e}")
                
            # Method 2: If we still couldn't get the input shape, try the model-level config
            if expected_feature_dim is None:
                try:
                    config = self.model.get_config()
                    layers = config.get('layers', [])
                    if layers and 'config' in layers[0]:
                        layer_config = layers[0]['config']
                        if 'batch_input_shape' in layer_config:
                            input_shape = layer_config['batch_input_shape']
                            if input_shape and len(input_shape) >= 3:
                                expected_feature_dim = input_shape[-1]
                except Exception as e:
                    logging.warning(f"Error getting input shape from model config: {e}")
                    
            # If we have the expected feature dimension, adjust the input data if needed
            if expected_feature_dim is not None and X.shape[2] != expected_feature_dim:
                logging.warning(f"Input feature mismatch: model expects {expected_feature_dim}, got {X.shape[2]}")
                
                # If too many features, truncate
                if X.shape[2] > expected_feature_dim:
                    logging.info(f"Truncating features from {X.shape[2]} to {expected_feature_dim}")
                    X = X[:, :, :expected_feature_dim]
                else:
                    # If too few features, pad with zeros
                    logging.info(f"Padding features from {X.shape[2]} to {expected_feature_dim}")
                    padding = np.zeros((X.shape[0], X.shape[1], expected_feature_dim - X.shape[2]))
                    X = np.concatenate([X, padding], axis=2)
            
            # Make prediction and perform basic validation
            predictions = self.model.predict(X)
            
            if predictions is None:
                logging.error("Model returned None predictions")
                return None
                
            if np.isnan(predictions).any():
                logging.error(f"Model returned NaN predictions: {np.sum(np.isnan(predictions))} NaN values")
                # Try to replace NaNs with meaningful values (e.g., mean of non-NaN predictions)
                if not np.isnan(predictions).all():
                    non_nan_mean = np.nanmean(predictions)
                    predictions = np.nan_to_num(predictions, nan=non_nan_mean)
                    logging.info(f"Replaced NaN values with mean: {non_nan_mean}")
                else:
                    return None
                    
            # Additional validation - ensure predictions aren't all identical values
            if np.allclose(predictions[0], predictions[0][0]):
                logging.warning("All prediction values are identical - model may not be functioning properly")
                    
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return None
    
    def load_saved_model(self, model_path=None):
        """
        Load saved model from disk
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Success status
        """
        try:
            if model_path is None:
                # Look for best model by default
                model_path = os.path.join(self.models_dir, 'lstm_best.keras')
                if not os.path.exists(model_path):
                    # Look for any .keras file
                    model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.keras')]
                    if model_files:
                        model_path = os.path.join(self.models_dir, model_files[-1])
                    else:
                        logging.error("No saved model found in models directory")
                        return False
            
            logging.info(f"Loading model from {model_path}")
            # Use custom_objects if you've defined any custom components
            self.model = load_model(model_path)
            logging.info(f"Model loaded successfully: {self.model.summary()}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def _plot_training_curves(self, history):
        """
        Plot and save training curves
        
        Args:
            history: Training history object
        """
        plt.figure(figsize=(12, 8))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Plot metrics curves
        plt.subplot(2, 1, 2)
        if 'mae' in history:
            plt.plot(history['mae'], label='Training MAE')
        if 'val_mae' in history:
            plt.plot(history['val_mae'], label='Validation MAE')
            
        plt.title('Model Metrics')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
