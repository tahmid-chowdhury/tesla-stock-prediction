import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HyperparameterTuner:
    def __init__(self, window_size=30, prediction_horizon=5, feature_dim=None, max_trials=10):
        """
        Initialize Hyperparameter Tuner
        
        Args:
            window_size: Size of the input sequence
            prediction_horizon: Number of days to predict
            feature_dim: Number of features in input data
            max_trials: Maximum number of trials for hyperparameter tuning
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.max_trials = max_trials
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        self.tuner_dir = os.path.join(self.models_dir, "tuner")
        
        # Create directories if they don't exist
        os.makedirs(self.tuner_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def build_model(self, hp):
        """
        Build a model with tunable hyperparameters
        
        Args:
            hp: Keras Tuner hyperparameters
            
        Returns:
            Compiled Keras model
        """
        if self.feature_dim is None:
            raise ValueError("Feature dimension must be set before building the model")
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.window_size, self.feature_dim)))
        
        # Architecture choice: standard LSTM vs Bidirectional LSTM
        use_bidirectional = hp.Boolean('bidirectional', default=False)
        
        # First LSTM layer
        lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
        return_sequences = hp.Boolean('return_sequences', default=True)
        
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(units=lstm_units_1, return_sequences=return_sequences)
            ))
        else:
            model.add(LSTM(units=lstm_units_1, return_sequences=return_sequences))
        
        # Dropout rate after first LSTM
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_1))
        
        # Add second LSTM layer conditionally
        if return_sequences:
            lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=96, step=16)
            
            if use_bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units_2)
                ))
            else:
                model.add(LSTM(units=lstm_units_2))
            
            # Dropout rate after second LSTM
            dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
            model.add(Dropout(dropout_2))
        
        # Optional dense layer before output
        if hp.Boolean('use_dense_layer', default=True):
            dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=8)
            activation = hp.Choice('dense_activation', values=['relu', 'tanh', 'selu'])
            model.add(Dense(dense_units, activation=activation))
            
            if hp.Boolean('use_second_dropout', default=False):
                dropout_3 = hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)
                model.add(Dropout(dropout_3))
        
        # Output layer
        model.add(Dense(self.prediction_horizon))
        
        # Compile model
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 3e-3])
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
        
        if optimizer_choice == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        
        return model
    
    def tune(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Tune hyperparameters using Keras Tuner
        
        Args:
            X_train: Training data
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets
            epochs: Maximum epochs for training
            batch_size: Batch size
            
        Returns:
            Best model
        """
        if self.feature_dim is None and X_train is not None:
            self.feature_dim = X_train.shape[2]
            
        # Create a unique directory name for this tuning session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tuner_dir = os.path.join(self.tuner_dir, f"tuning_{timestamp}")
            
        tuner = kt.Hyperband(
            self.build_model,
            objective='val_loss',
            max_epochs=epochs,
            factor=3,
            directory=tuner_dir,
            project_name='lstm_tuning'
        )
        
        # Define early stopping callback
        stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Search for best hyperparameters
        try:
            logging.info(f"Starting hyperparameter search with {self.max_trials} trials")
            # Track the best val_loss from previous trials to show improvement
            best_val_loss = float('inf')
            
            class PrintResultCallback(kt.callbacks.Callback):
                def on_trial_end(callback_self, trial, logs=None):
                    if logs:
                        val_loss = logs.get('val_loss', float('inf'))
                        if val_loss < best_val_loss:
                            improvement = "✓ IMPROVED"
                        else:
                            improvement = "✗ no improvement"
                        
                        logging.info(f"Trial {trial.trial_id} completed - val_loss: {val_loss:.4f} ({improvement})")
            
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[stop_early, PrintResultCallback()]
            )
            
            # Get the best hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            # Log the best hyperparameters
            logging.info("Best hyperparameters found:")
            for param, value in best_hps.values.items():
                logging.info(f"  {param}: {value}")
            
            # Build the model with the best hyperparameters
            best_model = tuner.hypermodel.build(best_hps)
            
            # Train the model with the best hyperparameters
            history = best_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[stop_early]
            )
            
            # Save the best model
            best_model_path = os.path.join(self.models_dir, f"lstm_tuned_{timestamp}.keras")
            best_model.save(best_model_path)
            logging.info(f"Tuned model saved to {best_model_path}")
            
            # Save hyperparameter info for reference
            with open(os.path.join(self.tuner_dir, f"best_params_{timestamp}.txt"), 'w') as f:
                for param, value in best_hps.values.items():
                    f.write(f"{param}: {value}\n")
            
            # Clean up old tuner files to save disk space
            self._clean_up_tuner_files(timestamp)
            
            return best_model, history, best_hps
            
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            return None, None, None
    
    def _clean_up_tuner_files(self, current_timestamp):
        """
        Delete old tuner directories and files to save disk space
        
        Args:
            current_timestamp: Current timestamp to identify files to keep
        """
        try:
            # Keep only the current tuning session and delete old ones
            deleted_dirs = 0
            
            # Get all directories in tuner_dir
            for item in os.listdir(self.tuner_dir):
                item_path = os.path.join(self.tuner_dir, item)
                
                # Skip current session files
                if current_timestamp in item:
                    continue
                    
                # Skip non-directories and important files
                if not os.path.isdir(item_path) and not item.endswith('.txt'):
                    continue
                
                # Delete old tuning directories
                try:
                    if os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    deleted_dirs += 1
                except Exception as e:
                    logging.warning(f"Could not delete old tuner file {item_path}: {e}")
            
            if deleted_dirs > 0:
                logging.info(f"Cleaned up {deleted_dirs} old tuner files/directories to save storage space")
        except Exception as e:
            logging.warning(f"Error while cleaning up tuner files: {e}")
