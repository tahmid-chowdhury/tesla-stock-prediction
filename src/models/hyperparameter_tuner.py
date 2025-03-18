import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import keras_tuner as kt
import logging
import traceback
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
        
        # Create model following best practices
        model = Sequential()
        
        # Define input shape properly using Input layer
        model.add(Input(shape=(self.window_size, self.feature_dim), name="input_layer"))
        
        # Architecture choice: standard LSTM vs Bidirectional LSTM
        use_bidirectional = hp.Boolean('bidirectional', default=False)
        
        # First LSTM layer
        lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
        return_sequences = hp.Boolean('return_sequences', default=True)
        
        # Add LSTM layer with proper naming to help with debugging
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(units=lstm_units_1, return_sequences=return_sequences),
                name="bidirectional_lstm_1"
            ))
        else:
            model.add(LSTM(
                units=lstm_units_1, 
                return_sequences=return_sequences,
                name="lstm_1"
            ))
        
        # Dropout rate after first LSTM
        dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
        model.add(Dropout(dropout_1, name="dropout_1"))
        
        # Add second LSTM layer conditionally
        if return_sequences:
            lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=96, step=16)
            
            if use_bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units_2, name="lstm_2_inner"),
                    name="bidirectional_lstm_2"
                ))
            else:
                model.add(LSTM(units=lstm_units_2, name="lstm_2"))
            
            # Dropout rate after second LSTM
            dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
            model.add(Dropout(dropout_2, name="dropout_2"))
        
        # Optional dense layer before output
        if hp.Boolean('use_dense_layer', default=True):
            dense_units = hp.Int('dense_units', min_value=8, max_value=64, step=8)
            activation = hp.Choice('dense_activation', values=['relu', 'tanh', 'selu'])
            model.add(Dense(dense_units, activation=activation, name="dense"))
            
            if hp.Boolean('use_second_dropout', default=False):
                dropout_3 = hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)
                model.add(Dropout(dropout_3, name="dropout_3"))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, name="output"))
        
        # Compile model
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 3e-3])
        optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
        
        if optimizer_choice == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            
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
        
        try:
            # Debug info
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            logging.info(f"Feature dimension: {self.feature_dim}")
            
            # Check for NaN or Inf values in training data
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logging.error("X_train contains NaN or Inf values. Cleaning data...")
                X_train = np.nan_to_num(X_train)
            
            if np.isnan(y_train).any() or np.isinf(y_train).any():
                logging.error("y_train contains NaN or Inf values. Cleaning data...")
                y_train = np.nan_to_num(y_train)
                
            if np.isnan(X_val).any() or np.isinf(X_val).any():
                logging.error("X_val contains NaN or Inf values. Cleaning data...")
                X_val = np.nan_to_num(X_val)
                
            if np.isnan(y_val).any() or np.isinf(y_val).any():
                logging.error("y_val contains NaN or Inf values. Cleaning data...")
                y_val = np.nan_to_num(y_val)
                
            # Create the Keras Tuner
            tuner = kt.Hyperband(
                self.build_model,
                objective='val_loss',
                max_epochs=epochs,
                factor=3,
                directory=tuner_dir,
                project_name='lstm_tuning',
                overwrite=True
            )
            
            # Define early stopping callback
            stop_early = EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            )
            
            # Custom callback to log validation loss
            class LoggingCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs and 'val_loss' in logs:
                        val_loss = logs['val_loss']
                        logging.info(f"Epoch {epoch+1}: val_loss: {val_loss:.4f}")
            
            # Set up callbacks
            callbacks = [stop_early, LoggingCallback()]
                        
            # Allow GPU memory growth to prevent OOM errors
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
            except Exception as e:
                logging.warning(f"Error configuring GPU memory growth: {e}")
            
            # Start the hyperparameter search
            logging.info(f"Starting hyperparameter search with {self.max_trials} trials")
            
            # Run the standard Keras tuner search
            tuner.search(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
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
                callbacks=callbacks,
                verbose=2
            )
            
            # Save the best model
            best_model_path = os.path.join(self.models_dir, f"lstm_tuned_{timestamp}.keras")
            best_model.save(best_model_path)
            logging.info(f"Tuned model saved to {best_model_path}")
            
            # Save hyperparameter info for reference
            with open(os.path.join(self.tuner_dir, f"best_params_{timestamp}.txt"), 'w') as f:
                for param, value in best_hps.values.items():
                    f.write(f"{param}: {value}\n")
            
            # Clean up old tuner files to save disk spacesion
            self._clean_up_tuner_files(timestamp)
                'timestamp': timestamp,
            return best_model, history, best_hps
                'window_size': self.window_size,
        except Exception as e:rizon': self.prediction_horizon,
            logging.error(f"Error during hyperparameter tuning: {e}")
            logging.error(traceback.format_exc())  # Print full traceback for better debugging
            return None, None, Nonea JSON file
            with open(os.path.join(self.models_dir, f"lstm_metadata_{timestamp}.json"), 'w') as f:
    def _clean_up_tuner_files(self, current_timestamp):
        """     json.dump(model_metadata, f, indent=2)
        Delete old tuner directories and files to save disk spaceself.models_dir, f'lstm_metadata_{timestamp}.json')}")
            
        Args: Clean up old tuner files to save disk space
            current_timestamp: Current timestamp to identify files to keep
        """ 
        try:return best_model, history, best_hps
            # Keep only the current tuning session and delete old ones
            deleted_dirs = 0e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            # Get all directories in tuner_dir())  # Print full traceback for better debugging
            for item in os.listdir(self.tuner_dir):
                item_path = os.path.join(self.tuner_dir, item)
                p_tuner_files(self, current_timestamp):
                # Skip current session files
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














            logging.warning(f"Error while cleaning up tuner files: {e}")        except Exception as e:                logging.info(f"Cleaned up {deleted_dirs} old tuner files/directories to save storage space")            if deleted_dirs > 0:                                logging.warning(f"Could not delete old tuner file {item_path}: {e}")                except Exception as e:                    deleted_dirs += 1                        os.remove(item_path)                    else:                        shutil.rmtree(item_path)                        import shutil                    if os.path.isdir(item_path):                try:
    def save_feature_info(self, feature_names, file_suffix=None):
        """
        Save feature names used during training to ensure consistency between training and inference
        
        Args:
            feature_names: List of feature names used in training
            file_suffix: Optional suffix for the filename (default: current timestamp)
        """
        if file_suffix is None:



















































            return None            logging.error(f"Error loading feature information: {e}")        except Exception as e:            return None                                        return json.load(f)                    import json                with open(latest_file, 'r') as f:                latest_file = max(feature_files, key=os.path.getmtime)            if feature_files:            feature_files = glob.glob(os.path.join(self.models_dir, "features_*.json"))            import glob            # If no specific model path or files not found, look for the latest feature info file                                            return json.load(f)                                import json                            with open(metadata_file_path, 'r') as f:                        if os.path.exists(metadata_file_path):                        metadata_file_path = os.path.join(self.models_dir, f"lstm_metadata_{identifier}.json")                        # Try to find metadata file                    else:                            return json.load(f)                            import json                        with open(feature_file_path, 'r') as f:                    if os.path.exists(feature_file_path):                                        feature_file_path = os.path.join(self.models_dir, f"features_{identifier}.json")                    # Look for corresponding feature file                                        identifier = parts[-1].split('.')[0]  # Get the part before file extension                if len(parts) >= 2:                parts = basename.split('_')                basename = os.path.basename(model_path)                # Extract timestamp or identifier from model path            if model_path:        try:        """            Dictionary with feature information or None if not found        Returns:                        model_path: Path to the model file        Args:                Load feature information for a model        """    def load_feature_info(self, model_path=None):                return False            logging.error(f"Error saving feature information: {e}")
        except Exception as e:            return True            logging.info(f"Feature information saved to {feature_file_path}")

                json.dump(feature_info, f, indent=2)


                import json            with open(feature_file_path, 'w') as f:        try:        feature_file_path = os.path.join(self.models_dir, f"features_{file_suffix}.json")            file_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save to JSON file




                }            'feature_names': feature_names            'feature_count': len(feature_names),        feature_info = {            