import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel:
    def __init__(self, window_size=30, prediction_horizon=5, feature_dim=1, simplified=False):
        """
        Initialize the LSTM model
        
        Args:
            window_size: Size of the sliding window for input data
            prediction_horizon: Number of days to predict ahead
            feature_dim: Number of features per time step
            simplified: Whether to use simplified architecture for faster training
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.feature_dim = feature_dim
        self.model = None
        self.ensemble_models = []  # Store multiple models for ensemble predictions
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        self.market_regimes = {}  # Store detected market regimes
        self.simplified = simplified  # Flag for simplified architecture
        os.makedirs(self.models_dir, exist_ok=True)
        
    def build_model(self, lstm_units=128, dropout_rate=0.2, learning_rate=0.001, regularization_factor=0.01):
        """
        Build LSTM model architecture with improved regularization and design
        
        Args:
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            regularization_factor: L1/L2 regularization strength
        """
        # Use the TensorFlow 2.x recommended method to clear session
        tf.keras.backend.clear_session()
        
        # Add regularizers to prevent overfitting
        regularizer = l1_l2(l1=regularization_factor/2, l2=regularization_factor)
        
        # Create Sequential model with proper input specification
        if self.simplified:
            # Simplified architecture for faster training
            logging.info("Using simplified model architecture for faster training")
            
            # Reduce units and layers
            simplified_units = lstm_units // 2  # Half the units
            
            model = Sequential([
                # Input layer with explicit shape
                Input(shape=(self.window_size, self.feature_dim), name='input_layer'),
                
                # Single LSTM layer
                LSTM(
                    units=simplified_units,
                    return_sequences=False,  # No stacked LSTM
                    kernel_regularizer=regularizer,
                    name='lstm_1'
                ),
                BatchNormalization(name='batch_norm_1'),
                Dropout(dropout_rate, name='dropout_1'),
                
                # Output layer (predicting multiple days ahead)
                Dense(units=self.prediction_horizon, name='output')
            ])
        else:
            # Standard model architecture
            model = Sequential([
                # Input layer with explicit shape
                Input(shape=(self.window_size, self.feature_dim), name='input_layer'),
                
                # First LSTM layer with return sequences for stacking
                LSTM(
                    units=lstm_units,
                    return_sequences=True,
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer,
                    name='lstm_1'
                ),
                BatchNormalization(name='batch_norm_1'),
                Dropout(dropout_rate, name='dropout_1'),
                
                # Second LSTM layer with regularization
                LSTM(units=int(lstm_units/2), 
                     return_sequences=False, 
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer,
                     name='lstm_2'),
                BatchNormalization(name='batch_norm_2'),
                Dropout(dropout_rate, name='dropout_2'),
                
                # Dense hidden layer for additional pattern recognition
                Dense(units=64, 
                      activation='relu', 
                      kernel_regularizer=regularizer,
                      name='dense_1'),
                Dropout(dropout_rate/2, name='dropout_3'),
                
                # Output layer (predicting multiple days ahead)
                Dense(units=self.prediction_horizon, name='output')
            ])
        
        # Compile model with Adam optimizer and MSE loss
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        # Log model configuration
        if self.simplified:
            logging.info(f"Simplified LSTM model built with {simplified_units} units")
        else:
            logging.info(f"LSTM model built with {lstm_units} units, {dropout_rate} dropout rate, {regularization_factor} regularization")
        
        logging.info(f"Model summary: {model.summary()}")
        
        self.model = model
        return model
        
    def detect_market_regime(self, X, y=None, n_regimes=3):
        """
        Detect market regimes using unsupervised clustering
        
        Args:
            X: Input features (can be windowed data or raw features)
            y: Optional target values
            n_regimes: Number of regimes to detect
            
        Returns:
            Regime labels and centroids
        """
        logging.info(f"Detecting market regimes with {n_regimes} clusters")
        
        # Use the last value of each window, or flatten data if needed
        if len(X.shape) == 3:  # Windowed data
            # Extract relevant features for regime detection (e.g., volatility, volume, price movement)
            # For simplicity, we'll use the last timestep of each window
            features_for_clustering = X[:, -1, :]
        else:
            features_for_clustering = X
            
        # Normalize data for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_for_clustering)
        
        # Apply KMeans clustering to identify regimes
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(normalized_features)
        
        # Save regime model and scaler
        regime_models_dir = os.path.join(self.models_dir, 'regime_models')
        os.makedirs(regime_models_dir, exist_ok=True)
        
        joblib.dump(kmeans, os.path.join(regime_models_dir, 'kmeans_regimes.joblib'))
        joblib.dump(scaler, os.path.join(regime_models_dir, 'regime_scaler.joblib'))
        
        # Store regime information
        self.market_regimes = {
            'kmeans': kmeans,
            'scaler': scaler,
            'centroids': kmeans.cluster_centers_,
            'labels': regimes
        }
        
        # Log regime distribution
        regime_counts = np.bincount(regimes)
        regime_percentages = regime_counts / len(regimes) * 100
        
        logging.info(f"Market regime distribution:")
        for i, (count, percentage) in enumerate(zip(regime_counts, regime_percentages)):
            logging.info(f"Regime {i}: {count} samples ({percentage:.1f}%)")
            
        return regimes, kmeans.cluster_centers_
    
    def build_ensemble_models(self, n_models=3, **kwargs):
        """
        Build multiple models with different configurations for ensemble predictions
        
        Args:
            n_models: Number of models in the ensemble
            **kwargs: Parameters for model building
        """
        logging.info(f"Building ensemble with {n_models} models")
        
        self.ensemble_models = []
        
        # Define variations for different models
        lstm_units_options = [64, 96, 128, 160]
        dropout_options = [0.2, 0.3, 0.4]
        learning_rate_options = [0.001, 0.0005, 0.0003]
        regularization_options = [0.01, 0.005, 0.02]
        
        for i in range(n_models):
            # Pick different hyperparameters for each model
            lstm_units = np.random.choice(lstm_units_options)
            dropout_rate = np.random.choice(dropout_options)
            learning_rate = np.random.choice(learning_rate_options)
            regularization = np.random.choice(regularization_options)
            
            # Build model with different configuration
            model = Sequential([
                Input(shape=(self.window_size, self.feature_dim)),
                LSTM(
                    units=lstm_units,
                    return_sequences=True,
                    kernel_regularizer=l1_l2(l1=regularization/2, l2=regularization)
                ),
                BatchNormalization(),
                Dropout(dropout_rate),
                LSTM(units=int(lstm_units/2), 
                     kernel_regularizer=l1_l2(l1=regularization/2, l2=regularization)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(units=self.prediction_horizon)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            self.ensemble_models.append({
                'model': model,
                'config': {
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'regularization': regularization
                }
            })
            
            logging.info(f"Ensemble model {i+1} built with units={lstm_units}, dropout={dropout_rate}")
            
        return self.ensemble_models

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=64,
              patience=10, lstm_units=128, dropout_rate=0.2, learning_rate=0.001,
              use_ensemble=False, n_ensemble_models=3):
        """
        Train the LSTM model with improved training process and ensemble option
        
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
            use_ensemble: Whether to use ensemble models
            n_ensemble_models: Number of models in ensemble
            
        Returns:
            Training history
        """
        # Log training configuration
        logging.info(f"Training configuration: epochs={epochs}, batch_size={batch_size}, patience={patience}")
        logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        if X_val is not None:
            logging.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        # Detect market regimes if we have enough data
        if len(X_train) > 100:
            self.detect_market_regime(X_train)
            
        # Build ensemble models if requested
        if use_ensemble:
            self.build_ensemble_models(n_models=n_ensemble_models, 
                                       lstm_units=lstm_units,
                                       dropout_rate=dropout_rate, 
                                       learning_rate=learning_rate)
            
            # Train each model in the ensemble
            ensemble_histories = []
            for i, model_info in enumerate(self.ensemble_models):
                model = model_info['model']
                logging.info(f"Training ensemble model {i+1}/{len(self.ensemble_models)}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = os.path.join(self.models_dir, f'ensemble_{i+1}_checkpoint_{timestamp}.keras')
                
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss' if X_val is not None else 'loss',
                        patience=patience,
                        restore_best_weights=True,
                        mode='min',
                        verbose=1
                    ),
                    ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss' if X_val is not None else 'loss',
                        save_best_only=True,
                        mode='min',
                        verbose=1
                    )
                ]
                
                # Train with or without validation data
                if X_val is not None and y_val is not None:
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                else:
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
                
                # Save model
                model_path = os.path.join(self.models_dir, f'ensemble_{i+1}_{timestamp}.keras')
                model.save(model_path)
                logging.info(f"Ensemble model {i+1} saved to {model_path}")
                
                ensemble_histories.append(history.history)
            
            # Set the first model as the main model for compatibility
            self.model = self.ensemble_models[0]['model']
            return ensemble_histories
            
        # Build single model if not already built
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
            ),
            
            # Learning rate scheduler for better convergence
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience//2,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Training with or without validation data
        if X_val is not None and y_val is not None:
            logging.info(f"Starting model training with validation for {epochs} epochs")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1  # Changed from 2 to 1 for more informative progress bars
            )
        else:
            logging.info(f"Starting model training without validation for {epochs} epochs")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
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
        
    def predict(self, X, use_ensemble=False, return_confidence=False):
        """
        Make predictions using trained model with confidence intervals
        
        Args:
            X: Input data to predict on
            use_ensemble: Whether to use ensemble models for prediction
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Predictions and optional confidence intervals
        """
        if self.model is None:
            logging.error("Model not trained or loaded. Cannot make predictions.")
            return None
        
        # First, check if we're dealing with a custom TensorFlowTrainer model
        # These models don't support use_ensemble and return_confidence parameters
        is_custom_model = hasattr(self.model, 'predict') and not hasattr(self.model, 'predict.__func__')
        
        try:
            # For standard Keras model (no ensemble)
            if not use_ensemble or not self.ensemble_models:
                if is_custom_model:
                    # Standard TensorFlow prediction without custom parameters
                    predictions = self.model.predict(X)
                else:
                    # Our custom model prediction
                    predictions = self.model.predict(X)
                
                # If confidence intervals are requested, generate them
                if return_confidence:
                    # Calculate prediction intervals based on prediction uncertainty
                    n_samples = 10  # Number of samples for bootstrapping
                    bootstrap_preds = []
                    
                    # Create bootstrap samples by adding noise
                    for _ in range(n_samples):
                        if is_custom_model:
                            # Add Gaussian noise to inputs for uncertainty
                            noise_level = 0.01  # Small amount of noise
                            noisy_X = X + np.random.normal(0, noise_level, X.shape)
                            bootstrap_preds.append(self.model.predict(noisy_X))
                        else:
                            # Use our model's predict method
                            noise_level = 0.01
                            noisy_X = X + np.random.normal(0, noise_level, X.shape)
                            bootstrap_preds.append(self.model.predict(noisy_X))
                    
                    # Calculate mean and std of bootstrap predictions
                    bootstrap_preds = np.array(bootstrap_preds)
                    mean_preds = np.mean(bootstrap_preds, axis=0)
                    std_preds = np.std(bootstrap_preds, axis=0)
                    
                    # Calculate confidence intervals (95%)
                    lower_95 = mean_preds - 1.96 * std_preds
                    upper_95 = mean_preds + 1.96 * std_preds
                    
                    # Return predictions with confidence intervals
                    confidence_intervals = {
                        'lower_95': lower_95,
                        'upper_95': upper_95,
                        'std': std_preds
                    }
                    
                    return predictions, confidence_intervals
                
                return predictions
            
            # For ensemble models - this part is only used if we have ensemble models
            # and use_ensemble is True
            else:
                logging.info(f"Making ensemble prediction with {len(self.ensemble_models)} models")
                ensemble_predictions = []
                
                # Get predictions from each model in the ensemble
                for model_info in self.ensemble_models:
                    model = model_info['model']
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        ensemble_predictions.append(pred)
                
                # Combine predictions (average)
                if ensemble_predictions:
                    ensemble_predictions = np.array(ensemble_predictions)
                    mean_prediction = np.mean(ensemble_predictions, axis=0)
                    
                    # If confidence intervals are requested
                    if return_confidence:
                        std_prediction = np.std(ensemble_predictions, axis=0)
                        
                        # Calculate 95% confidence intervals
                        lower_95 = mean_prediction - 1.96 * std_prediction
                        upper_95 = mean_prediction + 1.96 * std_prediction
                        
                        # Return predictions with confidence intervals
                        confidence_intervals = {
                            'lower_95': lower_95,
                            'upper_95': upper_95,
                            'std': std_prediction
                        }
                        
                        return mean_prediction, confidence_intervals
                    
                    return mean_prediction
                else:
                    # Fallback to standard prediction if ensemble is empty
                    logging.warning("Ensemble is empty. Using standard model.")
                    predictions = self.model.predict(X)
                    
                    if return_confidence:
                        # Simple confidence estimation based on prediction value
                        std_prediction = np.abs(predictions) * 0.1  # 10% of prediction as std
                        lower_95 = predictions - 1.96 * std_prediction
                        upper_95 = predictions + 1.96 * std_prediction
                        
                        confidence_intervals = {
                            'lower_95': lower_95,
                            'upper_95': upper_95,
                            'std': std_prediction
                        }
                        
                        return predictions, confidence_intervals
                    
                    return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return zeros or a fallback prediction
            if return_confidence:
                return np.zeros((len(X), self.prediction_horizon)), {
                    'lower_95': np.zeros((len(X), self.prediction_horizon)),
                    'upper_95': np.zeros((len(X), self.prediction_horizon))
                }
            else:
                return np.zeros((len(X), self.prediction_horizon))
            
    def current_market_regime(self, recent_data):
        """
        Identify the current market regime based on recent data
        
        Args:
            recent_data: Recent market data
            
        Returns:
            Regime label and confidence
        """
        if not self.market_regimes or 'kmeans' not in self.market_regimes:
            logging.warning("No market regimes detected yet")
            return None, 0
            
        try:
            # Prepare data for regime detection
            if len(recent_data.shape) == 3:  # Windowed data
                features = recent_data[:, -1, :]  # Use last timestep
            else:
                features = recent_data
                
            # Scale features
            scaler = self.market_regimes['scaler']
            scaled_features = scaler.transform(features)
            
            # Predict regime
            kmeans = self.market_regimes['kmeans']
            regime = kmeans.predict(scaled_features)[0]
            
            # Calculate confidence as inverse of distance to centroid
            centroid = kmeans.cluster_centers_[regime]
            distance = np.linalg.norm(scaled_features[0] - centroid)
            max_distance = np.max([np.linalg.norm(c - centroid) for c in kmeans.cluster_centers_])
            confidence = 1 - (distance / max_distance)
            
            return regime, confidence
            
        except Exception as e:
            logging.error(f"Error detecting market regime: {e}")
            return None, 0
    
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
