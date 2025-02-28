import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

class StockPredictionModel:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build_model(self):
        """Build and compile LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, save_path=None):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
        ]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=save_path, save_best_only=True))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call build_model() and train() first.")
        
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save. Call build_model() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from file"""
        self.model = load_model(filepath)

def train_model(X_train, y_train, X_test, y_test, sequence_length, n_features, save_dir="../../models"):
    """Train LSTM model and return the trained model"""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'lstm_model.h5')
    
    model = StockPredictionModel(sequence_length, n_features)
    model.build_model()
    
    # Use a portion of test data as validation
    val_size = int(len(X_test) * 0.5)
    X_val, y_val = X_test[:val_size], y_test[:val_size]
    X_test, y_test = X_test[val_size:], y_test[val_size:]
    
    history = model.train(X_train, y_train, X_val, y_val, save_path=model_path)
    model.save(model_path)
    
    # Evaluate model
    test_loss = model.model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")
    
    return model
