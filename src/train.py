import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from model import LSTMModel
from utils.data_preprocessing import load_processed_data

def train_lstm_model():
    # Load processed data
    X, y = load_processed_data()

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the LSTM model
    model = LSTMModel()

    # Define a checkpoint to save the best model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint])

if __name__ == "__main__":
    train_lstm_model()