import pandas as pd
import numpy as np
from keras.models import load_model
from utils.data_preprocessing import preprocess_data
from predict import load_trained_model, make_prediction

TRANSACTION_FEE_RATE = 0.001  # e.g. 0.1% fee per trade

def simulate_trades(close_prices, predictions, initial_balance=10000):
    balance = initial_balance
    shares = 0
    # Loop over each day (except last one) to decide trades.
    for i in range(len(close_prices)-1):
        current_price = close_prices[i]
        predicted_price = predictions[i]
        # If not holding and prediction is higher, buy shares.
        if shares == 0 and predicted_price > current_price:
            shares = balance / current_price
            fee = shares * current_price * TRANSACTION_FEE_RATE
            balance -= fee
        # If holding and prediction is lower, sell shares.
        elif shares > 0 and predicted_price < current_price:
            fee = shares * current_price * TRANSACTION_FEE_RATE
            balance += shares * current_price - fee
            shares = 0
    # If still holding at the end, sell at final day's price.
    if shares > 0:
        fee = shares * close_prices[-1] * TRANSACTION_FEE_RATE
        balance += shares * close_prices[-1] - fee
        shares = 0
    return balance

def main():
    # Load the trained model (update model_path if needed)
    model_path = 'best_model.h5'
    model = load_trained_model(model_path)
    
    # Load new market data (requires a 'close' column)
    new_data = pd.read_csv('data/raw/latest_data.csv')
    processed_data = preprocess_data(new_data)
    
    # Make predictions using the model.
    predictions_array = make_prediction(model, processed_data)
    # Assume predictions_array is aligned with close prices.
    # For simulation, convert predictions to a 1D list.
    predictions = [float(pred[0]) for pred in predictions_array]
    close_prices = new_data['close'].tolist()
    
    final_balance = simulate_trades(close_prices, predictions)
    
    print(f"Final Account Balance: {final_balance}")

if __name__ == "__main__":
    main()
