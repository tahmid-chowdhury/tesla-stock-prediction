import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import sys

# Add the parent directory to the path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_yahoo_data import download_stock_data

def load_data(file_path, download_if_missing=True, refresh_data=False):
    """
    Load raw TSLA stock data
    
    Parameters:
    - file_path: Path to the CSV file
    - download_if_missing: If True, download data if file doesn't exist
    - refresh_data: If True, force download fresh data
    
    Returns:
    - DataFrame with stock data
    """
    # Check if we need to download fresh data
    if refresh_data or (download_if_missing and not os.path.exists(file_path)):
        # Get directory from file path
        directory = os.path.dirname(file_path)
        # Get ticker from filename
        ticker = os.path.basename(file_path).split('.')[0]
        # Download fresh data
        df = download_stock_data(ticker=ticker, output_dir=directory)
    else:
        # Load existing data
        df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def calculate_rsi(series, window=14):
    """Calculate RSI using pandas"""
    delta = series.diff()
    
    # Make two series: one for gains and one for losses
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Calculate the EWMA (Exponentially Weighted Moving Average)
    roll_up = up.ewm(com=window-1, adjust=False).mean()
    roll_down = down.ewm(com=window-1, adjust=False).mean()
    
    # Calculate the RSI based on EWMA
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD using pandas"""
    # Calculate the Fast and Slow Exponential Moving Averages
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    
    # Calculate the MACD line
    macd = fast_ema - slow_ema
    
    # Calculate the signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # Calculate the histogram
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands using pandas"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band

def add_technical_indicators(df):
    """Add technical indicators as features"""
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = calculate_macd(
        df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(
        df['Close'], window=20, num_std=2)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    
    # Add volatility (standard deviation of returns)
    df['Volatility'] = df['Close'].pct_change().rolling(window=5).std()
    
    # Add daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Volume indicators
    df['Volume_1d_chg'] = df['Volume'].pct_change()
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def create_sequences(data, sequence_length):
    """Create sequences for time series model"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # Predicting Close price
    return np.array(X), np.array(y)

def preprocess_data(raw_data_path, sequence_length=10, test_split=0.2, refresh_data=False):
    """Preprocess data pipeline"""
    df = load_data(raw_data_path, download_if_missing=True, refresh_data=refresh_data)
    df = add_technical_indicators(df)
    
    # Determine project root directory based on raw_data_path
    # Assuming raw_data_path is like "../../data/raw/TSLA.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Create absolute path for processed data directory
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed data with absolute path
    processed_path = os.path.join(processed_dir, 'processed_data.csv')
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}")
    
    # Features for model
    feature_columns = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 
                      'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volatility', 'Return']
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split data
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, df

if __name__ == "__main__":
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to raw data file using absolute path
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    data_path = os.path.join(project_root, "data", "raw", "TSLA.csv")
    
    # Add command line argument handling for data refresh
    refresh_data = '--refresh' in sys.argv
    
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path, refresh_data=refresh_data)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
