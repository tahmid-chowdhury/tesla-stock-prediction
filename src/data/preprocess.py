import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import talib

def load_data(file_path):
    """Load raw TSLA stock data"""
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

def add_technical_indicators(df):
    """Add technical indicators as features"""
    # Calculate moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = talib.MACD(
        df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
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

def preprocess_data(raw_data_path, sequence_length=10, test_split=0.2):
    """Preprocess data pipeline"""
    df = load_data(raw_data_path)
    df = add_technical_indicators(df)
    
    # Save processed data
    processed_dir = os.path.join(os.path.dirname(raw_data_path), '../processed')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, 'processed_data.csv'), index=False)
    
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
    data_path = "../../data/raw/TSLA.csv"
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data(data_path)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
