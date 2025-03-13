import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, window_size=30, prediction_horizon=5, test_size=0.2):
        """
        Initialize Preprocessor with parameters
        
        Args:
            window_size: Size of sliding window for feature creation (30 days)
            prediction_horizon: Number of days to predict ahead (5 days)
            test_size: Proportion of data to use for testing (0.2 = 20%)
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure close price is 1-dimensional
        close_series = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
        
        # Calculate basic indicators
        df['MA_5'] = ta.trend.sma_indicator(close_series, window=5)
        df['MA_10'] = ta.trend.sma_indicator(close_series, window=10)
        df['MA_20'] = ta.trend.sma_indicator(close_series, window=20)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(close_series, window=14)
        
        # MACD
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_series)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
        
        # Volume indicators - ensure volume is also 1-dimensional
        volume_series = df['Volume'].squeeze() if hasattr(df['Volume'], 'squeeze') else df['Volume']
        df['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
        
        # Price transformation
        df['log_return'] = np.log(close_series / close_series.shift(1))
        df['pct_change'] = close_series.pct_change()
        
        # Volatility
        df['volatility'] = df['log_return'].rolling(window=20).std()
        
        # Drop NaN values resulting from indicators
        df = df.dropna()
        
        return df
    
    def create_sliding_windows(self, df):
        """
        Create sliding windows of data for sequence prediction
        """
        X, y = [], []
        
        # Debug information
        logging.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Try to find the Close column (case-insensitive)
        close_col = None
        for col in df.columns:
            if isinstance(col, str) and col.lower() == 'close':
                close_col = col
                break
        
        if close_col is None:
            # If we can't find 'Close', try using the first column as a fallback
            # or check if there's a numeric index for a column that might be Close
            logging.warning(f"Could not find 'Close' column. Available columns: {df.columns.tolist()}")
            
            # Assume the first column might be Close (common in financial datasets)
            close_col = df.columns[0]
            logging.warning(f"Using '{close_col}' as the target price column")
        else:
            logging.info(f"Found Close column: {close_col}")
            
        # Get only close prices for prediction targets
        prices = df[close_col].values
        
        # Get all features for X
        feature_matrix = df.values
        
        for i in range(len(df) - self.window_size - self.prediction_horizon + 1):
            X.append(feature_matrix[i:i+self.window_size])
            
            # Target is the sequence of closing prices for the next prediction_horizon days
            target_seq = prices[i+self.window_size:i+self.window_size+self.prediction_horizon]
            y.append(target_seq)
        
        return np.array(X), np.array(y)
    
    def process_news_sentiment(self, stock_data):
        """
        Process news sentiment and merge with stock data
        This is a simplified version - a real implementation would use NLP
        """
        news_path = os.path.join(self.raw_dir, "TSLA_news.json")
        
        if not os.path.exists(news_path):
            logging.warning("No news data found. Skipping sentiment analysis.")
            return stock_data
            
        try:
            with open(news_path, 'r') as f:
                news_data = json.load(f)
                
            # Convert news data to dataframe
            news_df = pd.DataFrame(news_data['articles'])
            
            # Convert publishedAt to datetime
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
            news_df['date'] = news_df['publishedAt'].dt.date
            
            # Group by date and count articles (very basic sentiment indicator)
            daily_news_count = news_df.groupby('date').size().reset_index(name='news_count')
            daily_news_count['date'] = pd.to_datetime(daily_news_count['date'])
            
            # Merge with stock data
            stock_data = stock_data.reset_index()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            merged_data = pd.merge(stock_data, daily_news_count, 
                                  left_on='Date', right_on='date', 
                                  how='left')
                                  
            merged_data = merged_data.drop('date', axis=1)
            merged_data['news_count'] = merged_data['news_count'].fillna(0)
            merged_data = merged_data.set_index('Date')
            
            return merged_data
        except Exception as e:
            logging.error(f"Error processing news data: {e}")
            return stock_data
    
    # Improve handling of MultiIndex columns
    def prepare_data(self, stock_data):
        """
        Main function to prepare data for model training
        """
        # Add news sentiment if available (optional)
        try:
            stock_data = self.process_news_sentiment(stock_data)
        except Exception as e:
            logging.error(f"Error in news sentiment processing: {e}")
        
        # Handle MultiIndex columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Convert MultiIndex to single level using the first level
            close_cols = [col for col in stock_data.columns if 'Close' in col[0]]
            if close_cols:
                # Create a standardized DataFrame with single-level column names
                renamed_data = {}
                for name in ['Close', 'Open', 'High', 'Low', 'Volume']:
                    col_matches = [col for col in stock_data.columns if name in col[0]]
                    if col_matches:
                        renamed_data[name] = stock_data[col_matches[0]]
                
                # Create new DataFrame with standard column names
                df = pd.DataFrame(renamed_data, index=stock_data.index)
                logging.info(f"Converted MultiIndex columns to standard format. New shape: {df.shape}")
            else:
                logging.error("Could not find required columns in MultiIndex DataFrame")
                df = stock_data.copy()
        else:
            df = stock_data.copy()
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        logging.info(f"Added technical indicators. Shape: {df.shape}")
        
        # Ensure consistent column naming for 'Close'
        if 'Close' not in df.columns:
            close_cols = [col for col in df.columns if isinstance(col, str) and col.lower() == 'close']
            if close_cols:
                # Rename the column to 'Close'
                df = df.rename(columns={close_cols[0]: 'Close'})
                logging.info(f"Renamed column '{close_cols[0]}' to 'Close'")
            else:
                logging.error(f"No 'Close' column found. Available columns: {df.columns.tolist()}")
        
        # Scale the closing prices for prediction
        close_prices = df['Close'].values.reshape(-1, 1)
        self.price_scaler.fit(close_prices)
        
        # Scale the features
        feature_columns = df.columns.tolist()
        self.feature_scaler.fit(df)
        df_scaled = pd.DataFrame(self.feature_scaler.transform(df), columns=feature_columns)
        
        # Create sequences with sliding window
        X, y = self.create_sliding_windows(df_scaled)
        logging.info(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        
        # Save processed data
        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)
        
        # Save scalers info
        np.save(os.path.join(self.processed_dir, "feature_columns.npy"), feature_columns)
        
        logging.info("Data processing complete and saved.")
        
        return X_train, X_test, y_train, y_test
