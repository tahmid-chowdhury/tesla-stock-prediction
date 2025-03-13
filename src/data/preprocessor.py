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
        Add enhanced technical indicators to the dataframe
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Log available columns for debugging
        logging.info(f"Columns before adding indicators: {df.columns.tolist()}")
        
        # Find price and volume columns with flexible naming patterns
        close_col = self._find_column_by_pattern(df, ['close', 'adj close', 'adjusted close'])
        high_col = self._find_column_by_pattern(df, ['high'])
        low_col = self._find_column_by_pattern(df, ['low'])
        open_col = self._find_column_by_pattern(df, ['open'])
        volume_col = self._find_column_by_pattern(df, ['volume'])
        
        if not close_col or not high_col or not low_col or not open_col or not volume_col:
            logging.error(f"Could not find all required price columns. Available: {df.columns.tolist()}")
            raise ValueError("Missing required price columns for technical indicators")
        
        # Log which columns we're using
        logging.info(f"Using columns: Close={close_col}, High={high_col}, Low={low_col}, Open={open_col}, Volume={volume_col}")
        
        # Ensure price and volume data are 1-dimensional
        close_series = df[close_col].squeeze() if hasattr(df[close_col], 'squeeze') else df[close_col]
        high_series = df[high_col].squeeze() if hasattr(df[high_col], 'squeeze') else df[high_col]
        low_series = df[low_col].squeeze() if hasattr(df[low_col], 'squeeze') else df[low_col]
        open_series = df[open_col].squeeze() if hasattr(df[open_col], 'squeeze') else df[open_col]
        volume_series = df[volume_col].squeeze() if hasattr(df[volume_col], 'squeeze') else df[volume_col]
        
        # Basic indicators
        df['MA_5'] = ta.trend.sma_indicator(close_series, window=5)
        df['MA_10'] = ta.trend.sma_indicator(close_series, window=10)
        df['MA_20'] = ta.trend.sma_indicator(close_series, window=20)
        df['MA_50'] = ta.trend.sma_indicator(close_series, window=50)
        df['MA_200'] = ta.trend.sma_indicator(close_series, window=200)
        
        # Exponential moving averages
        df['EMA_5'] = ta.trend.ema_indicator(close_series, window=5)
        df['EMA_10'] = ta.trend.ema_indicator(close_series, window=10)
        df['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
        
        # Moving average crossovers (as binary signals)
        df['MA_5_10_cross'] = (df['MA_5'] > df['MA_10']).astype(float)
        df['MA_10_20_cross'] = (df['MA_10'] > df['MA_20']).astype(float)
        
        # RSI with different periods
        df['RSI'] = ta.momentum.rsi(close_series, window=14)
        df['RSI_7'] = ta.momentum.rsi(close_series, window=7)
        df['RSI_21'] = ta.momentum.rsi(close_series, window=21)
        
        # MACD
        macd = ta.trend.MACD(close_series)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        df['MACD_cross'] = (df['MACD'] > df['MACD_signal']).astype(float)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_series)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['BB_mid'] = bollinger.bollinger_mavg()
        df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
        
        # Price position within Bollinger Bands (normalized 0-1)
        df['BB_position'] = (close_series - df['BB_low']) / (df['BB_high'] - df['BB_low'])
        
        # ATR - Average True Range (volatility indicator)
        df['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.psar_down(high_series, low_series, close_series)
        
        # ADX - Average Directional Index
        df['ADX'] = ta.trend.adx(high_series, low_series, close_series, window=14)
        df['ADX_pos'] = ta.trend.adx_pos(high_series, low_series, close_series, window=14)
        df['ADX_neg'] = ta.trend.adx_neg(high_series, low_series, close_series, window=14)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high_series, low_series, close_series)
        df['STOCH_k'] = stoch.stoch()
        df['STOCH_d'] = stoch.stoch_signal()
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.chaikin_money_flow(high_series, low_series, close_series, volume_series)
        
        # On-balance volume
        df['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
        
        # Volume indicators
        df['VPT'] = ta.volume.volume_price_trend(close_series, volume_series)
        
        # Price transformations
        df['log_return'] = np.log(close_series / close_series.shift(1))
        df['pct_change'] = close_series.pct_change()
        
        # Volatility (standard deviation of log returns)
        df['volatility_7'] = df['log_return'].rolling(window=7).std()
        df['volatility_14'] = df['log_return'].rolling(window=14).std()
        df['volatility_30'] = df['log_return'].rolling(window=30).std()
        
        # Price to moving average ratios
        df['price_to_MA50'] = close_series / df['MA_50']
        df['price_to_MA200'] = close_series / df['MA_200']
        
        # Candlestick patterns - binary features
        df['doji'] = ((abs(open_series - close_series) / (high_series - low_series)) < 0.1).astype(float)
        df['hammer'] = ((high_series - low_series > 3 * (open_series - close_series)) & 
                       (close_series > open_series) & 
                       ((close_series - low_series) / (.001 + high_series - low_series) > 0.6)).astype(float)
        
        # Rename the close column to 'Close' if it's not already called that
        if close_col != 'Close':
            df['Close'] = df[close_col]
            logging.info(f"Created 'Close' column from '{close_col}' for consistency")
        
        # Drop NaN values resulting from indicators
        df = df.dropna()
        
        return df

    def _find_column_by_pattern(self, df, patterns):
        """
        Find column by pattern, case insensitive
        
        Args:
            df: DataFrame to search in
            patterns: List of possible name patterns to look for
        
        Returns:
            Column name if found, None otherwise
        """
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check for exact matches first
            for pattern in patterns:
                if col_lower == pattern:
                    return col
                    
            # Then check for pattern matches (contains)
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        
        # If no match found, log and return None
        logging.warning(f"Could not find column matching patterns {patterns} in {df.columns.tolist()}")
        return None

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
        Performs sentiment analysis on news headlines and content
        """
        news_path = os.path.join(self.raw_dir, "TSLA_news.json")
        
        if not os.path.exists(news_path):
            logging.warning("No news data found. Skipping sentiment analysis.")
            return stock_data
            
        try:
            # Install required packages if not available
            try:
                from textblob import TextBlob
            except ImportError:
                import sys
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
                from textblob import TextBlob
            
            with open(news_path, 'r') as f:
                news_data = json.load(f)
            
            # Log the structure of the loaded news data
            logging.info(f"News data loaded. Structure: {list(news_data.keys()) if isinstance(news_data, dict) else 'Not a dictionary'}")
            
            # Ensure we have articles in the expected format
            if 'articles' not in news_data or not news_data['articles']:
                logging.warning("No articles found in news data or unexpected format")
                return stock_data
                
            # Convert news data to dataframe
            news_df = pd.DataFrame(news_data['articles'])
            
            # Convert publishedAt to datetime
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
            news_df['date'] = news_df['publishedAt'].dt.date
            
            # Calculate sentiment for each article using TextBlob
            def get_sentiment_scores(title, description):
                # Combine title and description, handling None values
                text = ' '.join(filter(None, [str(title) if title else "", str(description) if description else ""]))
                if not text:
                    return 0, 0  # Neutral if no text
                
                # Get sentiment polarity (-1 to 1) and subjectivity (0 to 1)
                sentiment = TextBlob(text)
                return sentiment.polarity, sentiment.subjectivity
            
            # Apply sentiment analysis to each article
            sentiments = news_df.apply(
                lambda x: get_sentiment_scores(x.get('title', ''), x.get('description', '')), 
                axis=1
            )
            news_df['polarity'] = [s[0] for s in sentiments]
            news_df['subjectivity'] = [s[1] for s in sentiments]
            
            # Group by date and calculate various sentiment metrics
            sentiment_metrics = news_df.groupby('date').agg({
                'title': 'count',                      # Number of articles
                'polarity': ['mean', 'std', 'min', 'max'],  # Sentiment polarity stats
                'subjectivity': ['mean', 'max']        # Subjectivity stats
            })
            
            # Flatten the MultiIndex columns
            flat_columns = []
            for col in sentiment_metrics.columns:
                if isinstance(col, tuple) and len(col) > 1:
                    flat_columns.append(f'news_{col[0]}_{col[1]}')
                else:
                    flat_name = col[0] if isinstance(col, tuple) else col
                    flat_columns.append(f'news_{flat_name}')
                    
            sentiment_metrics.columns = flat_columns
            
            # Rename count column if it exists
            if 'news_title_count' in sentiment_metrics.columns:
                sentiment_metrics.rename(columns={'news_title_count': 'news_count'}, inplace=True)
            
            # Add weighted sentiment (more weight to articles with high subjectivity)
            try:
                weighted_sentiment = news_df.groupby('date').apply(
                    lambda x: np.average(x['polarity'], weights=x['subjectivity']+0.1)
                    if len(x) > 0 else 0
                )
                sentiment_metrics['news_weighted_sentiment'] = weighted_sentiment
            except Exception as e:
                logging.error(f"Error calculating weighted sentiment: {e}")
                # Add a default weighted sentiment in case of error
                sentiment_metrics['news_weighted_sentiment'] = sentiment_metrics['news_polarity_mean'] if 'news_polarity_mean' in sentiment_metrics.columns else 0
            
            # Reset index for sentiment metrics
            sentiment_metrics = sentiment_metrics.reset_index()
            sentiment_metrics['date'] = pd.to_datetime(sentiment_metrics['date'])
            
            # Check for MultiIndex in rows or columns of stock_data
            has_multiindex_rows = isinstance(stock_data.index, pd.MultiIndex)
            has_multiindex_cols = isinstance(stock_data.columns, pd.MultiIndex)
            
            logging.info(f"Stock data index type: {type(stock_data.index).__name__}, MultiIndex: {has_multiindex_rows}")
            logging.info(f"Stock data columns type: {type(stock_data.columns).__name__}, MultiIndex: {has_multiindex_cols}")
            
            # Store original index and columns to restore later
            original_index = stock_data.index
            original_columns = stock_data.columns
            
            # Create a working copy with simplified structure for merging
            working_df = stock_data.copy()
            
            # Handle MultiIndex rows if present
            if has_multiindex_rows:
                logging.info(f"Stock data has MultiIndex rows with levels: {stock_data.index.names}")
                # Save the index names
                index_names = stock_data.index.names
                # Reset index to convert to regular columns
                working_df = working_df.reset_index()
            else:
                working_df = working_df.reset_index()
                # Ensure the index column is named properly
                if 'index' in working_df.columns:
                    working_df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Handle MultiIndex columns if present
            if has_multiindex_cols:
                logging.info(f"Stock data has MultiIndex columns: {[col for col in stock_data.columns[:5]]}")
                # Flatten MultiIndex columns for easier merging
                flat_stock_columns = []
                for col in working_df.columns:
                    if isinstance(col, tuple):
                        flat_stock_columns.append('_'.join(str(c) for c in col if c))
                    else:
                        flat_stock_columns.append(str(col))
                working_df.columns = flat_stock_columns
            
            # Ensure Date column exists for merging
            date_col = None
            for col in working_df.columns:
                if isinstance(col, str) and col.lower() == 'date':
                    date_col = col
                    break
            
            if date_col is None:
                logging.error(f"Cannot find date column for merging. Available columns: {working_df.columns.tolist()}")
                return stock_data
            
            # Make sure the date column is datetime type
            working_df[date_col] = pd.to_datetime(working_df[date_col])
            
            # Log column information before merge
            logging.info(f"Working DataFrame columns before merge: {working_df.columns.tolist()}")
            logging.info(f"Sentiment metrics columns before merge: {sentiment_metrics.columns.tolist()}")
            
            # Perform the merge on flattened DataFrames
            merged_data = pd.merge(
                working_df,
                sentiment_metrics,
                left_on=date_col,
                right_on='date',
                how='left'
            )
            
            # Fill missing sentiment values
            for col in merged_data.columns:
                if isinstance(col, str) and col.startswith('news_'):
                    if 'polarity' in col or 'sentiment' in col:
                        merged_data[col] = merged_data[col].fillna(0)  # Neutral sentiment
                    else:
                        merged_data[col] = merged_data[col].fillna(0)  # No news = 0
            
            # Drop the redundant date column
            if 'date' in merged_data.columns:
                merged_data = merged_data.drop('date', axis=1)
            
            # Try to restore original index structure if possible
            if has_multiindex_rows:
                try:
                    # Get the index level columns from the merged DataFrame
                    index_columns = [col for col in merged_data.columns if col in index_names or col in [str(name) for name in index_names]]
                    if index_columns:
                        merged_data = merged_data.set_index(index_columns)
                        logging.info(f"Restored MultiIndex with levels: {merged_data.index.names}")
                    else:
                        # Use date as index if original index columns not found
                        if date_col in merged_data.columns:
                            merged_data = merged_data.set_index(date_col)
                            logging.info(f"Used {date_col} as index because original index columns not found")
                except Exception as e:
                    logging.error(f"Error restoring original index: {e}")
                    # Fall back to using Date as index
                    if date_col in merged_data.columns:
                        merged_data = merged_data.set_index(date_col)
            else:
                # Restore simple index (Date column)
                if date_col in merged_data.columns:
                    merged_data = merged_data.set_index(date_col)
            
            # Count news features for logging
            news_feature_count = sum(1 for col in merged_data.columns if isinstance(col, str) and col.startswith('news_'))
            logging.info(f"Added {news_feature_count} news sentiment features")
            
            return merged_data
            
        except Exception as e:
            logging.error(f"Error processing news data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return stock_data
    
    def prepare_data(self, stock_data):
        """
        Main function to prepare data for model training
        """
        # Add news sentiment if available (optional)
        try:
            # Process news sentiment - this adds multiple sentiment features
            stock_data = self.process_news_sentiment(stock_data)
            logging.info(f"News sentiment features added to dataset")
            
            # Check which sentiment features were added (for debugging)
            sentiment_cols = []
            for col in stock_data.columns:
                if isinstance(col, str) and col.startswith('news_'):
                    sentiment_cols.append(col)
                elif isinstance(col, tuple) and len(col) > 0 and isinstance(col[0], str) and col[0].startswith('news_'):
                    sentiment_cols.append(col)
            
            if sentiment_cols:
                logging.info(f"Added sentiment features: {sentiment_cols}")
            else:
                logging.warning("No sentiment features were added")
                
        except Exception as e:
            logging.error(f"Error in news sentiment processing: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
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
        
        # Add technical indicators with flexible column name handling
        df = self.add_technical_indicators(df)
        logging.info(f"Added technical indicators. Shape: {df.shape}")
        
        # Ensure we have a 'Close' column for prediction
        close_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if col_str == 'close' or 'close' in col_str:
                close_col = col
                break
        
        if close_col is None:
            logging.error(f"No 'Close' column found after processing. Available columns: {df.columns.tolist()}")
            raise ValueError("Missing 'Close' column required for prediction target")
        
        if close_col != 'Close':
            df['Close'] = df[close_col]
            logging.info(f"Using '{close_col}' as the Close price column")
        
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
