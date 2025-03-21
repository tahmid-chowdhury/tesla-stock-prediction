import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessor:
    def __init__(self, window_size=15, prediction_horizon=5, test_size=0.2, detect_regimes=False, 
                 num_regimes=3, reduced_features=False, feature_count=20):
        """
        Initialize Preprocessor with parameters
        
        Args:
            window_size: Size of sliding window for feature creation (default 15 days)
            prediction_horizon: Number of days to predict ahead (5 days)
            test_size: Proportion of data to use for testing (0.2 = 20%)
            detect_regimes: Whether to detect market regimes
            num_regimes: Number of market regimes to detect
            reduced_features: Whether to use only essential features for faster training
            feature_count: Number of features to use when reduced_features is True
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.sentiment_scaler = RobustScaler()  # Better for sentiment data with outliers
        self.detect_regimes = detect_regimes
        self.num_regimes = num_regimes
        self.regime_model = None
        self.regime_scaler = None
        self.reduced_features = reduced_features
        self.feature_count = feature_count
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_model_metadata(self):
        """
        Load metadata from saved models to ensure feature compatibility
        
        Returns:
            Dictionary with model metadata including expected feature dimensions, or None if not found
        """
        try:
            # Try to find the best model file first
            best_model_path = os.path.join(self.models_dir, "lstm_best.keras")
            if os.path.exists(best_model_path):
                # Look for a corresponding metadata file
                metadata_files = [f for f in os.listdir(self.models_dir) if f.startswith("lstm_metadata_") and f.endswith(".json")]
                if metadata_files:
                    # Use the most recent metadata file
                    latest_metadata = max(metadata_files, key=lambda f: os.path.getmtime(os.path.join(self.models_dir, f)))
                    metadata_path = os.path.join(self.models_dir, latest_metadata)
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        logging.info(f"Loaded model metadata from {metadata_path}")
                        return metadata
                
                # If we can't find metadata files, try to get feature info from other sources
                feature_info_files = [f for f in os.listdir(self.models_dir) if f.startswith("features_") and f.endswith(".json")]
                if feature_info_files:
                    latest_feature_info = max(feature_info_files, key=lambda f: os.path.getmtime(os.path.join(self.models_dir, f)))
                    feature_path = os.path.join(self.models_dir, latest_feature_info)
                    
                    with open(feature_path, 'r') as f:
                        feature_info = json.load(f)
                        logging.info(f"Loaded feature information from {feature_path}")
                        return feature_info
            
            # If we still don't have metadata, try to load from saved model configuration
            try:
                import tensorflow as tf
                if os.path.exists(best_model_path):
                    model = tf.keras.models.load_model(best_model_path)
                    feature_dim = model.input_shape[-1]
                    metadata = {"feature_dim": feature_dim}
                    logging.info(f"Extracted feature dimension {feature_dim} directly from model")
                    return metadata
            except Exception as e:
                logging.warning(f"Could not load model to extract input shape: {e}")
                
            return None
        except Exception as e:
            logging.warning(f"Error loading model metadata: {e}")
            return None
    
    def adjust_features_for_compatibility(self, df, feature_dim):
        """
        Adjust features to ensure compatibility with model's expected input dimension
        
        Args:
            df: DataFrame with features
            feature_dim: Expected feature dimension for the model
        
        Returns:
            Adjusted DataFrame with compatible number of features
        """
        current_feature_count = len(df.columns)
        
        if current_feature_count == feature_dim:
            return df  # No adjustment needed
            
        if current_feature_count > feature_dim:
            logging.warning(f"Data has {current_feature_count} features but model expects {feature_dim}")
            
            # Get priority features to decide what to keep
            priority_patterns = ['close', 'open', 'high', 'low', 'volume', 'sentiment']
            priority_features = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in priority_patterns):
                    priority_features.append(col)
            
            # If we have enough priority features, use them
            if len(priority_features) >= feature_dim:
                logging.info(f"Using top {feature_dim} priority features")
                return df[priority_features[:feature_dim]]
            
            # Otherwise, keep all priority features and add other columns until we reach feature_dim
            other_features = [col for col in df.columns if col not in priority_features]
            features_to_keep = priority_features + other_features[:feature_dim - len(priority_features)]
            
            logging.info(f"Keeping {len(priority_features)} priority features and {len(features_to_keep) - len(priority_features)} other features")
            return df[features_to_keep]
            
        else:
            # We have fewer features than expected - this is unusual but could happen
            logging.error(f"Data has fewer features ({current_feature_count}) than model expects ({feature_dim})")
            # One option would be to add dummy features, but that's likely to cause issues
            # Instead, return as is and let calling code handle this case
            return df
        
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
        
        # Import our patched PSAR function to avoid the FutureWarning
        try:
            from src.utils.ta_utils import patched_psar
            # Use patched PSAR implementation
            df['PSAR'] = patched_psar(high_series, low_series, close_series)
        except ImportError:
            # Fall back to ta library if our patch is not available
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
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
            def get_sentiment_scores(title, description, content=None):
                # Combine title, description, and content, handling None values
                text_parts = [
                    str(title) if title else "",
                    str(description) if description else "",
                    str(content) if content else ""
                ]
                text = ' '.join(filter(None, text_parts))
                if not text:
                    return 0, 0  # Neutral if no text
                
                # Get sentiment polarity (-1 to 1) and subjectivity (0 to 1)
                sentiment = TextBlob(text)
                return sentiment.polarity, sentiment.subjectivity
            
            # Apply sentiment analysis to each article
            sentiments = news_df.apply(
                lambda x: get_sentiment_scores(
                    x.get('title', ''), 
                    x.get('description', ''),
                    x.get('content', '')  # Also include content if available
                ), 
                axis=1
            )
            news_df['polarity'] = [s[0] for s in sentiments]
            news_df['subjectivity'] = [s[1] for s in sentiments]
            
            # Add sentiment magnitude (absolute value of polarity)
            news_df['sentiment_strength'] = news_df['polarity'].abs()
            
            # Add sentiment category
            def categorize_sentiment(pol):
                if pol > 0.3:
                    return 'very_positive'
                elif pol > 0.1:
                    return 'positive'
                elif pol < -0.3:
                    return 'very_negative'
                elif pol < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
                    
            news_df['sentiment_category'] = news_df['polarity'].apply(categorize_sentiment)
            
            # Group by date and calculate various sentiment metrics
            sentiment_metrics = news_df.groupby('date').agg({
                'title': 'count',                      # Number of articles
                'polarity': ['mean', 'std', 'min', 'max'],  # Sentiment polarity stats
                'subjectivity': ['mean', 'max'],       # Subjectivity stats
                'sentiment_strength': ['mean', 'max']  # Strength of sentiment
            })
            
            # Count sentiment categories by date
            sentiment_categories = pd.crosstab(news_df['date'], news_df['sentiment_category'])
            sentiment_categories = sentiment_categories.reset_index()
            
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
            
            # Add high-conviction sentiment - polarized news with high subjectivity
            try:
                high_conviction_sentiment = news_df.groupby('date').apply(
                    lambda x: np.average(
                        x['polarity'], 
                        weights=(x['subjectivity'] * x['sentiment_strength'] + 0.1)
                    ) if len(x) > 0 else 0
                )
                sentiment_metrics['news_high_conviction_sentiment'] = high_conviction_sentiment
            except Exception as e:
                logging.error(f"Error calculating high conviction sentiment: {e}")
                sentiment_metrics['news_high_conviction_sentiment'] = 0
            
            # Add sentiment volatility - how much sentiment varies within a day
            sentiment_metrics['news_sentiment_volatility'] = news_df.groupby('date')['polarity'].apply(
                lambda x: x.max() - x.min() if len(x) > 1 else 0
            )
            
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
            logging.info(f"Working DataFrame date column type: {working_df[date_col].dtype}")
            logging.info(f"Sentiment metrics date column type: {sentiment_metrics['date'].dtype}")
            
            # Convert both date columns to the same format to ensure compatibility
            working_df[date_col] = working_df[date_col].dt.date
            sentiment_metrics['date'] = sentiment_metrics['date'].dt.date
            
            # Convert back to datetime for proper merging
            working_df[date_col] = pd.to_datetime(working_df[date_col])
            sentiment_metrics['date'] = pd.to_datetime(sentiment_metrics['date'])
            
            # Perform the merge on flattened DataFrames
            try:
                merged_data = pd.merge(
                    working_df,
                    sentiment_metrics,
                    left_on=date_col,
                    right_on='date',
                    how='left'
                )
            except ValueError as e:
                logging.warning(f"Merge error: {e}")
                logging.warning("Attempting alternative merge method using concat...")
                
                # Alternative approach using concat
                sentiment_metrics = sentiment_metrics.set_index('date')
                working_df_with_date_index = working_df.set_index(date_col)
                
                # Concat and then reset index
                merged_data = pd.concat([working_df_with_date_index, sentiment_metrics], axis=1, join='left')
                merged_data = merged_data.reset_index()
                merged_data = merged_data.rename(columns={'index': date_col})
            
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
            
            # List the specific sentiment features added
            sentiment_features = [col for col in merged_data.columns if isinstance(col, str) and col.startswith('news_')]
            logging.info(f"Sentiment features added: {sentiment_features}")
            
            return merged_data
            
        except Exception as e:
            logging.error(f"Error processing news data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return stock_data
    
    def identify_market_regimes(self, price_data, volume_data=None, volatility_window=20):
        """
        Identify market regimes based on price, volume, and volatility patterns
        
        Args:
            price_data: Historical price data
            volume_data: Optional volume data
            volatility_window: Window size for volatility calculation
            
        Returns:
            Array of regime labels
        """
        if not self.detect_regimes:
            return None
            
        # Calculate features for regime detection
        features = []
        
        # 1. Price momentum: rate of change over different periods
        for period in [5, 10, 20]:
            momentum = np.zeros_like(price_data)
            momentum[period:] = (price_data[period:] - price_data[:-period]) / price_data[:-period]
            features.append(momentum)
            
        # 2. Volatility: standard deviation of returns
        returns = np.zeros_like(price_data)
        returns[1:] = (price_data[1:] - price_data[:-1]) / price_data[:-1]
        
        volatility = np.zeros_like(returns)
        for i in range(volatility_window, len(returns)):
            volatility[i] = np.std(returns[i-volatility_window:i])
        features.append(volatility)
        
        # 3. Volume features if available
        if volume_data is not None:
            # Volume momentum
            vol_momentum = np.zeros_like(volume_data)
            vol_momentum[5:] = (volume_data[5:] - volume_data[:-5]) / (volume_data[:-5] + 1)  # Add 1 to avoid division by zero
            features.append(vol_momentum)
            
            # Volume relative to moving average
            vol_ma = np.zeros_like(volume_data)
            for i in range(10, len(volume_data)):
                vol_ma[i] = volume_data[i] / np.mean(volume_data[i-10:i])
            features.append(vol_ma)
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(normalized_features)
        
        # Save the regime model and scaler
        self.regime_model = kmeans
        self.regime_scaler = scaler
        
        processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        joblib.dump(kmeans, os.path.join(processed_dir, "regime_model.joblib"))
        joblib.dump(scaler, os.path.join(processed_dir, "regime_scaler.joblib"))
        
        # Log regime distribution
        regime_counts = np.bincount(regimes)
        regime_percentages = regime_counts / len(regimes) * 100
        
        logging.info(f"Market regime distribution:")
        for i, (count, percentage) in enumerate(zip(regime_counts, regime_percentages)):
            logging.info(f"Regime {i}: {count} days ({percentage:.1f}%)")
            
        # Save regime labels
        np.save(os.path.join(processed_dir, "regime_labels.npy"), regimes)
        
        # Visualize regimes
        self.visualize_regimes(price_data, regimes)
        
        return regimes
    
    def visualize_regimes(self, price_data, regimes):
        """Create a visualization of price data colored by regime"""
        plt.figure(figsize=(12, 6))
        
        # Create a time index
        x = np.arange(len(price_data))
        
        # Get unique regimes
        unique_regimes = np.unique(regimes)
        
        # Plot each regime with a different color
        for regime in unique_regimes:
            mask = regimes == regime
            plt.plot(x[mask], price_data[mask], '.', label=f'Regime {regime}')
        
        plt.title('Market Regimes')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the visualization
        plt.tight_layout()
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f'market_regimes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
    
    def select_important_features(self, df, n_features=20):
        """
        Select the most important features to reduce dimensionality and speed up training
        
        Args:
            df: DataFrame with features
            n_features: Number of features to select
            
        Returns:
            DataFrame with only the selected important features
        """
        # Always keep the core price columns
        essential_columns = []
        for col in df.columns:
            col_str = str(col).lower()
            if ('close' in col_str or 'open' in col_str or 'high' in col_str or 
                'low' in col_str or 'volume' in col_str):
                essential_columns.append(col)
        
        # Add important technical indicators
        important_patterns = [
            'ma_20', 'ma_50', 'ema_10', 'rsi', 'macd', 'bb_', 'atr',
            'volatility', 'log_return', 'pct_change'
        ]
        
        for pattern in important_patterns:
            for col in df.columns:
                col_str = str(col).lower()
                if pattern in col_str and col not in essential_columns:
                    essential_columns.append(col)
                    if len(essential_columns) >= n_features:
                        break
            if len(essential_columns) >= n_features:
                break
        
        # Add sentiment features if available
        sentiment_cols = [col for col in df.columns if str(col).startswith('news_')]
        for col in sentiment_cols:
            if 'weighted' in str(col) or 'polarity' in str(col):
                essential_columns.append(col)
                if len(essential_columns) >= n_features:
                    break
        
        # If we still need more features, add remaining columns until we reach n_features
        if len(essential_columns) < n_features:
            remaining_cols = [col for col in df.columns if col not in essential_columns]
            additional_cols = remaining_cols[:n_features - len(essential_columns)]
            essential_columns.extend(additional_cols)
        
        # Trim to exactly n_features if we have too many
        essential_columns = essential_columns[:n_features]
        
        logging.info(f"Selected {len(essential_columns)} features: {essential_columns}")
        
        # Return DataFrame with only selected features
        return df[essential_columns]
    
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
        
        # Reorganize columns to ensure important features come first
        if 'Close' in df.columns:
            # Define desired priority columns
            desired_priority_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
            
            # Filter to only use priority columns that actually exist in the dataframe
            priority_columns = [col for col in desired_priority_columns if col in df.columns]
            
            if priority_columns:
                # Check if we have alternative names for standard columns
                alt_cols = {}
                for std_col in desired_priority_columns:
                    if std_col not in df.columns:
                        # Look for alternative columns containing the standard name
                        alt_candidates = [col for col in df.columns if std_col.lower() in str(col).lower()]
                        if alt_candidates:
                            alt_cols[std_col] = alt_candidates[0]
                
                # Add alternative column names to priority columns
                priority_columns.extend(alt_cols.values())
                
                # Get all other columns (excluding priority ones)
                other_columns = [col for col in df.columns if col not in priority_columns]
                
                # Reorganize columns with priority columns first
                df = df[priority_columns + other_columns]
                logging.info(f"Reorganized columns to prioritize core price data: {priority_columns}")
            else:
                logging.warning("No standard price columns found for reorganization")

        # Check for model metadata to ensure feature compatibility
        model_metadata = self.load_model_metadata()
        feature_dim = None
        if model_metadata:
            if 'feature_dim' in model_metadata:
                feature_dim = model_metadata['feature_dim']
                logging.info(f"Found model with {feature_dim} expected features")
            elif 'feature_count' in model_metadata:
                feature_dim = model_metadata['feature_count']
                logging.info(f"Found model with {feature_dim} expected features")
        
        # Scale the closing prices for prediction
        close_prices = df['Close'].values.reshape(-1, 1)
        self.price_scaler.fit(close_prices)
        
        # Separate sentiment features for specialized scaling
        feature_columns = df.columns.tolist()
        sentiment_cols = [col for col in feature_columns if isinstance(col, str) and col.startswith('news_')]
        non_sentiment_cols = [col for col in feature_columns if col not in sentiment_cols]
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_scaled = df.copy()
        
        # Scale non-sentiment features
        if non_sentiment_cols:
            non_sentiment_data = df[non_sentiment_cols].values
            self.feature_scaler.fit(non_sentiment_data)
            df_scaled[non_sentiment_cols] = self.feature_scaler.transform(non_sentiment_data)
        
        # Scale sentiment features separately if they exist
        if sentiment_cols:
            sentiment_data = df[sentiment_cols].fillna(0).values
            self.sentiment_scaler.fit(sentiment_data)
            df_scaled[sentiment_cols] = self.sentiment_scaler.transform(sentiment_data)
            
            # Save the sentiment feature names for later use
            np.save(os.path.join(self.processed_dir, "sentiment_columns.npy"), sentiment_cols)
        
        # If feature_dim is available from model metadata, adjust features for compatibility
        if feature_dim is not None and len(df_scaled.columns) != feature_dim:
            df_original = df_scaled.copy()
            df_scaled = self.adjust_features_for_compatibility(df_scaled, feature_dim)
            logging.info(f"Adjusted features from {len(df_original.columns)} to {len(df_scaled.columns)} for model compatibility")
            # Update feature_columns after adjustment
            feature_columns = df_scaled.columns.tolist()
        
        # Apply feature selection if requested
        if self.reduced_features:
            original_feature_count = len(df.columns)
            df = self.select_important_features(df, n_features=self.feature_count)
            logging.info(f"Reduced features from {original_feature_count} to {len(df.columns)} for faster training")
        
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
        
        # Save feature names for future reference
        try:
            feature_names = np.array(df_scaled.columns)
            np.save(os.path.join(self.processed_dir, "feature_names.npy"), feature_names)
            logging.info(f"Saved {len(feature_names)} feature names for reference")
            
            # Save priority feature information - more robust approach
            priority_features = []
            priority_patterns = ['close', 'open', 'high', 'low', 'volume']
            
            for col in df_scaled.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in priority_patterns):
                    priority_features.append(col)
                    
            np.save(os.path.join(self.processed_dir, "priority_features.npy"), priority_features)
            logging.info(f"Saved {len(priority_features)} priority features: {priority_features}")
        except Exception as e:
            logging.warning(f"Could not save feature names: {e}")
        
        # After preparing the data, add regime detection if requested
        if self.detect_regimes and 'Close' in df.columns:
            prices = df['Close'].values
            volumes = df['Volume'].values if 'Volume' in df.columns else None
            
            # Identify market regimes
            regimes = self.identify_market_regimes(prices, volumes)
            
            # Store regime information with prepared data
            processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            # Map regimes to the proper timeframe (accounting for windows)
            if regimes is not None:
                # The windowed data has fewer samples due to the window creation
                # We need to align regimes with the actual training windows
                train_regimes = regimes[self.window_size:len(regimes)-self.prediction_horizon]
                if len(train_regimes) > len(X_train):
                    train_regimes = train_regimes[:len(X_train)]
                elif len(train_regimes) < len(X_train):
                    # Pad with the most common regime if needed
                    most_common_regime = np.bincount(train_regimes).argmax()
                    train_regimes = np.pad(
                        train_regimes, 
                        (0, len(X_train) - len(train_regimes)),
                        'constant', 
                        constant_values=most_common_regime
                    )
                
                np.save(os.path.join(processed_dir, "train_regimes.npy"), train_regimes)
                logging.info(f"Saved {len(train_regimes)} regime labels for training data")
        
        logging.info("Data processing complete and saved.")
        
        return X_train, X_test, y_train, y_test
