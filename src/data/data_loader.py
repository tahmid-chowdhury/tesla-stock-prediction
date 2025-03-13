import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, ticker="TSLA", api_key=None):
        self.ticker = ticker
        self.api_key = api_key
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def fetch_stock_data(self, start_date=None, end_date=None, period=None):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            start_date: Start date for data (format: YYYY-MM-DD)
            end_date: End date for data (format: YYYY-MM-DD)
            period: Alternative to start/end dates (e.g., 'max', '5y', '2y', '1y')
        
        Returns:
            pandas.DataFrame: Historical stock data
        """
        # If period is specified, use it instead of start/end dates
        if period:
            logging.info(f"Fetching {self.ticker} data for period: {period}")
            try:
                data = yf.download(self.ticker, period=period)
            except Exception as e:
                logging.error(f"Error fetching data with period {period}: {e}")
                # Fall back to default 1-year period
                period = None
                
        # If not using period or it failed, use start/end dates
        if not period:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            logging.info(f"Fetching {self.ticker} data from {start_date} to {end_date}")
            try:
                data = yf.download(self.ticker, start=start_date, end=end_date)
            except Exception as e:
                logging.error(f"Error fetching stock data: {e}")
                return None
        
        if data.empty:
            logging.error("Downloaded data is empty")
            return None
            
        # Save raw data to CSV
        csv_path = os.path.join(self.raw_dir, f"{self.ticker}.csv")
        data.to_csv(csv_path)
        logging.info(f"Stock data saved to {csv_path}")
        
        return data
        
    def fetch_complete_history(self):
        """
        Fetch the complete historical data for the stock
        """
        return self.fetch_stock_data(period='max')
        
    def combine_datasets(self, recent_data, historical_data):
        """
        Combine two datasets, keeping all rows from both without duplicates
        """
        if recent_data is None or historical_data is None:
            return recent_data if recent_data is not None else historical_data
            
        # Combine and remove duplicates, keeping newer data when duplicated
        combined = pd.concat([historical_data, recent_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        return combined
        
    def fetch_news_data(self, days_back=20):
        """
        Fetch news data for a ticker using NewsAPI
        
        Args:
            days_back: Number of days to look back (default: 20 for free tier limitation)
        """
        if not self.api_key:
            logging.warning("No NewsAPI key provided. Skipping news data retrieval.")
            return None
            
        # Calculate date range (limit to 20 days for free API tier)
        end_date = datetime.now()
        # Free tier of NewsAPI typically only allows 20 days back
        start_date = end_date - timedelta(days=min(days_back, 20))
        
        logging.info(f"Fetching news for {self.ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            newsapi = NewsApiClient(api_key=self.api_key)
            all_articles = newsapi.get_everything(
                q=self.ticker,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt'
            )
            
            # Save the data
            news_path = os.path.join(self.raw_dir, f"{self.ticker}_news.json")
            with open(news_path, 'w') as f:
                json.dump(all_articles, f)
                
            logging.info(f"News data saved to {news_path}")
            return all_articles
        except Exception as e:
            logging.error(f"Error fetching news data: {e}")
            return None

if __name__ == "__main__":
    loader = DataLoader("TSLA")
    stock_data = loader.fetch_stock_data()
    print(stock_data.head())
    
    # Uncomment to test news API if you have an API key
    # loader = DataLoader("TSLA", api_key="YOUR_API_KEY")
    # news_data = loader.fetch_news_data()
