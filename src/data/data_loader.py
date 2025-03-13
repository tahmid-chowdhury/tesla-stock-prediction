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
        """
        Initialize DataLoader with stock ticker and optional NewsAPI key
        """
        self.ticker = ticker
        self.api_key = api_key
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def fetch_stock_data(self, start_date=None, end_date=None):
        """
        Fetch historical stock data from Yahoo Finance
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logging.info(f"Fetching {self.ticker} data from {start_date} to {end_date}")
        
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date)
            
            # Save raw data to CSV
            csv_path = os.path.join(self.raw_dir, f"{self.ticker}.csv")
            data.to_csv(csv_path)
            logging.info(f"Stock data saved to {csv_path}")
            
            return data
        except Exception as e:
            logging.error(f"Error fetching stock data: {e}")
            return None
    
    def fetch_news_data(self, days=30):
        """
        Fetch news related to the ticker using NewsAPI
        """
        if not self.api_key:
            logging.warning("No NewsAPI key provided. Skipping news data.")
            return None
            
        newsapi = NewsApiClient(api_key=self.api_key)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logging.info(f"Fetching news for {self.ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            news = newsapi.get_everything(
                q=f"{self.ticker} OR Tesla",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt'
            )
            
            # Save news data to JSON
            json_path = os.path.join(self.raw_dir, f"{self.ticker}_news.json")
            with open(json_path, 'w') as f:
                json.dump(news, f)
            logging.info(f"News data saved to {json_path}")
            
            return news
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
