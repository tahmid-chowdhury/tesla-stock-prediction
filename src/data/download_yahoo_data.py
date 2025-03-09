import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(ticker="TSLA", start_date="2010-01-01", output_dir=None):
    """
    Download stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock ticker symbol (default: "TSLA")
    - start_date: Start date for data collection (default: "2010-01-01")
    - output_dir: Directory to save the data (default: "../../data/raw/")
    
    Returns:
    - df: DataFrame containing the downloaded data
    """
    # Calculate end date as yesterday
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    # Download data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Fix column names
    if isinstance(df.columns, pd.MultiIndex):
        # Check if we have standard price columns with ticker as second level
        # Typical format: ('Close', 'TSLA'), ('High', 'TSLA'), etc.
        if all(col[1] == ticker or col[1] == '' for col in df.columns if isinstance(col, tuple)):
            standard_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            actual_cols = []
            
            for i, col in enumerate(df.columns):
                if i == 0:  # First column is Date
                    actual_cols.append('Date')
                elif col[0] in standard_cols:
                    # Use first level if it's a standard column name
                    actual_cols.append(col[0])
                elif col[1] == ticker:
                    # If first level is missing but second is ticker, use position to determine name
                    pos = i % len(standard_cols)
                    actual_cols.append(standard_cols[pos])
                else:
                    # Fallback - use both levels joined
                    actual_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
                    
            df.columns = actual_cols
        else:
            # Fallback to previous method if pattern doesn't match
            df.columns = [col[1] if col[1] != "" else col[0] for col in df.columns]
    
    # Check if columns are all the same (except Date)
    value_cols = [col for col in df.columns if col != 'Date']
    if len(set(value_cols)) == 1 and len(value_cols) >= 5:
        # Rename columns based on standard order
        standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        new_cols = ['Date'] + standard_cols
        if len(df.columns) > len(new_cols):
            new_cols.insert(5, 'Adj Close')  # Add Adj Close if we have enough columns
        df.columns = new_cols[:len(df.columns)]
    
    # Ensure the output directory exists
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "../../data/raw/")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print(f"Downloaded {len(df)} rows of data")
    
    # Print column names for debugging
    print(f"Column names in DataFrame: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    # Change these parameters as needed
    ticker = "TSLA"
    start_date = "2010-01-01"  # Tesla's IPO was on June 29, 2010
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory relative to script location
    output_dir = os.path.join(script_dir, "../../data/raw/")
    
    # Download the data
    df = download_stock_data(ticker, start_date, output_dir)
    
    # Display the first few rows
    print("\nData Preview:")
    print(df.head())
    
    # Display basic statistics - fix for handling potentially complex objects
    print("\nData Statistics:")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total Trading Days: {len(df)}")
    
    # Use try/except for each statistic individually
    try:
        if 'Close' in df.columns:
            print(f"Min Close Price: ${float(df['Close'].min()):.2f}")
            print(f"Max Close Price: ${float(df['Close'].max()):.2f}")
        else:
            print("Close price column not found")
        
        if 'Volume' in df.columns:
            print(f"Average Volume: {float(df['Volume'].mean()):.0f}")
        else:
            print("Volume column not found")
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        print("Available columns:", df.columns.tolist())
