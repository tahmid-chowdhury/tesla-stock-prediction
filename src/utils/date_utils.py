import pandas as pd
from datetime import datetime, timedelta

def generate_future_trading_days(start_date, num_days=5):
    """
    Generate future trading days (weekdays only)
    
    Parameters:
    - start_date: Starting date (datetime object)
    - num_days: Number of trading days to generate
    
    Returns:
    - List of datetime objects representing future trading days
    """
    # Ensure start_date is a datetime object
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # If start_date is weekend, move to next Monday
    if start_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        days_until_monday = 7 - start_date.weekday()
        start_date = start_date + timedelta(days=days_until_monday)
    
    # Generate future dates
    future_dates = []
    current_date = start_date
    days_added = 0
    
    while days_added < num_days:
        current_date = current_date + timedelta(days=1)
        # Skip weekends
        if current_date.weekday() < 5:  # Monday to Friday
            future_dates.append(current_date)
            days_added += 1
    
    return future_dates

def get_last_trading_day():
    """
    Get the last trading day (today if weekday, Friday if weekend)
    
    Returns:
    - datetime object representing the last trading day
    """
    today = datetime.now()
    
    # If weekend, return the previous Friday
    if today.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        days_to_subtract = today.weekday() - 4  # 4 = Friday
        return today - timedelta(days=days_to_subtract)
    
    return today
