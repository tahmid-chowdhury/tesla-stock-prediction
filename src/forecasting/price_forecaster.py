import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockForecaster:
    def __init__(self, model, scaler, sequence_length, feature_columns):
        """
        Initialize the stock price forecaster
        
        Parameters:
        - model: Trained prediction model
        - scaler: Fitted scaler for data normalization
        - sequence_length: Length of input sequences for model
        - feature_columns: List of column names for features
        """
        self.model = model
        self.scaler = scaler
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
    
    def forecast_next_days(self, historical_data, future_dates):
        """
        Generate forecasts for future dates
        
        Parameters:
        - historical_data: DataFrame with historical data
        - future_dates: List of datetime objects for future dates
        
        Returns:
        - DataFrame with forecasted data for future dates
        """
        # Create a copy of historical data to avoid modifying the original
        df = historical_data.copy()
        
        # Sort by date to ensure proper order
        df = df.sort_values('Date')
        
        # Create a template row based on the last historical row
        template_row = df.iloc[-1].copy()
        
        # Create forecasted DataFrame
        forecast_data = []
        
        # Iterate through future dates
        for future_date in future_dates:
            # Get latest available sequence for prediction
            latest_data = df.tail(self.sequence_length)[self.feature_columns].values
            
            # Handle the MinMaxScaler feature name warning by creating a DataFrame with named features
            # This helps silence the warning about feature names when transforming
            try:
                feature_df = pd.DataFrame(latest_data, columns=self.feature_columns)
                scaled_data = self.scaler.transform(feature_df)
            except:
                # Fallback to direct transformation if DataFrame approach fails
                scaled_data = self.scaler.transform(latest_data)
            
            # Reshape for model input [1, sequence_length, n_features]
            model_input = scaled_data.reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Make prediction
            prediction = self.model.predict(model_input)
            
            # Unscale prediction (only the Close price)
            price_min = self.scaler.data_min_[0]
            price_max = self.scaler.data_max_[0]
            predicted_close = prediction[0][0] * (price_max - price_min) + price_min
            
            # Create new row for this forecasted date
            new_row = template_row.copy()
            new_row['Date'] = future_date
            new_row['Close'] = float(predicted_close)  # Ensure float type
            
            # Estimate other price columns based on historical relationships
            # (in a real system, you'd have more sophisticated methods)
            price_range = 0.02  # Assume 2% daily range
            new_row['Open'] = float(predicted_close * (1 - np.random.uniform(0, price_range/2)))
            new_row['High'] = float(predicted_close * (1 + np.random.uniform(0, price_range)))
            new_row['Low'] = float(predicted_close * (1 - np.random.uniform(0, price_range)))
            
            # Simple heuristic for volume (use average of last 5 days)
            new_row['Volume'] = float(df.tail(5)['Volume'].mean())
            
            # Add is_forecast flag to identify forecasted data
            new_row['is_forecast'] = True
            
            # Append to forecast data
            forecast_data.append(new_row)
            
            # Add this row to the dataframe to use for next prediction
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Create a DataFrame from the forecast data
        forecast_df = pd.DataFrame(forecast_data)
        
        # Calculate technical indicators for forecast data
        # (simplified here, would be more sophisticated in production)
        forecast_df['MA5'] = df.tail(len(forecast_data) + 4)['Close'].rolling(window=5).mean().tail(len(forecast_data)).values.astype(float)
        forecast_df['MA10'] = df.tail(len(forecast_data) + 9)['Close'].rolling(window=10).mean().tail(len(forecast_data)).values.astype(float)
        forecast_df['MA20'] = df.tail(len(forecast_data) + 19)['Close'].rolling(window=20).mean().tail(len(forecast_data)).values.astype(float)
        
        # Other indicators can be calculated similarly
        # For simplicity, we'll use the last values from historical data
        for col in self.feature_columns:
            if col not in forecast_df.columns and col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20']:
                forecast_df[col] = df[col].iloc[-len(forecast_data):].values.astype(float)
        
        return forecast_df
