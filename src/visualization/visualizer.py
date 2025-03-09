import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def plot_stock_history_and_prediction(historical_data, predicted_data, 
                                      save_path=None, figsize=(14, 8), 
                                      show_plot=True, title=None):
    """
    Plot stock price history and predictions with clear visual distinction.
    
    Parameters:
    - historical_data: DataFrame with 'Date' and 'Close' columns for historical data
    - predicted_data: DataFrame with 'Date' and 'Close' columns for prediction data
    - save_path: Optional path to save the figure
    - figsize: Figure size (width, height) in inches
    - show_plot: Whether to display the plot
    - title: Custom title for the plot
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Convert dates to datetime if they're not already
    if not pd.api.types.is_datetime64_any_dtype(historical_data['Date']):
        historical_data = historical_data.copy()
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
    if not pd.api.types.is_datetime64_any_dtype(predicted_data['Date']):
        predicted_data = predicted_data.copy()
        predicted_data['Date'] = pd.to_datetime(predicted_data['Date'])
    
    # Get the separation point (last historical date)
    separation_date = historical_data['Date'].iloc[-1]
    
    # Plot historical data in blue
    plt.plot(historical_data['Date'], historical_data['Close'], 
             color='royalblue', linewidth=2, label='Historical Data')
    
    # Update the plot to highlight forecasted data differently
    if 'is_forecast' in predicted_data.columns:
        # Separate actual prediction data and forecast data
        actual_prediction = predicted_data[predicted_data['is_forecast'] == False]
        forecast_data = predicted_data[predicted_data['is_forecast'] == True]
        
        # Plot actual prediction if available
        if not actual_prediction.empty:
            plt.plot(actual_prediction['Date'], actual_prediction['Close'],
                    color='crimson', linewidth=2, label='Actual Data', linestyle='-')
        
        # Plot forecast data with different style
        if not forecast_data.empty:
            plt.plot(forecast_data['Date'], forecast_data['Close'],
                    color='orange', linewidth=2, label='Forecast', linestyle='--')
            
            # Add shaded area for forecast region
            min_price = min(historical_data['Close'].min(), predicted_data['Close'].min()) * 0.95
            max_price = max(historical_data['Close'].max(), predicted_data['Close'].max()) * 1.05
            plt.fill_betweenx([min_price, max_price],
                            forecast_data['Date'].min(),
                            forecast_data['Date'].max(),
                            color='lightyellow', alpha=0.3)
            
            # Add annotation for forecast
            plt.annotate('Forecasted Data', 
                        xy=(forecast_data['Date'].iloc[len(forecast_data)//2], 
                            forecast_data['Close'].iloc[len(forecast_data)//2]),
                        xytext=(0, -50), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'), fontsize=10)
    else:
        # Original plot code if 'is_forecast' flag is not available
        plt.plot(predicted_data['Date'], predicted_data['Close'],
                color='crimson', linewidth=2, label='Prediction', linestyle='-')
    
    # Draw vertical line at the separation point
    plt.axvline(x=separation_date, color='black', linestyle='--', 
                linewidth=1.5, label='Prediction Start')
    
    # Add shaded area for prediction region
    min_price = min(historical_data['Close'].min(), predicted_data['Close'].min()) * 0.95
    max_price = max(historical_data['Close'].max(), predicted_data['Close'].max()) * 1.05
    
    plt.fill_betweenx([min_price, max_price],
                      predicted_data['Date'].min(),
                      predicted_data['Date'].max(),
                      color='mistyrose', alpha=0.3)
    
    # Customize the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    
    # Set the title
    if title is None:
        title = 'Tesla (TSLA) Stock Price: Historical Data and Predictions'
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Add annotations
    plt.annotate('Historical Data', 
                 xy=(historical_data['Date'].iloc[len(historical_data)//2], 
                     historical_data['Close'].iloc[len(historical_data)//2]),
                 xytext=(0, 30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    plt.annotate('Predicted Data', 
                 xy=(predicted_data['Date'].iloc[len(predicted_data)//2], 
                     predicted_data['Close'].iloc[len(predicted_data)//2]),
                 xytext=(0, -30), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), fontsize=10)
    
    # Add legend
    plt.legend(loc='best', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_trading_simulation_results(simulation_data, save_path=None, figsize=(14, 10), show_plot=True):
    """
    Plot simulation results including stock prices, predictions, and trading activities.
    
    Parameters:
    - simulation_data: DataFrame with simulation results including:
        - 'Date': dates
        - 'Close': actual closing prices
        - 'Predicted': predicted prices
        - 'Decision': trading decisions
    - save_path: Optional path to save the figure
    - figsize: Figure size (width, height) in inches
    - show_plot: Whether to display the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(simulation_data['Date']):
        simulation_data = simulation_data.copy()
        simulation_data['Date'] = pd.to_datetime(simulation_data['Date'])
    
    # Check if we have forecast data
    has_forecast = 'is_forecast' in simulation_data.columns
    
    # Plot 1: Stock prices and predictions
    if has_forecast:
        # Separate historical and forecast data
        historical = simulation_data[simulation_data['is_forecast'] == False]
        forecast = simulation_data[simulation_data['is_forecast'] == True]
        
        # Plot historical data if available
        if not historical.empty:
            ax1.plot(historical['Date'], historical['Close'], 
                    color='royalblue', linewidth=2, label='Historical Data')
        
        # Plot forecast data
        if not forecast.empty:
            ax1.plot(forecast['Date'], forecast['Close'],
                    color='orange', linewidth=2, label='Forecast', linestyle='--')
            
            # Add shaded area for forecast region
            min_price = simulation_data['Close'].min() * 0.95
            max_price = simulation_data['Close'].max() * 1.05
            ax1.fill_betweenx([min_price, max_price],
                            forecast['Date'].min(),
                            forecast['Date'].max(),
                            color='lightyellow', alpha=0.3)
            
            # Add vertical line separating historical from forecast
            if not historical.empty and not forecast.empty:
                ax1.axvline(x=historical['Date'].iloc[-1], color='black', linestyle='--', 
                        linewidth=1.5, label='Forecast Start')
    else:
        # Original plot code if no forecast data is available
        ax1.plot(simulation_data['Date'], simulation_data['Close'], 
                color='royalblue', linewidth=2, label='Actual Price')
    
    # Plot predicted prices if available
    if 'Predicted' in simulation_data.columns:
        # Only plot non-NaN values
        pred_data = simulation_data.dropna(subset=['Predicted'])
        if not pred_data.empty:
            ax1.plot(pred_data['Date'], pred_data['Predicted'],
                    color='crimson', linewidth=2, label='Model Prediction', linestyle=':')
    
    # Mark buy signals
    buy_signals = simulation_data[simulation_data['Decision'] == 'Buy']
    if not buy_signals.empty:
        ax1.scatter(buy_signals['Date'], buy_signals['Close'], 
                   color='green', s=100, marker='^', label='Buy Signal')
    
    # Mark sell signals
    sell_signals = simulation_data[simulation_data['Decision'] == 'Sell']
    if not sell_signals.empty:
        ax1.scatter(sell_signals['Date'], sell_signals['Close'], 
                   color='red', s=100, marker='v', label='Sell Signal')
    
    # Updated title to indicate forecast
    if has_forecast and not forecast.empty:
        ax1.set_title('Tesla (TSLA) Stock Price Forecast with Trading Signals', fontsize=16, fontweight='bold')
    else:
        ax1.set_title('Tesla (TSLA) Stock Price with Trading Signals', fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', fontsize=12)
    
    # Plot 2: Portfolio value
    if 'Portfolio_Value' in simulation_data.columns:
        ax2.plot(simulation_data['Date'], simulation_data['Portfolio_Value'], 
                 color='purple', linewidth=2)
        ax2.set_title('Projected Portfolio Value Over Time', fontsize=14)
        ax2.set_ylabel('Value ($)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add vertical line for forecast start in portfolio plot too
        if has_forecast and not historical.empty and not forecast.empty:
            ax2.axvline(x=historical['Date'].iloc[-1], color='black', linestyle='--', 
                    linewidth=1.5)
            
            # Add shaded area for forecast region in portfolio plot
            min_val = simulation_data['Portfolio_Value'].min() * 0.95
            max_val = simulation_data['Portfolio_Value'].max() * 1.05
            ax2.fill_betweenx([min_val, max_val],
                            forecast['Date'].min(),
                            forecast['Date'].max(),
                            color='lightyellow', alpha=0.3)
    
    # Common formatting
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    ax2.set_xlabel('Date', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
