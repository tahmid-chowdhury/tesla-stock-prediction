import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import os

def create_dashboard(historical_data, forecast_data, agent_trades, 
                   performance_metrics, save_path=None, show_plot=True):
    """
    Create a comprehensive dashboard visualizing predictions and performance
    """
    # Combine data
    hist_end = historical_data['Date'].max()
    
    # Create the dashboard layout
    plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # 1. Main price chart with forecast
    ax1 = plt.subplot(gs[0, :])
    
    # Plot historical data
    ax1.plot(historical_data['Date'], historical_data['Close'], 
             color='blue', label='Historical Data')
    
    # Plot forecast data with different style
    ax1.plot(forecast_data['Date'], forecast_data['Close'],
             color='orange', linestyle='--', label='Forecast')
    
    # Mark the separation point
    ax1.axvline(x=hist_end, color='black', linestyle=':', alpha=0.7)
    
    # Add buy/sell markers
    buys = agent_trades[agent_trades['Decision'] == 'Buy']
    sells = agent_trades[agent_trades['Decision'] == 'Sell']
    
    if not buys.empty:
        ax1.scatter(buys['Date'], buys['Close'], marker='^', color='green', s=100, label='Buy')
    if not sells.empty:
        ax1.scatter(sells['Date'], sells['Close'], marker='v', color='red', s=100, label='Sell')
    
    ax1.set_title('Tesla (TSLA) Stock Price & Forecast', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 2. Volume subplot
    ax2 = plt.subplot(gs[1, 0])
    ax2.bar(historical_data['Date'], historical_data['Volume'], color='blue', alpha=0.6)
    ax2.bar(forecast_data['Date'], forecast_data['Volume'], color='orange', alpha=0.6)
    ax2.set_title('Trading Volume', fontsize=14)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Technical indicators
    ax3 = plt.subplot(gs[1, 1])
    ax3.plot(historical_data['Date'], historical_data['MA5'], color='red', label='MA5')
    ax3.plot(historical_data['Date'], historical_data['MA20'], color='blue', label='MA20')
    ax3.plot(forecast_data['Date'], forecast_data['MA5'], color='red', linestyle='--')
    ax3.plot(forecast_data['Date'], forecast_data['MA20'], color='blue', linestyle='--')
    ax3.set_title('Moving Averages', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Portfolio value
    ax4 = plt.subplot(gs[2, 0])
    ax4.plot(agent_trades['Date'], agent_trades['Portfolio_Value'], color='purple')
    ax4.set_title('Portfolio Value', fontsize=14)
    ax4.set_ylabel('Value ($)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance metrics
    ax5 = plt.subplot(gs[2, 1])
    metrics = [
        f"Initial: ${performance_metrics['initial_balance']:.2f}",
        f"Final: ${performance_metrics['final_value']:.2f}",
        f"Return: {performance_metrics['return_pct']:.2f}%",
        f"Stock Return: {performance_metrics['stock_return_pct']:.2f}%",
        f"Alpha: {performance_metrics['outperformance_pct']:.2f}%",
        f"Trades: {performance_metrics['transactions']}",
        f"Fees: ${performance_metrics['total_fees']:.2f}"
    ]
    
    ax5.axis('off')
    y_pos = 0.9
    for metric in metrics:
        ax5.text(0.1, y_pos, metric, fontsize=12)
        y_pos -= 0.12
    
    ax5.set_title('Performance Summary', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
