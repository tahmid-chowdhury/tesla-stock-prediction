import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def compare_strategies(historical_data, forecast_data, initial_balance=10000, 
                     agent_results=None, save_path=None):
    """
    Compare different trading strategies on the same data
    
    Strategies:
    1. ML Agent's strategy (from simulation results)
    2. Buy and Hold
    3. Simple Moving Average Crossover
    """
    # Combine historical and forecast for complete timeline
    combined = pd.concat([historical_data, forecast_data])
    combined = combined.sort_values('Date')
    
    # Prepare results DataFrame
    dates = forecast_data['Date'].tolist()
    strategies = ['ML Agent', 'Buy and Hold', 'SMA Crossover']
    results = pd.DataFrame(index=dates, columns=strategies)
    
    # 1. ML Agent strategy (from simulation)
    if agent_results is not None:
        results['ML Agent'] = agent_results['Portfolio_Value'].values
    
    # 2. Buy and Hold strategy
    initial_price = combined[combined['Date'] == dates[0]]['Close'].iloc[0]
    shares_bought = initial_balance / initial_price
    
    for date in dates:
        price = combined[combined['Date'] == date]['Close'].iloc[0]
        results.loc[date, 'Buy and Hold'] = shares_bought * price
    
    # 3. Simple Moving Average Crossover (5-day and 20-day)
    sma_balance = initial_balance
    sma_shares = 0
    in_position = False
    
    for i, date in enumerate(dates):
        price = combined[combined['Date'] == date]['Close'].iloc[0]
        ma5 = combined[combined['Date'] == date]['MA5'].iloc[0]
        ma20 = combined[combined['Date'] == date]['MA20'].iloc[0]
        
        # Buy signal: 5-day MA crosses above 20-day MA
        if ma5 > ma20 and not in_position:
            sma_shares = sma_balance * 0.95 / price  # 5% kept as cash
            sma_balance = sma_balance * 0.05
            in_position = True
            
        # Sell signal: 5-day MA crosses below 20-day MA
        elif ma5 < ma20 and in_position:
            sma_balance += sma_shares * price * 0.99  # 1% transaction fee
            sma_shares = 0
            in_position = False
            
        # Calculate portfolio value
        results.loc[date, 'SMA Crossover'] = sma_balance + (sma_shares * price)
    
    # Calculate performance metrics
    performance = {strategy: {
        'final_value': results[strategy].iloc[-1],
        'return_pct': (results[strategy].iloc[-1] / initial_balance - 1) * 100
    } for strategy in strategies if strategy in results.columns}
    
    # Plot strategy comparison
    plt.figure(figsize=(12, 8))
    
    for strategy in results.columns:
        if not results[strategy].isna().all():
            plt.plot(results.index, results[strategy], label=f"{strategy} (Return: {performance[strategy]['return_pct']:.1f}%)")
    
    plt.title('Strategy Comparison - Forecasted Period', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Strategy comparison saved to {save_path}")
    
    plt.show()
    
    return results, performance
