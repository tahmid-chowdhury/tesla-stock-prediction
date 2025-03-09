import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import visualization utilities
from src.visualization.visualizer import plot_stock_history_and_prediction, plot_trading_simulation_results

class TradingSimulation:
    def __init__(self, agent, data, sequence_length, start_date, end_date):
        """
        Initialize backtesting environment
        
        Parameters:
        - agent: Trading agent instance
        - data: Processed data DataFrame
        - sequence_length: Length of input sequences for prediction
        - start_date: Start date for simulation
        - end_date: End date for simulation
        """
        self.agent = agent
        self.data = data
        self.sequence_length = sequence_length
        self.start_date = start_date
        self.end_date = end_date
        self.results = None
        
    def prepare_simulation_data(self):
        """Prepare data for simulation"""
        # Filter data for the simulation period
        mask = (self.data['Date'] >= self.start_date) & (self.data['Date'] <= self.end_date)
        sim_data = self.data[mask].copy()
        
        # Add columns for tracking simulation with proper data types
        sim_data['Predicted'] = np.nan
        sim_data['Decision'] = 'Hold'
        sim_data['Amount'] = 0.0
        
        # Ensure float type for numeric columns that will be updated
        sim_data['Portfolio_Value'] = float(self.agent.initial_balance)
        sim_data['Balance'] = float(self.agent.initial_balance)
        sim_data['Shares'] = 0.0  # Changed to float to avoid dtype warnings
        sim_data['Trade_Cost'] = 0.0
        sim_data['Trade_Fee'] = 0.0
        
        return sim_data
        
    def run_simulation(self):
        """Run trading simulation"""
        # Reset agent state
        self.agent.balance = self.agent.initial_balance
        self.agent.shares = 0
        self.agent.transaction_history = []
        
        # Prepare data
        sim_data = self.prepare_simulation_data()
        
        # Get feature columns
        feature_columns = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 
                          'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volatility', 'Return']
        
        # Run simulation for each trading day
        for i in range(len(sim_data)):
            # Get current date and closing price
            current_date = sim_data.iloc[i]['Date']
            current_price = sim_data.iloc[i]['Close']
            
            # Prepare input data for prediction
            # Find the historical data up to the current point
            historical_idx = self.data[self.data['Date'] < current_date].index
            if len(historical_idx) < self.sequence_length:
                # Not enough historical data
                continue
                
            # Get the most recent sequence_length data points
            seq_start_idx = historical_idx[-self.sequence_length]
            seq_data = self.data.loc[seq_start_idx:historical_idx[-1], feature_columns].values
            
            # Scale the data
            scaled_seq = self.agent.scaler.transform(seq_data)
            
            # Reshape for model input [1, sequence_length, n_features]
            model_input = scaled_seq.reshape(1, self.sequence_length, len(feature_columns))
            
            # Make trading decision - pass the complete dataset for YTD analysis
            decision, amount, order_text = self.agent.make_decision(
                model_input, current_price, next_day=current_date, available_data=self.data)
            
            # Record decision
            sim_data.at[sim_data.index[i], 'Decision'] = decision
            sim_data.at[sim_data.index[i], 'Amount'] = float(amount)  # Ensure float type
            
            # Execute the order
            self.agent.execute_order(decision, amount, current_price)
            
            # Record portfolio value
            portfolio_value = self.agent.get_portfolio_value(current_price)
            sim_data.at[sim_data.index[i], 'Portfolio_Value'] = float(portfolio_value)  # Ensure float type
            sim_data.at[sim_data.index[i], 'Balance'] = float(self.agent.balance)  # Ensure float type
            sim_data.at[sim_data.index[i], 'Shares'] = float(self.agent.shares)  # Ensure float type
            
            # Make prediction for next day's price and record it
            if i < len(sim_data) - 1:
                next_price_prediction = self.agent.predict_price_movement(model_input)
                
                # Unscale prediction
                price_min = self.agent.scaler.data_min_[0]
                price_max = self.agent.scaler.data_max_[0]
                predicted_price = next_price_prediction * (price_max - price_min) + price_min
                
                sim_data.at[sim_data.index[i+1], 'Predicted'] = float(predicted_price)  # Ensure float type
        
        # Store results
        self.results = sim_data
        
        return sim_data
    
    def get_performance_summary(self):
        """Calculate performance metrics"""
        if self.results is None:
            raise ValueError("Simulation has not been run yet.")
            
        if self.results.empty:
            return {
                'initial_balance': self.agent.initial_balance,
                'final_value': self.agent.initial_balance,
                'total_return': 0,
                'return_pct': 0,
                'transactions': 0,
                'total_fees': 0,
                'buys': 0,
                'sells': 0,
                'holds': 0,
                'stock_return_pct': 0,
                'outperformance_pct': 0
            }
            
        # Get initial and final portfolio values
        initial_value = self.agent.initial_balance
        final_value = self.results.iloc[-1]['Portfolio_Value']
        
        # Calculate returns
        total_return = final_value - initial_value
        pct_return = (total_return / initial_value) * 100
        
        # Calculate transactions and fees
        transactions = len(self.agent.transaction_history)
        total_fees = sum([tx.get('fee', 0) for tx in self.agent.transaction_history])
        
        # Calculate buy/sell decisions
        buys = len([d for d in self.results['Decision'] if d == 'Buy'])
        sells = len([d for d in self.results['Decision'] if d == 'Sell'])
        holds = len([d for d in self.results['Decision'] if d == 'Hold'])
        
        # Calculate stock performance for comparison
        if len(self.results) > 1:
            stock_start_price = self.results.iloc[0]['Close']
            stock_end_price = self.results.iloc[-1]['Close']
            stock_return = (stock_end_price - stock_start_price) / stock_start_price * 100
        else:
            stock_return = 0
        
        # Return performance summary
        return {
            'initial_balance': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': pct_return,
            'transactions': transactions,
            'total_fees': total_fees,
            'buys': buys,
            'sells': sells,
            'holds': holds,
            'stock_return_pct': stock_return,
            'outperformance_pct': pct_return - stock_return
        }
    
    def plot_results(self, save_path=None, show_plot=True):
        """Plot simulation results"""
        if self.results is None:
            raise ValueError("Simulation has not been run yet.")
        
        # Use the enhanced visualization function
        plot_trading_simulation_results(
            self.results,
            save_path=save_path,
            show_plot=show_plot
        )
        
        # Get data for history vs prediction visualization
        # Historical data is everything before start_date
        historical_data = self.data[self.data['Date'] < self.start_date].copy()
        
        # Prediction data includes both actual and predicted values in the simulation period
        prediction_data = self.results.copy()
        
        # Combine the last point of historical data with prediction data for continuity
        if not historical_data.empty and not prediction_data.empty:
            last_historical = historical_data.iloc[[-1]].copy()
            # Create a separate visualization for historical vs prediction data
            plot_stock_history_and_prediction(
                historical_data,
                prediction_data,
                save_path=save_path.replace('.png', '_history_vs_prediction.png') if save_path else None,
                show_plot=show_plot,
                title='Tesla (TSLA) Stock Price: History and Simulation Period'
            )
