import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class TradingSimulation:
    def __init__(self, agent, processed_data, sequence_length, start_date=None, end_date=None):
        """
        Initialize trading simulation
        
        Parameters:
        - agent: Trading agent
        - processed_data: Processed stock data
        - sequence_length: Length of sequence for predictions
        - start_date: Start date for simulation (str or datetime)
        - end_date: End date for simulation (str or datetime)
        """
        self.agent = agent
        self.data = processed_data
        self.sequence_length = sequence_length
        
        # Set simulation date range
        if start_date is None:
            # Default to last week of March 2025 (March 24-28, 2025)
            self.start_date = datetime(2025, 3, 24)
        else:
            self.start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            self.end_date = datetime(2025, 3, 28)
        else:
            self.end_date = pd.to_datetime(end_date)
        
        self.results = []
    
    def generate_simulated_data(self):
        """Generate simulated data for March 24-28, 2025 based on historical patterns"""
        # Use the most recent data as a base
        base_data = self.data.copy()
        
        # Create date range for simulation
        simulation_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
        # Select recent price movements as a pattern
        recent_pattern = base_data.tail(len(simulation_dates)).copy()
        recent_pattern['Date'] = simulation_dates
        
        # Add some random variation to prices (±3%)
        for col in ['Open', 'High', 'Low', 'Close']:
            variation = np.random.uniform(-0.03, 0.03, len(recent_pattern))
            base_values = recent_pattern[col].values
            recent_pattern[col] = base_values * (1 + variation)
        
        return recent_pattern
    
    def run_simulation(self):
        """Run the trading simulation"""
        # Generate simulated data for March 24-28, 2025
        simulation_data = self.generate_simulated_data()
        
        for i in range(len(simulation_data)):
            current_date = simulation_data.iloc[i]['Date']
            
            # Skip if before simulation start date
            if current_date < self.start_date:
                continue
                
            # Stop if after simulation end date
            if current_date > self.end_date:
                break
            
            print(f"\nTrading day: {current_date.strftime('%Y-%m-%d')}")
            
            # Get current price
            current_price = simulation_data.iloc[i]['Open']
            
            # Prepare sequence data for model prediction
            # We need to prepare this from the feature columns
            feature_data = self.data.iloc[-(self.sequence_length+1):-1]
            feature_columns = ['Close', 'Volume', 'MA5', 'MA10', 'MA20', 'RSI', 
                              'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Volatility', 'Return']
            
            # Scale the data
            scaled_features = self.agent.scaler.transform(feature_data[feature_columns])
            scaled_sequence = np.array([scaled_features])
            
            # Make trading decision (9:00 AM)
            decision, amount, order = self.agent.make_decision(
                scaled_sequence, 
                current_price,
                next_day=current_date.strftime('%Y-%m-%d')
            )
            
            print(f"9:00 AM Decision: {order}")
            
            # Execute order at 10:00 AM price
            execution_price = simulation_data.iloc[i]['Close']
            executed = self.agent.execute_order(decision, amount, execution_price)
            
            # Record results
            portfolio_value = self.agent.get_portfolio_value(execution_price)
            
            result = {
                'date': current_date,
                'open_price': current_price,
                'close_price': execution_price,
                'decision': decision,
                'amount': amount,
                'order': order,
                'executed': executed,
                'balance': self.agent.balance,
                'shares': self.agent.shares,
                'portfolio_value': portfolio_value
            }
            
            self.results.append(result)
            
            print(f"10:00 AM Execution Price: ${execution_price:.2f}")
            print(f"Balance: ${self.agent.balance:.2f}")
            print(f"Shares: {self.agent.shares}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
        
        return pd.DataFrame(self.results)
    
    def plot_results(self, save_path=None):
        """Plot simulation results"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        results_df = pd.DataFrame(self.results)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot stock price
        ax1.plot(results_df['date'], results_df['close_price'], label='Stock Price', color='blue')
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title('Tesla Stock Price During Simulation')
        ax1.grid(True)
        
        # Plot portfolio value
        ax2.plot(results_df['date'], results_df['portfolio_value'], label='Portfolio Value', color='green')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_xlabel('Date')
        ax2.set_title('Portfolio Value Over Time')
        ax2.grid(True)
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def get_performance_summary(self):
        """Generate performance summary"""
        if not self.results:
            return "No results available. Run simulation first."
        
        results_df = pd.DataFrame(self.results)
        
        initial_value = self.agent.initial_balance
        final_value = results_df.iloc[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        summary = {
            'initial_balance': initial_value,
            'final_balance': results_df.iloc[-1]['balance'],
            'final_shares': results_df.iloc[-1]['shares'],
            'final_portfolio_value': final_value,
            'total_return_pct': total_return,
            'num_trades': len(self.agent.transaction_history),
            'transaction_fees': sum(t['fee'] for t in self.agent.transaction_history)
        }
        
        return summary
