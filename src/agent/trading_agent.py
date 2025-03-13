import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingAgent:
    def __init__(self, model, price_scaler, initial_capital=10000, transaction_fee=0.01, risk_factor=0.3,
                 stop_loss_pct=5.0, trailing_stop_pct=None):
        """
        Initialize the trading agent
        
        Args:
            model: Trained prediction model
            price_scaler: Scaler for price data
            initial_capital: Initial investment amount
            transaction_fee: Fee per transaction (as a percentage)
            risk_factor: Risk factor for position sizing (0.1-0.5)
            stop_loss_pct: Stop loss percentage (e.g., 5.0 means sell if price drops 5% from purchase)
            trailing_stop_pct: Trailing stop percentage (adjust stop loss as price increases)
        """
        self.model = model
        self.price_scaler = price_scaler
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.risk_factor = risk_factor
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.shares_owned = 0
        self.transaction_history = []
        self.portfolio_history = []
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        
        # Stop loss tracking variables
        self.purchase_price = 0  # Price at which shares were purchased
        self.highest_price_since_purchase = 0  # For trailing stop loss
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
    def predict_prices(self, current_data):
        """
        Predict prices for the next days using the model
        """
        if self.model is None:
            logging.error("Model not trained yet")
            # Return some default values to avoid crashing
            return np.ones(5) * current_data[0][-1][0]  # Use the last known price
        
        try:
            predictions = self.model.predict(current_data)
            
            # Check if predictions are valid
            if predictions is None or np.isscalar(predictions) or np.any(np.isnan(predictions)):
                logging.error(f"Invalid prediction result: {predictions}")
                # Return some default values
                return np.ones(5) * current_data[0][-1][0]  # Use the last known price
                
            # Inverse transform the predictions if they were scaled
            if self.price_scaler is not None:
                try:
                    # Ensure predictions is a 2D array
                    pred_reshaped = predictions.reshape(-1, predictions.shape[-1]) if len(predictions.shape) > 2 else predictions
                    pred_transformed = self.price_scaler.inverse_transform(pred_reshaped)
                    # Reshape back to original shape if needed
                    if len(predictions.shape) > 2:
                        pred_transformed = pred_transformed.reshape(predictions.shape)
                    return pred_transformed[0]
                except Exception as e:
                    logging.error(f"Error transforming predictions: {e}")
                    return np.ones(5) * current_data[0][-1][0]  # Use the last known price
                    
            return predictions[0]
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            # Return some default values
            return np.ones(5) * current_data[0][-1][0]  # Use the last known price
    
    # Improve trading decision logic to reduce losses
    def decide_action(self, current_price, predicted_price, available_capital, current_shares):
        """
        Decide trading action based on prediction and stop-loss criteria
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted stock price
            available_capital: Available cash
            current_shares: Current number of shares owned
        
        Returns:
            action: 'buy', 'sell', or 'hold'
            quantity: Number of shares to buy or sell
        """
        # Check stop-loss if we own shares
        if current_shares > 0:
            # Update highest price seen since purchase (for trailing stop)
            if current_price > self.highest_price_since_purchase:
                self.highest_price_since_purchase = current_price
            
            # Calculate price drop from purchase or from highest point if using trailing stop
            if self.trailing_stop_pct is not None:
                # Trailing stop: calculate drop from highest price seen
                price_drop_pct = (self.highest_price_since_purchase - current_price) / self.highest_price_since_purchase * 100
                stop_threshold = self.trailing_stop_pct
            else:
                # Regular stop loss: calculate drop from purchase price
                price_drop_pct = (self.purchase_price - current_price) / self.purchase_price * 100
                stop_threshold = self.stop_loss_pct
                
            # Trigger stop loss if price drops below threshold
            if price_drop_pct >= stop_threshold:
                logging.info(f"Stop-loss triggered: Price dropped {price_drop_pct:.2f}% from {'highest' if self.trailing_stop_pct else 'purchase'} price")
                return 'sell', current_shares  # Sell all shares
        
        # Standard trading logic if stop-loss not triggered
        # Calculate price change percentage for prediction
        price_change = (predicted_price - current_price) / current_price * 100
        
        # More conservative thresholds
        buy_threshold = 3.0
        sell_threshold = -3.0
        
        # Risk-adjusted position sizing
        max_position = min(0.5, self.risk_factor)
        
        if price_change > buy_threshold and available_capital > current_price:
            # Buy signal
            # Calculate the position size based on conviction
            conviction = min(price_change / 10, 1.0)
            position_size = available_capital * max_position * conviction
            
            # Ensure we don't spend more than available
            max_shares = int(position_size / (current_price * (1 + self.transaction_fee)))
            quantity = max(1, min(max_shares, int(available_capital / (current_price * (1 + self.transaction_fee)))))
            
            # Update purchase price for stop-loss calculation
            if current_shares == 0:  # First purchase
                self.purchase_price = current_price
                self.highest_price_since_purchase = current_price
            else:  # Additional purchase - calculate average price
                total_value = (self.purchase_price * current_shares) + (current_price * quantity)
                self.purchase_price = total_value / (current_shares + quantity)
                
            return 'buy', quantity
            
        elif price_change < sell_threshold and current_shares > 0:
            # Sell signal
            # Stronger conviction = sell more shares
            conviction = min(abs(price_change) / 10, 1.0)
            quantity = max(1, int(current_shares * conviction))
            
            return 'sell', quantity
        
        # Hold by default
        return 'hold', 0
    
    def execute_trade(self, action, amount, current_price, timestamp):
        """
        Execute a trade based on the decided action
        """
        if action == 'buy' and amount > 0:
            # Calculate shares to buy (considering transaction fee)
            shares_to_buy = (amount * (1 - self.transaction_fee)) / current_price
            if self.current_capital >= amount:
                self.shares_owned += shares_to_buy
                self.current_capital -= amount
                transaction_cost = amount * self.transaction_fee
                
                # Record transaction
                self.transaction_history.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'amount': amount,
                    'fee': transaction_cost,
                    'capital_after': self.current_capital,
                    'shares_after': self.shares_owned,
                    'portfolio_value': self.current_capital + (self.shares_owned * current_price)
                })
                logging.info(f"BUY: {shares_to_buy:.2f} shares at ${current_price:.2f}")
                
        elif action == 'sell' and self.shares_owned > 0 and amount > 0:
            # Ensure we don't sell more than we own
            shares_to_sell = min(amount, self.shares_owned)
            sale_amount = shares_to_sell * current_price
            self.current_capital += sale_amount * (1 - self.transaction_fee)
            self.shares_owned -= shares_to_sell
            transaction_cost = sale_amount * self.transaction_fee
            
            # Record transaction
            self.transaction_history.append({
                'timestamp': timestamp,
                'action': 'sell',
                'price': current_price,
                'shares': shares_to_sell,
                'amount': sale_amount,
                'fee': transaction_cost,
                'capital_after': self.current_capital,
                'shares_after': self.shares_owned,
                'portfolio_value': self.current_capital + (self.shares_owned * current_price)
            })
            logging.info(f"SELL: {shares_to_sell:.2f} shares at ${current_price:.2f}")
            
        else:
            # Record hold action
            self.transaction_history.append({
                'timestamp': timestamp,
                'action': 'hold',
                'price': current_price,
                'shares': 0,
                'amount': 0,
                'fee': 0,
                'capital_after': self.current_capital,
                'shares_after': self.shares_owned,
                'portfolio_value': self.current_capital + (self.shares_owned * current_price)
            })
            logging.info(f"HOLD: Current portfolio value: ${self.current_capital + (self.shares_owned * current_price):.2f}")
    
    def run_simulation(self, test_data, test_dates, feature_scaler, final_trade=True):
        """
        Run a trading simulation using test data
        
        Args:
            test_data: Processed test data for prediction
            test_dates: Dates corresponding to the test data
            feature_scaler: Scaler for feature transformation
            final_trade: Whether to liquidate all shares at the end
            
        Returns:
            final_value: Final portfolio value
            roi: Return on investment (%)
        """
        logging.info("Starting trading simulation...")
        
        # Ensure we have at least window_size + prediction_horizon days of data
        window_size = test_data.shape[1]
        
        # Calculate how many iterations we can safely do
        # We need to make sure i + window_size is less than the length of test_dates
        max_iterations = min(len(test_data) - 1, len(test_dates) - window_size)
        
        for i in range(max_iterations):
            # Get current and next date indices safely
            current_date_idx = min(i + window_size - 1, len(test_dates) - 1)
            next_date_idx = min(i + window_size, len(test_dates) - 1)
            
            # Ensure we don't go out of bounds
            if next_date_idx >= len(test_dates):
                logging.warning(f"Reached the end of available dates at iteration {i}. Stopping simulation.")
                break
                
            # Current date and price
            current_date = test_dates[current_date_idx]
            next_date = test_dates[next_date_idx]
            
            # Ensure we're within bounds for test_data as well
            if i >= len(test_data) or i+1 >= len(test_data):
                logging.warning(f"Reached the end of available test data at iteration {i}. Stopping simulation.")
                break
            
            current_price = test_data[i][-1][0]  # Last day in sequence, first feature (Close)
            next_price = test_data[i+1][-1][0]  # Using the first feature as Close price
            
            # Make prediction for next 5 days
            predicted_prices = self.predict_prices(np.array([test_data[i]]))
            
            # Decide action with all required parameters
            action, amount = self.decide_action(
                current_price=current_price,
                predicted_price=predicted_prices[0],
                available_capital=self.current_capital,
                current_shares=self.shares_owned
            )
            
            # Execute trade using the NEXT day's price (simulating real trading)
            self.execute_trade(action, amount, next_price, next_date)
            
            # Record portfolio value history
            self.portfolio_history.append({
                'date': next_date,
                'price': next_price,
                'capital': self.current_capital,
                'shares': self.shares_owned,
                'portfolio_value': self.current_capital + (self.shares_owned * next_price)
            })
        
        # Final liquidation if requested
        if final_trade and self.shares_owned > 0 and len(test_data) > 0:
            final_price = test_data[-1][-1][0]
            final_date = test_dates[-1] if len(test_dates) > 0 else datetime.now()
            self.execute_trade('sell', self.shares_owned, final_price, final_date)
        
        # Calculate final portfolio value and return
        final_value = self.current_capital
        roi = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        logging.info(f"Simulation complete. Final value: ${final_value:.2f}, ROI: {roi:.2f}%")
        self.save_results()
        
        return final_value, roi
    
    def save_results(self):
        """
        Save trading results to CSV and generate plots
        """
        # Save transaction history
        if self.transaction_history:
            transactions_df = pd.DataFrame(self.transaction_history)
            transactions_path = os.path.join(self.results_dir, 'transactions.csv')
            transactions_df.to_csv(transactions_path, index=False)
            logging.info(f"Transaction history saved to {transactions_path}")
        
        # Save portfolio history
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_path = os.path.join(self.results_dir, 'portfolio.csv')
            portfolio_df.to_csv(portfolio_path, index=False)
            logging.info(f"Portfolio history saved to {portfolio_path}")
            
            # Plot portfolio value over time
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'portfolio_value.png'))
            plt.close()
            
            # Plot transactions
            buy_dates = [t['timestamp'] for t in self.transaction_history if t['action'] == 'buy']
            buy_prices = [t['price'] for t in self.transaction_history if t['action'] == 'buy']
            
            sell_dates = [t['timestamp'] for t in self.transaction_history if t['action'] == 'sell']
            sell_prices = [t['price'] for t in self.transaction_history if t['action'] == 'sell']
            
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df['date'], portfolio_df['price'], label='Stock Price')
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy')
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell')
            plt.title('Trading Decisions')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'trading_decisions.png'))
            plt.close()
