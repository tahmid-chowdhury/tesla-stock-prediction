import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TradingAgent:
    def __init__(self, model, scaler, initial_balance=10000, transaction_fee=0.01):
        """
        Initialize trading agent
        
        Parameters:
        - model: Trained prediction model
        - scaler: Scaler used for data normalization
        - initial_balance: Starting capital in USD
        - transaction_fee: Fee percentage per transaction
        """
        self.model = model
        self.scaler = scaler
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares = 0
        self.transaction_fee = transaction_fee
        self.transaction_history = []
        
    def predict_price_movement(self, current_data):
        """Predict price movement using the model"""
        prediction = self.model.predict(current_data)
        return prediction[0][0]  # Get first prediction
        
    def calculate_risk_factor(self, data):
        """Calculate risk factor based on volatility and other metrics"""
        # Get volatility from the last row of data
        volatility = data[-1, -2]  # Assuming volatility is the second last column
        rsi = data[-1, 5]  # RSI at current position
        
        # Higher risk with high volatility or extreme RSI
        risk_factor = volatility * 2
        if rsi > 70 or rsi < 30:
            risk_factor *= 1.5
            
        # Cap risk factor between 0.1 and 0.5
        return min(max(risk_factor, 0.1), 0.5)
    
    def make_decision(self, current_data, current_price, next_day=None):
        """
        Make trading decision for the next day
        
        Parameters:
        - current_data: Current market data for prediction
        - current_price: Current stock price
        - next_day: Next trading day date
        
        Returns:
        - decision: "Buy", "Sell", or "Hold"
        - amount: Amount to buy/sell
        """
        # Get price prediction
        predicted_price = self.predict_price_movement(current_data)
        
        # Unscale the predicted price (assuming Close is the first column)
        price_min = self.scaler.data_min_[0]
        price_max = self.scaler.data_max_[0]
        predicted_price = predicted_price * (price_max - price_min) + price_min
        
        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Calculate risk factor
        risk_factor = self.calculate_risk_factor(current_data[0])
        
        # Decision logic
        decision = "Hold"
        amount = 0
        
        # Strong buy signal: significant positive price movement expected
        if price_change_pct > 0.02:
            decision = "Buy"
            # Calculate amount to buy based on balance and risk
            amount = self.balance * risk_factor
            if amount < 100:  # Minimum transaction amount
                decision = "Hold"
                amount = 0
        
        # Strong sell signal: significant negative price movement expected
        elif price_change_pct < -0.02 and self.shares > 0:
            decision = "Sell"
            # Sell shares based on predicted drop and risk
            amount = int(self.shares * min(1.0, abs(price_change_pct * 20)))
            if amount == 0 and self.shares > 0:
                amount = 1  # Sell at least 1 share
            
        # Format the order according to required format
        order = self._format_order(decision, amount, current_price, next_day)
        
        return decision, amount, order
    
    def execute_order(self, decision, amount, price):
        """Execute a trading order"""
        if decision == "Buy" and amount > 0:
            max_shares = int(amount / price)
            if max_shares > 0:
                cost = max_shares * price
                fee = cost * self.transaction_fee
                total_cost = cost + fee
                
                if total_cost <= self.balance:
                    self.balance -= total_cost
                    self.shares += max_shares
                    
                    transaction = {
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'action': 'Buy',
                        'shares': max_shares,
                        'price': price,
                        'cost': cost,
                        'fee': fee,
                        'balance': self.balance,
                        'total_shares': self.shares
                    }
                    self.transaction_history.append(transaction)
                    return True
            
        elif decision == "Sell" and amount > 0:
            sell_shares = min(amount, self.shares)
            if sell_shares > 0:
                revenue = sell_shares * price
                fee = revenue * self.transaction_fee
                net_revenue = revenue - fee
                
                self.balance += net_revenue
                self.shares -= sell_shares
                
                transaction = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'action': 'Sell',
                    'shares': sell_shares,
                    'price': price,
                    'revenue': revenue,
                    'fee': fee,
                    'balance': self.balance,
                    'total_shares': self.shares
                }
                self.transaction_history.append(transaction)
                return True
        
        return False
    
    def _format_order(self, decision, amount, current_price, date=None):
        """Format order based on simulation requirements"""
        date_str = date if date else datetime.now().strftime('%Y-%m-%d')
        
        if decision == "Buy":
            shares_to_buy = int(amount / current_price)
            if shares_to_buy > 0:
                return f"Buy: ${amount:.2f}"
            else:
                return "Hold: No transaction"
        
        elif decision == "Sell":
            if amount > 0:
                return f"Sell: {amount} shares"
            else:
                return "Hold: No transaction"
        
        else:
            return "Hold: No transaction"
    
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        stock_value = self.shares * current_price
        return self.balance + stock_value
    
    def get_transaction_history(self):
        """Return transaction history as DataFrame"""
        return pd.DataFrame(self.transaction_history)
