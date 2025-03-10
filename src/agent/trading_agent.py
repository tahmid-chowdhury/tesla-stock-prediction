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
        self.day_counter = 0  # Add counter to track days for alternating strategy
        self.previous_predictions = []  # Store recent predictions to detect trends
        self.ytd_metrics = {}  # Store YTD metrics
        
    def predict_price_movement(self, current_data):
        """Predict price movement using the model"""
        prediction = self.model.predict(current_data)
        return prediction[0][0]  # Get first prediction
        
    def predict_price_direction(self, current_data):
        """Predict price direction (up=1, down=0) using the model"""
        if hasattr(self.model, 'predict_direction'):
            direction = self.model.predict_direction(current_data)
            return direction[0]  # Get first prediction
        else:
            # Fall back to deriving direction from price prediction
            prediction = self.predict_price_movement(current_data)
            
            # Unscale the predicted price
            price_min = self.scaler.data_min_[0]
            price_max = self.scaler.data_max_[0]
            predicted_price = prediction * (price_max - price_min) + price_min
            
            # Get current price from the input data
            current_price = current_data[0, -1, 0]  # Last time step, first feature
            current_price = current_price * (price_max - price_min) + price_min
            
            return 1 if predicted_price > current_price else 0
    
    def calculate_risk_factor(self, data, ytd_data=None):
        """Calculate risk factor based on volatility and other metrics, emphasizing YTD data"""
        # Get volatility and RSI from current data
        volatility = data[-1, 10]  # Assuming volatility is at index 10
        rsi = data[-1, 5]  # RSI at index 5
        
        # Base risk factor calculation
        risk_factor = volatility * 3
        
        # If YTD data is provided, adjust risk based on YTD trends
        if ytd_data is not None:
            # Calculate YTD volatility trend (increasing or decreasing)
            ytd_vol_trend = ytd_data.get('volatility_trend', 0)
            
            # Calculate YTD momentum
            ytd_momentum = ytd_data.get('price_momentum', 0)
            
            # Increase risk appetite if YTD momentum is positive and volatility trend is decreasing
            if ytd_momentum > 0 and ytd_vol_trend < 0:
                risk_factor *= 1.2
            
            # Decrease risk appetite if YTD momentum is negative or volatility trend is increasing
            elif ytd_momentum < 0 or ytd_vol_trend > 0:
                risk_factor *= 0.8
        
        # Adjust based on RSI (oversold/overbought conditions)
        if rsi > 60 or rsi < 40:  
            risk_factor *= 1.75
            
        # Cap risk factor
        return min(max(risk_factor, 0.2), 0.6)
    
    def analyze_ytd_data(self, current_date, available_data):
        """
        Analyze Year-to-Date data to extract relevant trading metrics
        
        Parameters:
        - current_date: Current trading date
        - available_data: DataFrame with complete available data
        
        Returns:
        - Dictionary with YTD metrics
        """
        # Extract YTD data (from January 1st of current year to current date)
        year_start = datetime(current_date.year, 1, 1)
        ytd_mask = (available_data['Date'] >= year_start) & (available_data['Date'] <= current_date)
        ytd_data = available_data[ytd_mask].copy()
        
        if len(ytd_data) < 5:  # Need minimum data for analysis
            return {}
        
        # Calculate YTD metrics
        metrics = {}
        
        # YTD price movement
        start_price = ytd_data.iloc[0]['Close']
        current_price = ytd_data.iloc[-1]['Close']
        ytd_return = (current_price - start_price) / start_price
        
        # YTD volatility trend (last 30 days vs. first 30 days)
        if len(ytd_data) >= 60:
            first_period = ytd_data.iloc[:30]
            last_period = ytd_data.iloc[-30:]
            first_vol = first_period['Volatility'].mean()
            last_vol = last_period['Volatility'].mean()
            vol_trend = last_vol - first_vol
        else:
            vol_trend = 0
        
        # YTD moving average trends
        short_ma = ytd_data['MA5'].iloc[-1]
        long_ma = ytd_data['MA20'].iloc[-1]
        ma_diff = (short_ma - long_ma) / long_ma
        
        # YTD price momentum (rate of change over last 20 days)
        if len(ytd_data) >= 20:
            momentum = (ytd_data['Close'].iloc[-1] - ytd_data['Close'].iloc[-20]) / ytd_data['Close'].iloc[-20]
        else:
            momentum = 0
        
        # Store metrics
        metrics['ytd_return'] = ytd_return
        metrics['volatility_trend'] = vol_trend
        metrics['ma_trend'] = ma_diff
        metrics['price_momentum'] = momentum
        metrics['rsi'] = ytd_data['RSI'].iloc[-1]
        metrics['avg_volume'] = ytd_data['Volume'].mean()
        metrics['recent_volume'] = ytd_data['Volume'].iloc[-5:].mean()
        metrics['volume_trend'] = metrics['recent_volume'] / metrics['avg_volume'] - 1
        
        return metrics
    
    def make_decision(self, current_data, current_price, next_day=None, available_data=None):
        """
        Make trading decision emphasizing YTD data and direction prediction
        
        Parameters:
        - current_data: Current market data for prediction
        - current_price: Current stock price
        - next_day: Next trading day date
        - available_data: Optional complete dataset for YTD analysis
        
        Returns:
        - decision: "Buy", "Sell", or "Hold"
        - amount: Amount to buy/sell
        """
        # Get price prediction from model
        predicted_price = self.predict_price_movement(current_data)
        
        # Store prediction history
        self.previous_predictions.append(predicted_price)
        if len(self.previous_predictions) > 5:
            self.previous_predictions.pop(0)
            
        # Unscale the predicted price
        price_min = self.scaler.data_min_[0]
        price_max = self.scaler.data_max_[0]
        predicted_price = predicted_price * (price_max - price_min) + price_min
        
        # Calculate price change percentage
        price_change_pct = (predicted_price - current_price) / current_price
        
        # Get direction prediction (1=up, 0=down)
        direction_prediction = self.predict_price_direction(current_data)
        direction_confidence = 0.7  # Default confidence, replace with probability if available
        
        # YTD analysis if data is available
        ytd_metrics = {}
        if available_data is not None and next_day is not None:
            ytd_metrics = self.analyze_ytd_data(next_day, available_data)
            self.ytd_metrics = ytd_metrics  # Store for later use
        
        # Calculate risk factor with YTD emphasis
        risk_factor = self.calculate_risk_factor(current_data[0], ytd_metrics)
        
        # Calculate momentum from recent predictions
        prediction_momentum = 0
        if len(self.previous_predictions) >= 3:
            prediction_momentum = sum([1 if self.previous_predictions[i] > self.previous_predictions[i-1] else -1 
                                    for i in range(1, len(self.previous_predictions))])
        
        # Decision logic with YTD emphasis and direction prediction
        decision = "Hold"
        amount = 0
        
        # Strongly weight direction prediction in buy/sell decisions
        if ytd_metrics:
            # Strong YTD buy signals
            ytd_buy_signal = (
                (ytd_metrics.get('price_momentum', 0) > 0.02) or  # Strong positive momentum
                (ytd_metrics.get('ma_trend', 0) > 0.01) or        # Short MA above long MA
                (ytd_metrics.get('volume_trend', 0) > 0.1)         # Increasing volume
            )
            
            # YTD cautionary signals
            ytd_caution = (
                (ytd_metrics.get('rsi', 50) > 70) or              # Overbought
                (ytd_metrics.get('volatility_trend', 0) > 0.1)    # Increasing volatility
            )
            
            # If YTD signals suggest buying AND direction prediction is up
            if (ytd_buy_signal or direction_prediction == 1) and price_change_pct > 0 and not ytd_caution:
                decision = "Buy"
                # Calculate position size based on YTD metrics, risk and direction confidence
                confidence = min(1.0, abs(ytd_metrics.get('price_momentum', 0) * 10) + direction_confidence)
                # Slightly lower threshold to trigger more buys
                amount = self.balance * risk_factor * confidence
                # Ensure minimum transaction
                if amount < 50:
                    amount = 50
            
            # Sell signals with YTD emphasis
            ytd_sell_signal = (
                (ytd_metrics.get('price_momentum', 0) < -0.02) or  # Negative momentum
                (ytd_metrics.get('ma_trend', 0) < -0.01) or        # Short MA below long MA
                (ytd_metrics.get('rsi', 50) > 75)                  # Overbought
            )
            
            if (ytd_sell_signal or direction_prediction == 0) and self.shares > 0 and (price_change_pct < 0 or ytd_metrics.get('rsi', 50) > 70):
                decision = "Sell"
                # Sell ratio based on YTD metrics
                sell_ratio = min(1.0, abs(ytd_metrics.get('price_momentum', 0) * 15))
                amount = max(1, int(self.shares * sell_ratio))
        else:
            # Fallback to basic signals when YTD data isn't available
            # More aggressively use direction prediction here
            if direction_prediction == 1 or (prediction_momentum > 1 and price_change_pct > 0):
                decision = "Buy"
                confidence = 0.5 if direction_prediction == 1 else 0.3
                amount = self.balance * risk_factor * confidence * (1 + abs(price_change_pct) * 5)
                # Lower minimum to trigger more buys
                if amount < 20:
                    amount = 20
            
            elif (direction_prediction == 0 or prediction_momentum < -1) and self.shares > 0:
                decision = "Sell"
                sell_ratio = 0.5 if direction_prediction == 0 else 0.3
                amount = max(1, int(self.shares * sell_ratio))
        
        # Format the order
        order = self._format_order(decision, amount, current_price, next_day)
        
        # Debug logging to diagnose trading issues
        print(f"Decision: {decision}, Direction: {'UP' if direction_prediction == 1 else 'DOWN'}, "
              f"Price change: {price_change_pct:.2%}, Amount: {amount:.2f}")
        
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
