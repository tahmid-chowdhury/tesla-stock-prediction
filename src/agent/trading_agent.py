import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingAgent:
    def __init__(self, model, price_scaler, initial_capital=10000, transaction_fee=0.01, risk_factor=0.3,
                 stop_loss_pct=5.0, trailing_stop_pct=None, max_trades_per_day=2,
                 max_drawdown_pct=20.0, volatility_scaling=True, window_size=30, prediction_horizon=5):
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
            max_trades_per_day: Maximum number of trades allowed per day
            max_drawdown_pct: Maximum allowable portfolio drawdown before halting trading
            volatility_scaling: Whether to adjust stop-loss based on market volatility
            window_size: Size of sliding window in days (default: 30)
            prediction_horizon: Days to predict ahead (default: 5)
        """
        self.model = model
        self.price_scaler = price_scaler
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.risk_factor = risk_factor
        self.base_stop_loss_pct = stop_loss_pct
        self.stop_loss_pct = stop_loss_pct  # This can be adjusted dynamically
        self.trailing_stop_pct = trailing_stop_pct
        self.shares_owned = 0
        self.transaction_history = []
        self.portfolio_history = []
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        
        # Window and prediction configuration
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        # Prediction tracking for evaluation
        self.prediction_history = []
        
        # Stop loss tracking variables
        self.purchase_price = 0  # Price at which shares were purchased
        self.highest_price_since_purchase = 0  # For trailing stop loss
        
        # Trading limits
        self.max_trades_per_day = max_trades_per_day
        self.max_drawdown_pct = max_drawdown_pct
        self.volatility_scaling = volatility_scaling
        self.trades_today = {}  # Dict to track trades per day
        self.peak_portfolio_value = initial_capital
        self.trading_paused = False
        self.trend_buffer = []  # Store recent price movements to detect trends
        self.consecutive_losses = 0  # Track consecutive losing trades
        
        # Enhanced adaptive trading parameters
        self.recent_trades = []  # Track recent trade performance
        self.success_rate = 0.5  # Initial success rate estimate (neutral)
        self.market_regime = 'unknown'  # Current market regime: bull, bear, sideways, unknown
        self.regime_factor = 1.0  # Adjustment factor based on market regime
        self.threshold_history = []  # Track historical threshold adjustments
        self.max_threshold_history = 30  # Maximum number of threshold records to keep
        self.prediction_accuracy = 0.0  # Model accuracy tracker
        
        # Multiple timeframe trend tracking
        self.short_term_trend = 0  # 1-5 day trend
        self.medium_term_trend = 0  # 5-15 day trend
        self.long_term_trend = 0  # 15-30 day trend
        
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
    
    def calculate_dynamic_stop_loss(self, current_price, price_history=None):
        """
        Calculate a dynamic stop loss based on recent price volatility
        
        Args:
            current_price: Current price of the asset
            price_history: List of recent prices (optional)
            
        Returns:
            Dynamic stop loss percentage
        """
        # Default to base stop loss
        stop_loss = self.base_stop_loss_pct
        
        # If volatility scaling is enabled and we have price history
        if self.volatility_scaling and price_history is not None and len(price_history) >= 10:
            # Calculate recent volatility (standard deviation of returns)
            returns = np.diff(price_history) / price_history[:-1]
            volatility = np.std(returns) * 100  # Convert to percentage
            
            # Scale stop loss based on volatility (higher volatility = higher stop loss)
            # with minimum of base stop loss and maximum of 2x base stop loss
            stop_loss = max(self.base_stop_loss_pct, 
                           min(self.base_stop_loss_pct * 2, 
                               self.base_stop_loss_pct * (1 + volatility)))
            
            logging.info(f"Dynamic stop-loss adjusted to {stop_loss:.2f}% based on volatility of {volatility:.2f}%")
        
        return stop_loss
    
    def calculate_volatility_metrics(self, price_history):
        """
        Calculate multiple volatility metrics from price history
        
        Args:
            price_history: List of recent prices
            
        Returns:
            Dictionary of volatility metrics
        """
        if not price_history or len(price_history) < 5:
            return {'recent_volatility': 0.05, 'volatility_trend': 0, 'normalized_volatility': 1.0}
            
        # Calculate returns
        prices = np.array(price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Recent volatility (standard deviation of returns)
        recent_volatility = np.std(returns) * 100
        
        # Volatility trend (is volatility increasing or decreasing?)
        if len(returns) >= 10:
            vol_first_half = np.std(returns[:len(returns)//2]) * 100
            vol_second_half = np.std(returns[len(returns)//2:]) * 100
            volatility_trend = vol_second_half - vol_first_half
        else:
            volatility_trend = 0
            
        # Normalized volatility (compared to historical average)
        # Use a default if we don't have enough history
        avg_volatility = 2.0  # Default for Tesla stock
        if len(returns) >= 20:
            avg_volatility = np.mean([np.std(returns[i:i+5])*100 for i in range(0, len(returns)-5, 5)])
        
        normalized_volatility = max(0.5, min(3.0, recent_volatility / (avg_volatility or 1.0)))
        
        return {
            'recent_volatility': recent_volatility,
            'volatility_trend': volatility_trend,
            'normalized_volatility': normalized_volatility
        }
    
    def detect_market_regime(self, price_history):
        """
        Detect the current market regime (bull, bear, sideways)
        
        Args:
            price_history: List of recent prices
            
        Returns:
            Market regime and confidence
        """
        if not price_history or len(price_history) < 10:
            return {'regime': 'unknown', 'confidence': 0, 'factor': 1.0}
        
        prices = np.array(price_history)
        
        # Calculate trends at different timeframes
        if len(prices) >= 30:
            # Short-term trend (5-day)
            self.short_term_trend = (prices[-1] / prices[-min(5, len(prices))] - 1) * 100
            
            # Medium-term trend (15-day)
            self.medium_term_trend = (prices[-1] / prices[-min(15, len(prices))] - 1) * 100
            
            # Long-term trend (30-day)
            self.long_term_trend = (prices[-1] / prices[-min(30, len(prices))] - 1) * 100
            
            # Simple moving averages
            sma5 = np.mean(prices[-5:])
            sma15 = np.mean(prices[-15:])
            sma30 = np.mean(prices[-30:])
        else:
            # Shorter timeframes if we don't have enough history
            split = len(prices) // 3
            if split > 0:
                self.short_term_trend = (prices[-1] / prices[-min(split, len(prices))] - 1) * 100
                self.medium_term_trend = (prices[-1] / prices[-min(2*split, len(prices))] - 1) * 100
                self.long_term_trend = (prices[-1] / prices[0] - 1) * 100
                
                # Simple moving averages with shorter windows
                sma5 = np.mean(prices[-min(split, len(prices)):])
                sma15 = np.mean(prices[-min(2*split, len(prices)):])
                sma30 = np.mean(prices)
            else:
                return {'regime': 'unknown', 'confidence': 0, 'factor': 1.0}
        
        # Determine regime based on moving averages and trends
        regime = 'unknown'
        confidence = 0
        factor = 1.0
        
        # Bull market indicators
        bull_signals = 0
        if self.short_term_trend > 0: bull_signals += 1
        if self.medium_term_trend > 0: bull_signals += 1
        if self.long_term_trend > 0: bull_signals += 1
        if sma5 > sma15: bull_signals += 1
        if sma15 > sma30: bull_signals += 1
        
        # Bear market indicators
        bear_signals = 0
        if self.short_term_trend < 0: bear_signals += 1
        if self.medium_term_trend < 0: bear_signals += 1
        if self.long_term_trend < 0: bear_signals += 1
        if sma5 < sma15: bear_signals += 1
        if sma15 < sma30: bear_signals += 1
        
        # Determine regime
        if bull_signals >= 4:
            regime = 'bull'
            confidence = bull_signals / 5
            factor = 0.8  # More aggressive in bull markets
        elif bear_signals >= 4:
            regime = 'bear'
            confidence = bear_signals / 5
            factor = 1.3  # More conservative in bear markets
        else:
            # Check for sideways market (low volatility, mixed signals)
            volatility = np.std(np.diff(prices) / prices[:-1]) * 100
            if volatility < 2.0:  # Low volatility threshold
                regime = 'sideways'
                confidence = 1 - (volatility / 2.0)
                factor = 1.1  # Slightly more conservative in sideways markets
            else:
                # Mixed or transitioning market
                regime = 'mixed'
                confidence = 0.5
                factor = 1.2  # Somewhat conservative in mixed markets
        
        return {'regime': regime, 'confidence': confidence, 'factor': factor}
    
    def update_performance_metrics(self):
        """
        Update trading performance metrics based on recent trades
        """
        # Need at least a few trades to calculate meaningful metrics
        if len(self.transaction_history) < 3:
            return
            
        # Get last 10 trades (or fewer if we don't have that many)
        recent_trades = [t for t in self.transaction_history[-10:] 
                         if t['action'] in ['buy', 'sell']]
        
        if not recent_trades:
            return
            
        # Calculate success rate
        gains = 0
        for i in range(1, len(recent_trades), 2):
            if i < len(recent_trades):
                # If we have a buy followed by a sell
                if recent_trades[i-1]['action'] == 'buy' and recent_trades[i]['action'] == 'sell':
                    buy_price = recent_trades[i-1]['price']
                    sell_price = recent_trades[i]['price']
                    if sell_price > buy_price:
                        gains += 1
        
        # Calculate success rate (percentage of profitable buy-sell pairs)
        pairs = len(recent_trades) // 2
        self.success_rate = gains / pairs if pairs > 0 else 0.5
        
        # Update prediction accuracy if we have prediction history
        if self.prediction_history:
            pred_df = pd.DataFrame(self.prediction_history)
            if not pred_df.empty and len(pred_df) > 1:
                # Get actual next day prices by shifting
                pred_df['next_day_actual'] = pred_df['actual_price'].shift(-1)
                
                # Drop the last row which has NaN in next_day_actual
                pred_df = pred_df.dropna()
                
                if not pred_df.empty:
                    # Calculate directional accuracy
                    correct_direction = ((pred_df['predicted_next_day'] > pred_df['actual_price']) == 
                                        (pred_df['next_day_actual'] > pred_df['actual_price']))
                    self.prediction_accuracy = correct_direction.mean()
    
    def calculate_adaptive_thresholds(self, price_history, current_price, predicted_price):
        """
        Calculate adaptive trading thresholds based on multiple factors
        
        Args:
            price_history: List of recent prices
            current_price: Current stock price
            predicted_price: Predicted next day price
            
        Returns:
            Dictionary with buy and sell thresholds
        """
        # Update performance metrics first
        self.update_performance_metrics()
        
        # Default base thresholds
        base_buy_threshold = 5.0
        base_sell_threshold = -5.0
        
        # 1. Adjust for volatility
        vol_metrics = self.calculate_volatility_metrics(price_history)
        volatility_factor = vol_metrics['normalized_volatility']
        
        # Scale thresholds with volatility (higher volatility = higher thresholds)
        vol_adjusted_buy = base_buy_threshold * volatility_factor
        vol_adjusted_sell = base_sell_threshold * volatility_factor
        
        # 2. Adjust for market regime
        regime_data = self.detect_market_regime(price_history)
        self.market_regime = regime_data['regime']
        self.regime_factor = regime_data['factor']
        
        # Apply regime factor
        regime_adjusted_buy = vol_adjusted_buy * self.regime_factor
        regime_adjusted_sell = vol_adjusted_sell * self.regime_factor
        
        # 3. Adjust for trading performance
        # If we're doing well, be more aggressive. If not, be more conservative.
        performance_factor = 1.0 - (self.success_rate - 0.5)  # Map 0-1 success to 1.5-0.5 factor
        performance_factor = max(0.7, min(1.3, performance_factor))
        
        perf_adjusted_buy = regime_adjusted_buy * performance_factor
        perf_adjusted_sell = regime_adjusted_sell * performance_factor
        
        # 4. Adjust for prediction accuracy
        # If model has been accurate, trust it more (lower thresholds)
        accuracy_factor = 1.0
        if self.prediction_accuracy > 0:
            accuracy_factor = 1.0 - (self.prediction_accuracy - 0.5)  # Map 0.5-1.0 accuracy to 1.0-0.5 factor
            accuracy_factor = max(0.7, min(1.3, accuracy_factor))
        
        accuracy_adjusted_buy = perf_adjusted_buy * accuracy_factor
        accuracy_adjusted_sell = perf_adjusted_sell * accuracy_factor
        
        # 5. Adjust based on multiple timeframe trends
        trend_factor = 1.0
        
        # If all trends agree, strengthen the signal
        if (self.short_term_trend > 0 and self.medium_term_trend > 0 and self.long_term_trend > 0):
            # Strong bullish trend - lower buy threshold, raise sell threshold
            trend_factor = 0.8
        elif (self.short_term_trend < 0 and self.medium_term_trend < 0 and self.long_term_trend < 0):
            # Strong bearish trend - raise buy threshold, lower sell threshold
            trend_factor = 1.2
        
        final_buy_threshold = accuracy_adjusted_buy * trend_factor
        final_sell_threshold = accuracy_adjusted_sell * trend_factor
        
        # Cap thresholds within reasonable bounds
        final_buy_threshold = max(2.0, min(10.0, final_buy_threshold))
        final_sell_threshold = min(-2.0, max(-10.0, final_sell_threshold))
        
        # Store threshold for history
        self.threshold_history.append({
            'timestamp': datetime.now(),
            'buy_threshold': final_buy_threshold,
            'sell_threshold': final_sell_threshold,
            'volatility': vol_metrics['recent_volatility'],
            'market_regime': self.market_regime,
            'success_rate': self.success_rate,
            'prediction_accuracy': self.prediction_accuracy
        })
        
        # Keep history to maximum size
        if len(self.threshold_history) > self.max_threshold_history:
            self.threshold_history = self.threshold_history[-self.max_threshold_history:]
        
        logging.info(f"Adaptive thresholds: Buy {final_buy_threshold:.2f}%, Sell {final_sell_threshold:.2f}% " +
                    f"[Regime: {self.market_regime}, Volatility: {vol_metrics['recent_volatility']:.2f}%, " +
                    f"Success rate: {self.success_rate:.2f}]")
        
        return {
            'buy_threshold': final_buy_threshold,
            'sell_threshold': final_sell_threshold
        }
    
    def _increment_trade_count(self):
        """
        Increment the trade count for today
        Returns True if trade is allowed, False if max trades reached
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date not in self.trades_today:
            self.trades_today[current_date] = 0
            
        if self.trades_today[current_date] >= self.max_trades_per_day:
            return False
            
        self.trades_today[current_date] += 1
        return True
    
    def analyze_technical_indicators(self, price_history):
        """
        Analyze technical indicators to generate trading signals
        
        Args:
            price_history: List of recent prices
            
        Returns:
            Dictionary containing technical signals
        """
        if not price_history or len(price_history) < 30:
            return {'signal': 'neutral', 'strength': 0}
        
        prices = np.array(price_history)
        
        # Calculate moving averages
        sma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        sma10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        # Calculate exponential moving averages (EMA)
        def calculate_ema(data, span):
            if len(data) >= span:
                alpha = 2 / (span + 1)
                ema = data[0]
                for i in range(1, len(data)):
                    ema = alpha * data[i] + (1 - alpha) * ema
                return ema
            return data[-1]
        
        ema12 = calculate_ema(prices[-min(50, len(prices)):], 12)
        ema26 = calculate_ema(prices[-min(50, len(prices)):], 26)
        
        # Calculate MACD
        macd = ema12 - ema26
        
        # Calculate RSI if enough data points
        rsi = 50  # Default neutral value
        if len(prices) >= 14:
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            # Average gains and losses over 14 periods
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            elif avg_gain != 0:
                rsi = 100
        
        # Calculate Bollinger Bands
        if len(prices) >= 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            upper_band = sma20 + (2 * std20)
            lower_band = sma20 - (2 * std20)
            
            # Price position within bands (0-1 scale)
            if upper_band != lower_band:
                bb_position = (prices[-1] - lower_band) / (upper_band - lower_band)
            else:
                bb_position = 0.5
        else:
            bb_position = 0.5
        
        # Accumulate technical buy/sell signals
        buy_signals = 0
        sell_signals = 0
        signal_count = 0
        
        # Moving average crossovers
        if sma5 > sma20:
            buy_signals += 1
        else:
            sell_signals += 1
        signal_count += 1
        
        if sma10 > sma50:
            buy_signals += 1
        else:
            sell_signals += 1
        signal_count += 1
        
        # MACD signal
        if macd > 0:
            buy_signals += 1
        else:
            sell_signals += 1
        signal_count += 1
        
        # RSI signals
        if rsi < 30:  # Oversold
            buy_signals += 1.5  # Stronger signal
        elif rsi > 70:  # Overbought
            sell_signals += 1.5  # Stronger signal
        signal_count += 1.5
        
        # Bollinger Band signals
        if bb_position < 0.2:  # Price near lower band (potential buy)
            buy_signals += 1
        elif bb_position > 0.8:  # Price near upper band (potential sell)
            sell_signals += 1
        signal_count += 1
        
        # Current price vs moving averages
        current_price = prices[-1]
        if current_price > sma20:
            buy_signals += 0.5
        else:
            sell_signals += 0.5
        signal_count += 0.5
        
        # Calculate overall signal strength (-1 to 1 scale)
        if signal_count > 0:
            buy_strength = buy_signals / signal_count
            sell_strength = sell_signals / signal_count
            overall_strength = buy_strength - sell_strength
        else:
            overall_strength = 0
        
        # Determine signal
        if overall_strength > 0.3:
            signal = 'buy'
        elif overall_strength < -0.3:
            signal = 'sell'
        else:
            signal = 'neutral'
        
        logging.info(f"Technical analysis: Signal={signal}, Strength={overall_strength:.2f}, " +
                    f"SMA5={sma5:.2f}, SMA20={sma20:.2f}, RSI={rsi:.2f}")
        
        return {
            'signal': signal,
            'strength': abs(overall_strength),
            'moving_averages': {
                'sma5': sma5,
                'sma10': sma10,
                'sma20': sma20,
                'sma50': sma50
            },
            'indicators': {
                'macd': macd,
                'rsi': rsi,
                'bb_position': bb_position
            }
        }
    
    def decide_action(self, current_price, predicted_price, available_capital, current_shares, price_history=None):
        """
        Decide trading action based on prediction and stop-loss criteria
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted stock price
            available_capital: Available cash
            current_shares: Current number of shares owned
            price_history: Recent price history for trend analysis
            
        Returns:
            action: 'buy', 'sell', or 'hold'
            quantity: Number of shares to buy or sell
        """
        # Check if trading is paused due to excessive drawdown
        if self.trading_paused:
            logging.info("Trading is paused due to exceeding maximum drawdown")
            return 'hold', 0
            
        # Check if we've reached the maximum number of trades for today
        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date in self.trades_today and self.trades_today[current_date] >= self.max_trades_per_day:
            logging.info(f"Maximum trades for today ({self.max_trades_per_day}) reached")
            return 'hold', 0
        
        # Update trend buffer if we have history
        if price_history is not None and len(price_history) > 1:
            # Ensure we're using at most the last 30 days (window_size) of data
            self.trend_buffer = price_history[-self.window_size:] if len(price_history) >= self.window_size else price_history
            
            # Update stop-loss percentage based on volatility
            if self.volatility_scaling:
                self.stop_loss_pct = self.calculate_dynamic_stop_loss(current_price, self.trend_buffer)
        
        # Check for excessive drawdown
        current_portfolio_value = available_capital + (current_shares * current_price)
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        drawdown_pct = ((self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value) * 100
        if drawdown_pct >= self.max_drawdown_pct:
            logging.warning(f"Maximum drawdown of {drawdown_pct:.2f}% reached. Trading paused.")
            self.trading_paused = True
            
            # If we own shares, sell them to prevent further losses
            if current_shares > 0:
                logging.info(f"Emergency sell triggered due to maximum drawdown")
                return 'sell', current_shares
            return 'hold', 0
            
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
                
                # Add to today's trade count
                self._increment_trade_count()
                
                # Sell only part of position if consecutive losses are detected
                if self.consecutive_losses > 2:
                    # Sell only 50% of position to reduce risk of selling at bottom
                    quantity = max(1, int(current_shares * 0.5))
                    logging.info(f"Partial stop-loss sell due to multiple consecutive losses")
                    return 'sell', quantity
                else:
                    return 'sell', current_shares  # Sell all shares
        
        # Analyze technical indicators
        tech_signals = self.analyze_technical_indicators(price_history)
        
        # Standard trading logic if stop-loss not triggered
        # Calculate price change percentage for prediction
        price_change = (predicted_price - current_price) / current_price * 100
        
        # Get adaptive thresholds based on current market conditions
        thresholds = self.calculate_adaptive_thresholds(price_history, current_price, predicted_price)
        buy_threshold = thresholds['buy_threshold']
        sell_threshold = thresholds['sell_threshold']
        
        # Adjust thresholds based on technical signals
        if tech_signals['signal'] == 'buy':
            # Lower buy threshold, raise sell threshold
            adjustment_factor = 1 - (0.3 * tech_signals['strength'])
            buy_threshold *= adjustment_factor
            sell_threshold *= adjustment_factor
            logging.info(f"Technical buy signal detected. Adjusted thresholds: Buy={buy_threshold:.2f}, Sell={sell_threshold:.2f}")
        elif tech_signals['signal'] == 'sell':
            # Raise buy threshold, lower sell threshold
            adjustment_factor = 1 + (0.3 * tech_signals['strength'])
            buy_threshold *= adjustment_factor
            sell_threshold *= adjustment_factor
            logging.info(f"Technical sell signal detected. Adjusted thresholds: Buy={buy_threshold:.2f}, Sell={sell_threshold:.2f}")
        
        # Risk-adjusted position sizing
        max_position = min(0.5, self.risk_factor)
        
        # If we've had consecutive losses, reduce position size
        if self.consecutive_losses > 0:
            max_position = max_position * (1 - min(self.consecutive_losses * 0.1, 0.5))
        
        if price_change > buy_threshold and available_capital > current_price:
            # Buy logic
            # Strengthen or weaken conviction based on technical signals
            if tech_signals['signal'] == 'buy':
                conviction_modifier = 1 + (0.5 * tech_signals['strength'])
            elif tech_signals['signal'] == 'sell':
                conviction_modifier = 1 - (0.5 * tech_signals['strength'])
            else:
                conviction_modifier = 1.0
            
            # Check trade count limit
            if self._increment_trade_count():
                # Buy signal
                # Calculate the position size based on conviction and adjusted for risk
                conviction = min(price_change / buy_threshold, 1.5) * conviction_modifier
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
            # Sell logic
            # Strengthen or weaken conviction based on technical signals
            if tech_signals['signal'] == 'sell':
                conviction_modifier = 1 + (0.5 * tech_signals['strength'])
            elif tech_signals['signal'] == 'buy':
                conviction_modifier = 1 - (0.5 * tech_signals['strength'])
            else:
                conviction_modifier = 1.0
            
            # Check trade count limit
            if self._increment_trade_count():
                # Sell signal
                # Stronger conviction = sell more shares
                conviction = min(abs(price_change / sell_threshold), 1.5) * conviction_modifier
                quantity = max(1, min(current_shares, int(current_shares * conviction)))
                
                return 'sell', quantity
        
        # Hold by default
        return 'hold', 0
    
    def execute_trade(self, action, amount, current_price, timestamp):
        """
        Execute a trade based on the decided action
        """
        initial_portfolio_value = self.current_capital + (self.shares_owned * current_price)
        
        if action == 'buy' and amount > 0:
            # Calculate shares to buy (considering transaction fee)
            cost = amount * current_price
            transaction_cost = cost * self.transaction_fee
            actual_cost = cost + transaction_cost
            
            if self.current_capital >= actual_cost:
                shares_to_buy = amount
                self.shares_owned += shares_to_buy
                self.current_capital -= actual_cost
                
                # Record transaction
                self.transaction_history.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'amount': actual_cost,
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
            transaction_cost = sale_amount * self.transaction_fee
            net_sale_amount = sale_amount - transaction_cost
            
            self.current_capital += net_sale_amount
            self.shares_owned -= shares_to_sell
            
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
        
        # Track consecutive losses
        if len(self.transaction_history) >= 2 and action in ['buy', 'sell']:
            final_portfolio_value = self.current_capital + (self.shares_owned * current_price)
            if final_portfolio_value < initial_portfolio_value:
                self.consecutive_losses += 1
                logging.info(f"Trade resulted in loss. Consecutive losses: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0
    
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
        
        # Track price history for volatility and trend analysis
        price_history = []
        
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
            
            # Update price history
            price_history.append(current_price)
            
            # Make prediction for next 5 days
            predicted_prices = self.predict_prices(np.array([test_data[i]]))
            
            # Track predictions for evaluation
            self.prediction_history.append({
                'date': current_date,
                'actual_price': current_price,
                'predicted_next_day': predicted_prices[0],
                'predicted_5_day': predicted_prices[-1] if len(predicted_prices) >= 5 else predicted_prices[-1],
            })
            
            # Decide action with all required parameters
            action, amount = self.decide_action(
                current_price=current_price,
                predicted_price=predicted_prices[0],
                available_capital=self.current_capital,
                current_shares=self.shares_owned,
                price_history=price_history
            )
            
            # Execute trade using the NEXT day's price (simulating real trading)
            self.execute_trade(action, amount, next_price, next_date)
            
            # Record portfolio value history
            portfolio_value = self.current_capital + (self.shares_owned * next_price)
            self.portfolio_history.append({
                'date': next_date,
                'price': next_price,
                'capital': self.current_capital,
                'shares': self.shares_owned,
                'portfolio_value': portfolio_value
            })
            
            # Update peak portfolio value
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value
        
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
        try:
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
                
                # Save prediction performance
                if self.prediction_history:
                    predictions_df = pd.DataFrame(self.prediction_history)
                    predictions_path = os.path.join(self.results_dir, 'predictions.csv') 
                    predictions_df.to_csv(predictions_path, index=False)
                    logging.info(f"Prediction history saved to {predictions_path}")
                
                # Plot transactions
                buy_dates = [t['timestamp'] for t in self.transaction_history if t['action'] == 'buy']
                buy_prices = [t['price'] for t in self.transaction_history if t['action'] == 'buy']
                
                sell_dates = [t['timestamp'] for t in self.transaction_history if t['action'] == 'sell']
                sell_prices = [t['price'] for t in self.transaction_history if t['action'] == 'sell']
                
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_df['date'], portfolio_df['price'], label='Stock Price')
                
                # Plot predictions if available
                if self.prediction_history:
                    pred_df = pd.DataFrame(self.prediction_history)
                    if not pred_df.empty:
                        # Only plot some predictions to avoid clutter
                        step = max(1, len(pred_df) // 10)
                        plt.plot(pred_df['date'][::step], 
                                 pred_df['predicted_next_day'][::step], 
                                 'o--', color='purple', alpha=0.7, 
                                 label='Price Predictions (Next Day)')
                
                plt.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy')
                plt.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell')
                plt.title('Trading Decisions with Predictions')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'trading_decisions.png'))
                plt.close()
                
                # Add prediction accuracy metrics
                if self.prediction_history:
                    plt.figure(figsize=(12, 6))
                    pred_df = pd.DataFrame(self.prediction_history)
                    
                    if not pred_df.empty and len(pred_df) > 1:
                        # Get actual next day prices by shifting
                        pred_df['next_day_actual'] = pred_df['actual_price'].shift(-1)
                        
                        # Drop the last row which has NaN in next_day_actual
                        pred_df = pred_df.dropna()
                        
                        if not pred_df.empty:
                            # Calculate prediction errors
                            pred_df['prediction_error'] = pred_df['predicted_next_day'] - pred_df['next_day_actual']
                            pred_df['error_pct'] = (pred_df['prediction_error'] / pred_df['next_day_actual']) * 100
                            
                            # Plot prediction errors
                            plt.plot(pred_df['date'], pred_df['error_pct'])
                            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                            plt.title('Prediction Error Percentage Over Time')
                            plt.xlabel('Date')
                            plt.ylabel('Error (%)')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(os.path.join(self.results_dir, 'prediction_errors.png'))
                            plt.close()
                            
                            # Calculate and save accuracy metrics
                            mse = np.mean(pred_df['prediction_error'] ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(pred_df['prediction_error']))
                            mape = np.mean(np.abs(pred_df['error_pct']))
                            
                            directional_accuracy = np.mean(
                                (pred_df['predicted_next_day'] > pred_df['actual_price']) == 
                                (pred_df['next_day_actual'] > pred_df['actual_price'])
                            ) * 100
                            
                            with open(os.path.join(self.results_dir, 'prediction_metrics.txt'), 'w') as f:
                                f.write(f"Mean Squared Error: {mse:.4f}\n")
                                f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
                                f.write(f"Mean Absolute Error: {mae:.4f}\n")
                                f.write(f"Mean Absolute Percentage Error: {mape:.2f}%\n")
                                f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
                            
                            logging.info(f"Prediction metrics saved to {os.path.join(self.results_dir, 'prediction_metrics.txt')}")
                
                # Calculate and save trading performance metrics
                trading_metrics = self.calculate_performance_metrics()
                
                # Create confusion matrix for trading decisions if we have enough data
                if len(self.transaction_history) > 5:
                    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
                    
                    # Simple confusion matrix for trading decisions
                    # [Buy/Profit, Buy/Loss]
                    # [Sell/Loss, Sell/Profit]
                    cm = np.zeros((2, 2))
                    
                    # Calculate the confusion matrix values
                    buy_profit = trading_metrics.get('profitable_trades', 0)
                    buy_loss = trading_metrics.get('unprofitable_trades', 0)
                    
                    # Just for visualization - in a real system we'd track true sell decisions
                    # Here we're simplifying by assuming buy/profit = good decision, buy/loss = bad decision
                    cm[0, 0] = buy_profit  # Good buy decisions
                    cm[0, 1] = buy_loss    # Bad buy decisions
                    cm[1, 0] = buy_loss    # Bad sell decisions (mirrored for visualization)
                    cm[1, 1] = buy_profit  # Good sell decisions (mirrored for visualization)
                    
                    # Plot the confusion matrix
                    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.colorbar(im)
                    
                    # Add labels
                    ax.set_xticks(np.arange(2))
                    ax.set_yticks(np.arange(2))
                    ax.set_xticklabels(['Profit', 'Loss'])
                    ax.set_yticklabels(['Buy', 'Sell'])
                    
                    # Add text annotations
                    thresh = cm.max() / 2.0        
                    for i in range(2):        
                        for j in range(2):        
                            ax.text(j, i, f"{int(cm[i, j])}",        
                                    ha="center", va="center",        
                                    color="white" if cm[i, j] > thresh else "black")        
                                
                    plt.title('Trading Decision Outcomes')        
                    plt.tight_layout()        
                    plt.savefig(os.path.join(self.results_dir, 'trading_confusion_matrix.png'))        
                    plt.close()        
                
                # Save trading metrics to CSV
                metrics_results = {        
                    'Metric': [        
                        'ROI (%)',           
                        'Total Trades',        
                        'Win Rate (%)',        
                        'Accuracy',        
                        'Precision',        
                        'F1 Score',        
                        'Sharpe Ratio',        
                        'Max Drawdown (%)'        
                    ],        
                    'Value': [        
                        trading_metrics['roi'],        
                        trading_metrics['total_trades'],        
                        trading_metrics['win_rate'],        
                        trading_metrics['accuracy'],        
                        trading_metrics['precision'],        
                        trading_metrics['f1_score'],        
                        trading_metrics['sharpe_ratio'],        
                        trading_metrics['max_drawdown']        
                    ]        
                }        
                
                metrics_df = pd.DataFrame(metrics_results)        
                metrics_df.to_csv(os.path.join(self.results_dir, 'trading_performance_metrics.csv'), index=False)        
                
                # Save detailed metrics text file
                with open(os.path.join(self.results_dir, 'trading_performance.txt'), 'w') as f:        
                    f.write(f"==== Trading Performance Summary ====\n\n")        
                    f.write(f"ROI: {trading_metrics['roi']:.2f}%\n")        
                    f.write(f"Total Trades: {trading_metrics['total_trades']}\n")        
                    f.write(f"Win Rate: {trading_metrics['win_rate']:.2f}%\n")        
                    f.write(f"Profitable Trades: {trading_metrics['profitable_trades']}\n")        
                    f.write(f"Unprofitable Trades: {trading_metrics['unprofitable_trades']}\n")        
                    f.write(f"Neutral Trades: {trading_metrics['neutral_trades']}\n\n")        
                    
                    f.write(f"=== Advanced Metrics ===\n")        
                    f.write(f"Accuracy: {trading_metrics['accuracy']:.4f}\n")        
                    f.write(f"Precision: {trading_metrics['precision']:.4f}\n")        
                    f.write(f"Recall: {trading_metrics['recall']:.4f}\n")        
                    f.write(f"F1 Score: {trading_metrics['f1_score']:.4f}\n")        
                    f.write(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}\n")        
                    f.write(f"Maximum Drawdown: {trading_metrics['max_drawdown']:.2f}%\n")        
                
                logging.info(f"Trading performance metrics saved to {os.path.join(self.results_dir, 'trading_performance_metrics.csv')}")        
                        
            # Save threshold history
            if self.threshold_history:        
                thresh_df = pd.DataFrame(self.threshold_history)        
                thresh_path = os.path.join(self.results_dir, 'thresholds.csv')        
                thresh_df.to_csv(thresh_path, index=False)        
                logging.info(f"Threshold history saved to {thresh_path}")        
                
                # Plot threshold adaptation over time
                if len(thresh_df) > 1:        
                    plt.figure(figsize=(12, 8))        
                    plt.subplot(2, 1, 1)        
                    plt.plot(range(len(thresh_df)), thresh_df['buy_threshold'], 'g-', label='Buy Threshold')        
                    plt.plot(range(len(thresh_df)), -thresh_df['sell_threshold'], 'r-', label='Sell Threshold')        
                    plt.title('Adaptive Trading Thresholds')        
                    plt.xlabel('Trading Decision')        
                    plt.ylabel('Threshold (%)')        
                    plt.legend()        
                    
                    plt.subplot(2, 1, 2)        
                    plt.plot(range(len(thresh_df)), thresh_df['volatility'], 'b-', label='Volatility')        
                    if 'success_rate' in thresh_df.columns:        
                        plt.plot(range(len(thresh_df)), thresh_df['success_rate'] * 100, 'g--', label='Success Rate (%)')        
                    if 'prediction_accuracy' in thresh_df.columns:        
                        plt.plot(range(len(thresh_df)), thresh_df['prediction_accuracy'] * 100, 'y--', label='Prediction Accuracy (%)')        
                    plt.xlabel('Trading Decision')        
                    plt.ylabel('Metrics')        
                    plt.legend()        
                    
                    plt.tight_layout()        
                    plt.savefig(os.path.join(self.results_dir, 'adaptive_thresholds.png'))        
                    plt.close()        
            
            # Make sure to close all figures at the end
            plt.close('all')
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            # Still try to close figures even on error
            plt.close('all')
    
    def calculate_performance_metrics(self):
        """
        Calculate comprehensive trading performance metrics
        
        Returns:
            Dictionary of performance metrics including accuracy, precision, recall, and F1-score
        """
        # Need at least a few transactions to calculate meaningful metrics
        if len(self.transaction_history) < 3:
            return {
                'roi': 0,
                'total_trades': 0,
                'win_rate': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0, 
                'f1_score': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profitable_trades': 0,
                'unprofitable_trades': 0,
                'neutral_trades': 0
            }
        
        # Create dataframe from transaction history
        trades_df = pd.DataFrame(self.transaction_history)
        
        # Calculate ROI
        if len(self.portfolio_history) > 0:
            initial_value = self.initial_capital
            final_value = self.portfolio_history[-1]['portfolio_value']
            roi = ((final_value - initial_value) / initial_value) * 100
        else:
            roi = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Extract buy and sell trades
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        # Calculate total number of trades (buy + sell)
        total_trades = len(buy_trades) + len(sell_trades)
        
        # Prepare for win/loss calculation and profit analysis
        profitable_trades = 0
        unprofitable_trades = 0
        neutral_trades = 0
        
        # Calculate profit for each completed trade cycle (buy then sell)
        trade_results = []
        trade_returns = []
        
        # Track buy prices and quantities for profit calculation
        buy_queue = []
        
        for i, trade in enumerate(self.transaction_history):
            if trade['action'] == 'buy':
                buy_queue.append((trade['price'], trade['shares']))
            elif trade['action'] == 'sell' and buy_queue:
                # Calculate profit for this sell based on FIFO (First-In-First-Out)
                sell_price = trade['price']
                shares_to_sell = trade['shares']
                profit = 0
                
                while shares_to_sell > 0 and buy_queue:
                    buy_price, buy_shares = buy_queue[0]
                    
                    # Calculate how many shares to process in this iteration
                    shares_processed = min(shares_to_sell, buy_shares)
                    
                    # Calculate profit for these shares
                    trade_profit = (sell_price - buy_price) * shares_processed
                    profit += trade_profit
                    
                    # Update remaining shares
                    shares_to_sell -= shares_processed
                    
                    if shares_processed == buy_shares:
                        # All shares from this buy were sold
                        buy_queue.pop(0)
                    else:
                        # Only part of the shares were sold, update the buy queue
                        buy_queue[0] = (buy_price, buy_shares - shares_processed)
                
                # Record the result of this trade
                trade_results.append(1 if profit > 0 else (0 if profit == 0 else -1))
                trade_returns.append(profit)
                
                # Update profitable/unprofitable counts
                if profit > 0:
                    profitable_trades += 1
                elif profit < 0:
                    unprofitable_trades += 1
                else:
                    neutral_trades += 1
        
        # Win rate (percentage of profitable trades)
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
        else:
            win_rate = 0
        
        # Calculate portfolio returns for Sharpe ratio
        if len(self.portfolio_history) > 1:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_df['return'] = portfolio_df['portfolio_value'].pct_change()
            
            # Sharpe ratio (assuming risk-free rate of 0% and annualized by multiplying by sqrt(252))
            sharpe_ratio = 0
            if len(portfolio_df) > 1:
                avg_return = portfolio_df['return'].mean()
                std_return = portfolio_df['return'].std()
                if std_return > 0:
                    sharpe_ratio = (avg_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown = 0
        if len(self.portfolio_history) > 1:
            portfolio_values = [ph['portfolio_value'] for ph in self.portfolio_history]
            peak = portfolio_values[0]
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_drawdown:
                    max_drawdown = dd
            max_drawdown *= 100  # Convert to percentage
        
        # Calculate trading accuracy metrics if we have predictions
        accuracy = precision = recall = f1 = 0
        if self.prediction_history and len(trade_results) > 0:
            # True positives: bought when price went up or sold when price went down
            # False positives: bought when price went down or sold when price went up
            # True negatives: held when no profit opportunity
            # False negatives: held when there was a profit opportunity
            
            # For simplicity, we'll use the trade results to calculate basic metrics
            true_positives = profitable_trades
            false_positives = unprofitable_trades
            
            # Calculate precision, recall, and f1
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            
            # Recall and F1 require more complex calculations with hold decisions
            # For simplicity, we'll use win rate as accuracy
            accuracy = win_rate / 100
            
            # Simple F1 calculation
            if precision > 0 and accuracy > 0:
                f1 = 2 * (precision * accuracy) / (precision + accuracy)
        
        # Create metrics dictionary
        metrics = {
            'roi': roi,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'profitable_trades': profitable_trades,
            'unprofitable_trades': unprofitable_trades,
            'neutral_trades': neutral_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        logging.info(f"Trading Performance Metrics: ROI={roi:.2f}%, Win Rate={win_rate:.2f}%, " +
                     f"Max Drawdown={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return metrics