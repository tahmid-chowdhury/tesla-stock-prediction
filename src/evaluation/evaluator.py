import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from datetime import datetime
from scipy import stats

# Try to import seaborn, but don't fail if it's not available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    logging.warning("Seaborn is not installed. Using matplotlib for visualizations instead.")
    logging.warning("For better visualizations, install seaborn using: pip install seaborn")
    SEABORN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self):
        """
        Initialize the evaluator
        """
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_price_predictions(self, actual, predicted, scaler=None):
        """
        Evaluate price predictions with enhanced metrics
        
        Args:
            actual: Actual price values
            predicted: Predicted price values
            scaler: Optional scaler for transforming values back to original scale
            
        Returns:
            Dictionary of evaluation metrics
        """
        if scaler:
            # Reshape for inverse transform if necessary
            if len(actual.shape) > 1:
                actual_reshaped = actual.reshape(-1, actual.shape[-1])
                predicted_reshaped = predicted.reshape(-1, predicted.shape[-1])
            else:
                actual_reshaped = actual.reshape(-1, 1)
                predicted_reshaped = predicted.reshape(-1, 1)
                
            # Transform back to original scale
            actual_orig = scaler.inverse_transform(actual_reshaped)
            predicted_orig = scaler.inverse_transform(predicted_reshaped)
            
            # Reshape back if necessary
            if len(actual.shape) > 1:
                actual_values = actual_orig.reshape(actual.shape)
                predicted_values = predicted_orig.reshape(predicted.shape)
            else:
                actual_values = actual_orig.flatten()
                predicted_values = predicted_orig.flatten()
        else:
            actual_values = actual
            predicted_values = predicted
        
        # Basic metrics
        mse = np.mean((actual_values - predicted_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_values - predicted_values))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        
        # Direction accuracy - whether prediction correctly captures up/down movement
        actual_direction = np.sign(np.diff(actual_values, axis=0))
        predicted_direction = np.sign(np.diff(predicted_values, axis=0))
        
        # Flatten if multidimensional
        if len(actual_direction.shape) > 1:
            actual_direction = actual_direction.flatten()
            predicted_direction = predicted_direction.flatten()
            
        # Replace any zeros with small positive value to avoid ambiguity
        actual_direction[actual_direction == 0] = 0.1
        predicted_direction[predicted_direction == 0] = 0.1
        
        direction_match = (actual_direction * predicted_direction) > 0
        direction_accuracy = np.mean(direction_match)
        
        # Calculate additional metrics for trading signals
        # Consider prediction as "buy" signal if predicted price goes up
        buy_signals = predicted_direction > 0
        
        # Calculate binary classification metrics
        actual_up = actual_direction > 0
        accuracy = accuracy_score(actual_up, buy_signals)
        
        # Precision, recall, F1-score
        try:
            precision = np.sum((buy_signals == True) & (actual_up == True)) / np.sum(buy_signals)
        except ZeroDivisionError:
            precision = 0
            
        try:
            recall = np.sum((buy_signals == True) & (actual_up == True)) / np.sum(actual_up)
        except ZeroDivisionError:
            recall = 0
            
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        # Calculate Sharpe ratio-like metric for predicted returns
        predicted_returns = np.diff(predicted_values, axis=0) / predicted_values[:-1]
        actual_returns = np.diff(actual_values, axis=0) / actual_values[:-1]
        
        if len(predicted_returns.shape) > 1:
            predicted_returns = predicted_returns.flatten()
            actual_returns = actual_returns.flatten()
        
        # Sharpe ratio (assuming daily data, annualized)
        pred_sharpe = np.sqrt(252) * np.mean(predicted_returns) / np.std(predicted_returns) if np.std(predicted_returns) > 0 else 0
        actual_sharpe = np.sqrt(252) * np.mean(actual_returns) / np.std(actual_returns) if np.std(actual_returns) > 0 else 0
        
        # Information coefficient - correlation between predicted and actual returns
        try:
            ic = np.corrcoef(predicted_returns, actual_returns)[0, 1]
        except:
            ic = 0
            
        # Maximum drawdown analysis
        actual_mdd = self.calculate_max_drawdown(actual_values)
        pred_mdd = self.calculate_max_drawdown(predicted_values)
            
        # Collect all metrics
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'information_coefficient': ic,
            'pred_sharpe': pred_sharpe,
            'actual_sharpe': actual_sharpe,
            'actual_max_drawdown': actual_mdd,
            'pred_max_drawdown': pred_mdd
        }
        
        # Log the metrics
        logging.info("Prediction Evaluation Metrics:")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"MAPE: {mape:.2f}%")
        logging.info(f"Direction Accuracy: {direction_accuracy:.4f}")
        logging.info(f"Buy Signal Accuracy: {accuracy:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Information Coefficient: {ic:.4f}")
        
        # Plot actual vs predicted values
        self.plot_predictions(actual_values, predicted_values)
        
        return metrics
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from a series of prices"""
        # Create a series of running maximum
        if len(prices.shape) > 1:
            prices = prices.flatten()
            
        running_max = np.maximum.accumulate(prices)
        # Calculate drawdown as percentage from running maximum
        drawdown = (running_max - prices) / running_max
        # Get the maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def detect_lookahead_bias(self, model, X_train, y_train, X_test, y_test, n_permutations=10):
        """
        Detect potential look-ahead bias in the model by testing permutations of future data
        
        Args:
            model: The trained model
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_permutations: Number of permutations to test
            
        Returns:
            Bias score (higher value indicates more look-ahead bias)
        """
        logging.info("Testing for look-ahead bias...")
        
        # Original performance
        orig_predictions = model.predict(X_test)
        orig_mse = mean_squared_error(y_test, orig_predictions)
        
        # Permutation test
        permutation_scores = []
        
        for i in range(n_permutations):
            # Create permuted test set with randomized future data
            X_permuted = X_test.copy()
            
            # For time series data, shuffle the time dimension within each sample
            # This simulates having access to "future" data in a random order
            for j in range(X_permuted.shape[0]):
                # Get indices except the last few (which are used for prediction)
                indices = np.arange(X_permuted.shape[1] - 5)
                np.random.shuffle(indices)
                
                # Apply shuffled indices
                X_permuted[j, :-5, :] = X_permuted[j, indices, :]
            
            # Test model on permuted data
            perm_predictions = model.predict(X_permuted)
            perm_mse = mean_squared_error(y_test, perm_predictions)
            
            # Calculate ratio of original MSE to permuted MSE
            # A value close to 1 suggests look-ahead bias (model works well even with shuffled data)
            ratio = orig_mse / perm_mse if perm_mse > 0 else 1.0
            permutation_scores.append(ratio)
            
        # Average bias score - closer to 1 means high risk of look-ahead bias
        bias_score = np.mean(permutation_scores)
        
        # Assess bias risk
        if bias_score > 0.9:
            logging.warning(f"High risk of look-ahead bias detected (score: {bias_score:.4f})")
        elif bias_score > 0.75:
            logging.warning(f"Moderate risk of look-ahead bias detected (score: {bias_score:.4f})")
        else:
            logging.info(f"Low risk of look-ahead bias (score: {bias_score:.4f})")
        
        return bias_score
    
    def evaluate_confidence_intervals(self, actual, predictions, confidence_intervals):
        """
        Evaluate the quality of prediction confidence intervals
        
        Args:
            actual: Actual values
            predictions: Predicted values
            confidence_intervals: Dict with lower and upper confidence bounds
            
        Returns:
            Metrics on confidence interval quality
        """
        if confidence_intervals is None:
            return {}
            
        # Extract bounds
        lower_bounds = confidence_intervals['lower_95']
        upper_bounds = confidence_intervals['upper_95']
        
        # Calculate coverage (percentage of actual values within confidence interval)
        within_bounds = (actual >= lower_bounds) & (actual <= upper_bounds)
        coverage = np.mean(within_bounds)
        
        # Calculate average interval width
        interval_width = np.mean(upper_bounds - lower_bounds)
        
        # Calculate interval score (proper scoring rule for interval forecasts)
        # Lower is better - penalizes wide intervals and violations
        alpha = 0.05  # for 95% confidence intervals
        interval_score = np.mean(
            (upper_bounds - lower_bounds) + 
            (2/alpha) * (lower_bounds - actual) * (actual < lower_bounds) +
            (2/alpha) * (actual - upper_bounds) * (actual > upper_bounds)
        )
        
        metrics = {
            'coverage': coverage,
            'interval_width': interval_width,
            'interval_score': interval_score
        }
        
        logging.info(f"Confidence Interval Evaluation:")
        logging.info(f"Coverage (target: 0.95): {coverage:.4f}")
        logging.info(f"Average interval width: {interval_width:.4f}")
        logging.info(f"Interval score (lower is better): {interval_score:.4f}")
        
        return metrics
        
    def plot_predictions(self, actual, predicted):
        """Plot actual vs predicted values with visualization enhancements"""
        # For multivariate predictions (multiple days ahead)
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            plt.figure(figsize=(12, 8))
            
            # Plot the first day predictions
            plt.subplot(2, 1, 1)
            plt.plot(actual[:, 0], label='Actual', linewidth=2)
            plt.plot(predicted[:, 0], label='Predicted', linestyle='--', linewidth=2)
            plt.title('1-Day Ahead Predictions')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Plot the last day predictions
            plt.subplot(2, 1, 2)
            plt.plot(actual[:, -1], label='Actual', linewidth=2)
            plt.plot(predicted[:, -1], label='Predicted', linestyle='--', linewidth=2)
            plt.title(f'{actual.shape[1]}-Day Ahead Predictions')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
        else:
            # For univariate predictions
            plt.figure(figsize=(12, 6))
            plt.plot(actual, label='Actual', linewidth=2)
            plt.plot(predicted, label='Predicted', linestyle='--', linewidth=2)
            plt.title('Price Predictions')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.results_dir, f'prediction_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
    
    def evaluate_trading_decisions(self, transactions_df):
        """
        Evaluate trading decision effectiveness
        """
        if transactions_df is None or len(transactions_df) == 0:
            logging.warning("No transactions to evaluate")
            return None
            
        # Calculate trade metrics
        num_trades = len(transactions_df[transactions_df['action'].isin(['buy', 'sell'])])
        num_buy = len(transactions_df[transactions_df['action'] == 'buy'])
        num_sell = len(transactions_df[transactions_df['action'] == 'sell'])
        num_hold = len(transactions_df[transactions_df['action'] == 'hold'])
        
        # Calculate profit/loss
        initial_capital = transactions_df.iloc[0]['capital_after'] + \
                         transactions_df.iloc[0]['shares_after'] * transactions_df.iloc[0]['price']
                         
        final_capital = transactions_df.iloc[-1]['capital_after'] + \
                       transactions_df.iloc[-1]['shares_after'] * transactions_df.iloc[-1]['price']
                       
        total_profit = final_capital - initial_capital
        roi = (total_profit / initial_capital) * 100
        
        # Calculate fees
        total_fees = transactions_df['fee'].sum()
        
        # Save results
        results = {
            'Metric': ['Total Trades', 'Buy Orders', 'Sell Orders', 'Hold Orders', 
                      'Total Profit/Loss', 'ROI (%)', 'Total Fees'],
            'Value': [num_trades, num_buy, num_sell, num_hold, 
                     total_profit, roi, total_fees]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, 'trading_metrics.csv'), index=False)
        
        # Plot profit/loss over time
        plt.figure(figsize=(12, 6))
        portfolio_values = transactions_df['portfolio_value']
        plt.plot(transactions_df['timestamp'], portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'portfolio_performance.png'))
        plt.close()
        
        return {
            'num_trades': num_trades,
            'num_buy': num_buy,
            'num_sell': num_sell,
            'num_hold': num_hold,
            'total_profit': total_profit,
            'roi': roi,
            'total_fees': total_fees
        }
    
    def evaluate_classifier_metrics(self, y_true, y_pred):
        """
        Calculate classifier metrics for trading decisions
        
        Args:
            y_true: Array of true classes (0: Sell, 1: Hold, 2: Buy)
            y_pred: Array of predicted classes
        """
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        
        # Use seaborn if available, otherwise use matplotlib
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Sell', 'Hold', 'Buy'],
                      yticklabels=['Sell', 'Hold', 'Buy'])
        else:
            # Create a simple heatmap with matplotlib
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add labels and values to the plot
            tick_marks = np.arange(len(['Sell', 'Hold', 'Buy']))
            plt.xticks(tick_marks, ['Sell', 'Hold', 'Buy'], rotation=45)
            plt.yticks(tick_marks, ['Sell', 'Hold', 'Buy'])
            
            # Add text annotations for values
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Save metrics to CSV
        results = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [accuracy, precision, recall, f1]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, 'classification_metrics.csv'), index=False)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _plot_predictions(self, y_true, y_pred):
        """
        Plot true vs predicted values and save the figure
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted
        plt.plot(y_true, label='True Values', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predictions', color='red', alpha=0.7)
        
        plt.title('Stock Price Prediction: True vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        filename = os.path.join(self.results_dir, f'prediction_vs_true_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        plt.close()
        
        # Create scatterplot to show correlation
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Add y=x line
        
        plt.title('Scatter Plot: True vs Predicted')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        
        # Save the figure
        filename = os.path.join(self.results_dir, f'prediction_scatter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        plt.close()
        
    def _plot_horizon_errors(self, horizon_metrics):
        """
        Plot prediction errors over the forecasting horizon
        
        Args:
            horizon_metrics: Dictionary containing error metrics for each day in the horizon
        """
        if not horizon_metrics:
            return
        
        horizons = sorted(horizon_metrics.keys())
        days = [int(h.split('_')[1]) for h in horizons]
        
        rmse_values = [horizon_metrics[h]['rmse'] for h in horizons]
        mape_values = [horizon_metrics[h]['mape'] for h in horizons]
        
        plt.figure(figsize=(12, 6))
        
        # Plot RMSE by day
        plt.subplot(1, 2, 1)
        plt.plot(days, rmse_values, 'o-', color='blue')
        plt.title('RMSE by Forecast Horizon')
        plt.xlabel('Days Ahead')
        plt.ylabel('RMSE')
        plt.grid(True)
        
        # Plot MAPE by day
        plt.subplot(1, 2, 2)
        plt.plot(days, mape_values, 'o-', color='red')
        plt.title('MAPE by Forecast Horizon')
        plt.xlabel('Days Ahead')
        plt.ylabel('MAPE (%)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.results_dir, f'horizon_errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        plt.close()
    
    def _plot_trading_results(self, transactions_df):
        """
        Create visualizations of trading results
        
        Args:
            transactions_df: DataFrame containing transaction data
        """
        if transactions_df.empty:
            return
            
        # Plot portfolio value over time
        plt.figure(figsize=(12, 8))
        
        # Add portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(transactions_df.index, transactions_df['portfolio_value'], 'b-')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Trade')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        
        # Plot buy/sell points with profit coloring
        plt.subplot(2, 1, 2)
        plt.plot(transactions_df.index, transactions_df['price'], 'k-', alpha=0.5)
        
        # Mark buy points
        buys = transactions_df[transactions_df['action'] == 'BUY']
        plt.scatter(buys.index, buys['price'], color='green', marker='^', s=100, label='Buy')
        
        # Mark sell points colored by profit/loss
        sells = transactions_df[transactions_df['action'] == 'SELL']
        
        # Profitable sells (green)
        profitable_sells = sells[sells['profit'] > 0]
        plt.scatter(profitable_sells.index, profitable_sells['price'], color='darkgreen', marker='v', s=100, label='Sell (Profit)')
        
        # Unprofitable sells (red)
        unprofitable_sells = sells[sells['profit'] <= 0]
        plt.scatter(unprofitable_sells.index, unprofitable_sells['price'], color='red', marker='v', s=100, label='Sell (Loss)')
        
        plt.title('Trading Actions')
        plt.xlabel('Trade')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.results_dir, f'trading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        plt.close()
        
        # Create profit/loss distribution chart
        plt.figure(figsize=(10, 6))
        sns.histplot(transactions_df['profit'], kde=True, bins=20)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save the figure
        filename = os.path.join(self.results_dir, f'profit_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(filename)
        plt.close()
