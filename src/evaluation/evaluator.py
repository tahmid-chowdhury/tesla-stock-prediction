import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

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
    
    def evaluate_price_predictions(self, y_true, y_pred, price_scaler=None):
        """
        Evaluate price prediction accuracy with enhanced metrics
        """
        # If scaler is provided, inverse transform to original price scale
        if price_scaler is not None:
            if len(y_true.shape) == 2:
                y_true = price_scaler.inverse_transform(y_true)
                y_pred = price_scaler.inverse_transform(y_pred)
            else:
                # Reshape to 2D if needed
                y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
                y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])
                
                y_true = price_scaler.inverse_transform(y_true_reshaped).reshape(y_true.shape)
                y_pred = price_scaler.inverse_transform(y_pred_reshaped).reshape(y_pred.shape)
                
        # Calculate regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate directional accuracy (up/down)
        y_true_dir = np.diff(y_true, axis=1) > 0
        y_pred_dir = np.diff(y_pred, axis=1) > 0
        
        direction_correct = np.mean(y_true_dir == y_pred_dir)
        
        # Calculate additional metrics
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate price direction prediction accuracy (up or down)
        # For multi-step predictions, we calculate for each step
        direction_accuracies = []
        for i in range(1, y_true.shape[1]):
            y_true_dir = (y_true[:, i] > y_true[:, i-1])
            y_pred_dir = (y_pred[:, i] > y_pred[:, i-1])
            direction_accuracies.append(np.mean(y_true_dir == y_pred_dir))
        
        overall_direction_accuracy = np.mean(direction_accuracies)
        
        # Generate directional trading signals (1: Buy, 0: Hold, -1: Sell)
        y_true_signals = np.zeros_like(y_true[:, :-1])
        y_pred_signals = np.zeros_like(y_pred[:, :-1])
        
        # Signal is Buy if price increases more than 1%, Sell if decreases more than 1%, else Hold
        threshold = 0.01  # 1%
        
        for i in range(y_true.shape[1]-1):
            y_true_pct_change = (y_true[:, i+1] - y_true[:, i]) / y_true[:, i]
            y_pred_pct_change = (y_pred[:, i+1] - y_pred[:, i]) / y_pred[:, i]
            
            # True signals
            y_true_signals[:, i] = np.where(y_true_pct_change > threshold, 1, 
                                            np.where(y_true_pct_change < -threshold, -1, 0))
            
            # Predicted signals
            y_pred_signals[:, i] = np.where(y_pred_pct_change > threshold, 1, 
                                           np.where(y_pred_pct_change < -threshold, -1, 0))
        
        # Flatten for classification metrics
        y_true_signals_flat = y_true_signals.flatten()
        y_pred_signals_flat = y_pred_signals.flatten()
        
        # Only calculate classification metrics if we have more than one class
        unique_classes = np.unique(np.concatenate([y_true_signals_flat, y_pred_signals_flat]))
        
        classification_metrics = {}
        if len(unique_classes) > 1:
            # Convert to classes compatible with sklearn
            y_true_classes = y_true_signals_flat + 1  # Convert from [-1,0,1] to [0,1,2]
            y_pred_classes = y_pred_signals_flat + 1
            
            # Calculate classification metrics
            try:
                classification_metrics = {
                    'accuracy': accuracy_score(y_true_classes, y_pred_classes),
                    'precision': precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0),
                    'recall': recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0),
                    'f1': f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
                }
                
                # Create confusion matrix for signal prediction
                cm = confusion_matrix(y_true_classes, y_pred_classes)
                
                # Plot confusion matrix
                plt.figure(figsize=(10, 8))
                if SEABORN_AVAILABLE:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Sell', 'Hold', 'Buy'],
                              yticklabels=['Sell', 'Hold', 'Buy'])
                else:
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.colorbar()
                    
                    # Add labels and values to the plot
                    tick_marks = np.arange(3)
                    plt.xticks(tick_marks, ['Sell', 'Hold', 'Buy'])
                    plt.yticks(tick_marks, ['Sell', 'Hold', 'Buy'])
                    
                    # Add text annotations for values
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > cm.max() / 2 else "black")
                
                plt.title('Trading Signal Confusion Matrix')
                plt.xlabel('Predicted Signal')
                plt.ylabel('True Signal')
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'signal_confusion_matrix.png'))
                plt.close()
            except Exception as e:
                logging.warning(f"Could not calculate classification metrics: {e}")
        
        # Save results to CSV with added metrics
        results = {
            'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'Direction Accuracy'],
            'Value': [mse, rmse, mae, mape, overall_direction_accuracy]
        }
        
        # Add classification metrics if available
        for metric, value in classification_metrics.items():
            results['Metric'].append(f'Signal {metric.capitalize()}')
            results['Value'].append(value)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, 'price_prediction_metrics.csv'), index=False)
        
        # Return all metrics
        metrics_dict = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': overall_direction_accuracy
        }
        metrics_dict.update(classification_metrics)
        
        return metrics_dict
    
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
