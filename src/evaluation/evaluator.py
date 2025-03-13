import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import logging

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
        Evaluate price prediction accuracy
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
        
        # Save results to CSV
        results = {
            'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'Direction Accuracy'],
            'Value': [mse, rmse, mae, mape, direction_correct]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, 'price_prediction_metrics.csv'), index=False)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_correct
        }
    
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Sell', 'Hold', 'Buy'],
                    yticklabels=['Sell', 'Hold', 'Buy'])
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
