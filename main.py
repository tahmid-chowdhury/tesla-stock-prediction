import os
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import custom modules
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.lstm_model import LSTMModel
from src.agent.trading_agent import TradingAgent
from src.evaluation.evaluator import ModelEvaluator
from src.models.hyperparameter_tuner import HyperparameterTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tesla Stock Trading Agent')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'trade'],
                        help='Mode of operation: train, test, or trade')
    
    parser.add_argument('--window-size', type=int, default=30,
                        help='Size of sliding window (days) for feature creation')
    
    parser.add_argument('--prediction-horizon', type=int, default=5,
                        help='Number of days to predict ahead')
    
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial investment amount')
    
    parser.add_argument('--transaction-fee', type=float, default=0.01,
                        help='Transaction fee as a percentage (0.01 = 1%)')
    
    parser.add_argument('--risk-factor', type=float, default=0.3,
                        help='Risk factor for trade sizing (0.1-0.5)')
    
    parser.add_argument('--newsapi-key', type=str, default=None,
                        help='API key for NewsAPI (optional)')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to saved model file')
    
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    
    parser.add_argument('--max-trials', type=int, default=10,
                        help='Maximum number of trials for hyperparameter tuning')
    
    parser.add_argument('--use-complete-history', action='store_true',
                        help='Use complete historical data for training')
    
    parser.add_argument('--stop-loss', type=float, default=7.0,
                        help='Stop loss percentage (e.g., 7.0 = 7%)')
    
    parser.add_argument('--trailing-stop', type=float, default=5.0,
                        help='Trailing stop percentage')
    
    parser.add_argument('--max-trades-per-day', type=int, default=2,
                        help='Maximum number of trades per day')
    
    parser.add_argument('--max-drawdown', type=float, default=20.0,
                        help='Maximum drawdown percentage before halting trading')
                        
    parser.add_argument('--volatility-scaling', action='store_true',
                        help='Enable volatility-based stop-loss scaling')
    
    return parser.parse_args()

def train_model(args):
    """Train the LSTM model"""
    logger.info("Starting model training process...")
    
    # Load data with option to use complete history
    data_loader = DataLoader("TSLA", api_key=args.newsapi_key)
    
    if args.use_complete_history:
        logger.info("Fetching complete historical data for TSLA")
        historical_data = data_loader.fetch_complete_history()
        
        # Also get recent data for better predictions
        recent_data = data_loader.fetch_stock_data()
        
        # Combine datasets
        stock_data = data_loader.combine_datasets(recent_data, historical_data)
        logger.info(f"Using combined dataset with {len(stock_data)} rows")
    else:
        stock_data = data_loader.fetch_stock_data()
    
    if args.newsapi_key:
        data_loader.fetch_news_data()
    
    # Preprocess data
    preprocessor = Preprocessor(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(stock_data)
    
    # Hyperparameter tuning if requested
    if args.tune:
        logger.info(f"Starting hyperparameter tuning with {args.max_trials} trials")
        
        tuner = HyperparameterTuner(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon,
            feature_dim=X_train.shape[2],
            max_trials=args.max_trials
        )
        
        best_model, history, best_hps = tuner.tune(
            X_train, y_train, 
            X_test, y_test, 
            epochs=100, 
            batch_size=32
        )
        
        # If tuning fails, use default model
        if best_model is None:
            logger.warning("Hyperparameter tuning failed. Using default model configuration.")
            model = LSTMModel(
                window_size=args.window_size,
                prediction_horizon=args.prediction_horizon,
                feature_dim=X_train.shape[2]
            )
            history = model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)
        else:
            model = best_model
            logger.info("Successfully created model with tuned hyperparameters")
    else:
        # Create and train model with default architecture
        feature_dim = X_train.shape[2]
        model = LSTMModel(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon,
            feature_dim=feature_dim
        )
        
        history = model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    predictions = model.predict(X_test)
    
    metrics = evaluator.evaluate_price_predictions(y_test, predictions, preprocessor.price_scaler)
    
    # Log all metrics including the new classification metrics
    logger.info(f"Training complete. RMSE: {metrics['rmse']:.2f}, Direction Accuracy: {metrics['direction_accuracy']:.2f}")
    
    if 'accuracy' in metrics:
        logger.info(f"Classification metrics - Accuracy: {metrics['accuracy']:.2f}, F1: {metrics['f1']:.2f}")
    
    return model, preprocessor, X_test, y_test

def test_model(args, model=None, preprocessor=None):
    """Test the model on historical data"""
    logger.info("Starting model testing process...")
    
    model_loaded = False
    
    if model is None or preprocessor is None:
        # Load data if not provided
        data_loader = DataLoader("TSLA")
        stock_data = data_loader.fetch_stock_data()
        
        preprocessor = Preprocessor(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon
        )
        
        try:
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(stock_data)
            
            # Load saved model
            model = LSTMModel(
                window_size=args.window_size,
                prediction_horizon=args.prediction_horizon,
                feature_dim=X_test.shape[2]
            )
            
            if args.load_model:
                model_loaded = model.load_saved_model(args.load_model)
            else:
                model_loaded = model.load_saved_model()
                
            if not model_loaded:
                logger.error("Failed to load model. Cannot proceed with testing.")
                return None
        except Exception as e:
            logger.error(f"Error preparing data or loading model: {e}")
            return None
    else:
        # Get test data from preprocessor
        try:
            processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
            X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
            y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
            model_loaded = True
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return None
    
    # Only proceed if we have a valid model
    if not model_loaded:
        logger.error("No valid model available for testing")
        return None
        
    # Create trading agent with stop-loss mechanism and new parameters
    agent = TradingAgent(
        model=model,
        price_scaler=preprocessor.price_scaler,
        initial_capital=args.initial_capital,
        transaction_fee=args.transaction_fee,
        risk_factor=args.risk_factor,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        max_trades_per_day=args.max_trades_per_day,
        max_drawdown_pct=args.max_drawdown,
        volatility_scaling=args.volatility_scaling
    )
    
    try:
        # Load dates for test data
        data_loader = DataLoader("TSLA")
        stock_data = data_loader.fetch_stock_data()
        test_dates = stock_data.index[-len(X_test):]
        
        # Run simulation
        final_value, roi = agent.run_simulation(
            test_data=X_test,
            test_dates=test_dates,
            feature_scaler=preprocessor.feature_scaler,
            final_trade=True
        )
        
        # Evaluate trading performance
        evaluator = ModelEvaluator()
        transactions_df = pd.DataFrame(agent.transaction_history)
        if not transactions_df.empty:
            metrics = evaluator.evaluate_trading_decisions(transactions_df)
            logger.info(f"Testing complete. Final portfolio value: ${final_value:.2f}, ROI: {roi:.2f}%")
        else:
            logger.warning("No transactions were executed during testing")
            
        return agent
    except Exception as e:
        logger.error(f"Error during testing simulation: {e}")
        return None

def trade(args):
    """Run the agent in trading mode on recent data"""
    logger.info("Starting trading mode...")
    
    # Load fresh data
    data_loader = DataLoader("TSLA", api_key=args.newsapi_key)
    
    # Set a more reasonable date range - last 365 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_data = data_loader.fetch_stock_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if args.newsapi_key:
        # Use a smaller window for news data to respect API limits
        data_loader.fetch_news_data(days_back=30)
    
    # Process data
    preprocessor = Preprocessor(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    preprocessor.prepare_data(stock_data)
    
    # Load model
    model = LSTMModel(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    
    if args.load_model:
        model.load_saved_model(args.load_model)
    else:
        model.load_saved_model()
    
    # Get most recent window of data for prediction
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    recent_window = X_train[-1:]  # Get the most recent window
    
    # Make prediction for next 5 days
    predictions = model.predict(recent_window)
    if preprocessor.price_scaler is not None:
        predictions = preprocessor.price_scaler.inverse_transform(predictions)
    
    # Get current price, handling MultiIndex columns if present
    try:
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Find the Close column in the MultiIndex
            close_cols = [col for col in stock_data.columns if 'Close' in col]
            if close_cols:
                close_col = close_cols[0]  # Take the first match
                current_price = stock_data[close_col].iloc[-1]  # Use iloc for positional indexing
                logger.info(f"Using MultiIndex column {close_col} for close price.")
            else:
                # Fallback to the first column if no Close column is found
                logger.warning("No 'Close' column found in MultiIndex. Using first column.")
                current_price = stock_data.iloc[:, 0].iloc[-1]
        else:
            # Regular single-level columns
            current_price = stock_data['Close'].iloc[-1]
        
        logger.info(f"Current price: ${current_price:.2f}")
    except Exception as e:
        # Provide a detailed error message and fallback value
        logger.error(f"Error retrieving current price: {e}")
        logger.error(f"Available columns: {stock_data.columns.tolist()}")
        logger.error(f"Using fallback current price of $1.00")
        current_price = 1.00
    
    # Calculate next 5 trading days
    today = datetime.now()
    next_days = []
    day_count = 0
    for i in range(1, 8):  # Look ahead up to 7 calendar days to find 5 trading days
        next_day = today + timedelta(days=i)
        if next_day.weekday() < 5:  # Monday to Friday (0-4)
            next_days.append(next_day.strftime('%Y-%m-%d'))
            day_count += 1
            if day_count >= args.prediction_horizon:
                break
    
    # Create trading recommendations
    trading_agent = TradingAgent(
        model=model,
        price_scaler=preprocessor.price_scaler,
        initial_capital=args.initial_capital,
        transaction_fee=args.transaction_fee
    )
    
    recommendations = []
    for i in range(args.prediction_horizon):
        pred_price = predictions[0][i]
        price_change = (pred_price - current_price) / current_price * 100
        
        if price_change > 2:
            action = "BUY"
        elif price_change < -2:
            action = "SELL"
        else:
            action = "HOLD"
        
        recommendations.append({
            'date': next_days[i],
            'predicted_price': pred_price,
            'change_percent': price_change,
            'recommendation': action
        })
    
    # Display recommendations
    rec_df = pd.DataFrame(recommendations)
    print("\nTrading Recommendations for Next 5 Trading Days:")
    print(rec_df.to_string(index=False))
    
    # Save recommendations
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    rec_df.to_csv(os.path.join(results_dir, f'recommendations_{datetime.now().strftime("%Y%m%d")}.csv'), index=False)
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    
    # Plot historical data (last 30 days)
    hist_dates = stock_data.index[-30:]
    hist_prices = stock_data['Close'][-30:].values
    plt.plot(range(len(hist_dates)), hist_prices, 'b-', label='Historical')
    
    # Plot predictions
    pred_prices = predictions[0]
    plt.plot(range(len(hist_dates), len(hist_dates) + len(pred_prices)), 
             pred_prices, 'r--', label='Predicted')
    
    # Set x-axis labels
    all_dates = np.concatenate([hist_dates.strftime('%Y-%m-%d').values, next_days])
    plt.xticks(range(0, len(all_dates), 5), all_dates[::5], rotation=45)
    
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, f'prediction_chart_{datetime.now().strftime("%Y%m%d")}.png'))
    plt.close()
    
    return rec_df

def main():
    """Main execution function"""
    args = parse_arguments()
    
    try:
        if args.mode == 'train':
            model, preprocessor, X_test, y_test = train_model(args)
            
            # Also run test after training
            test_model(args, model, preprocessor)
            
        elif args.mode == 'test':
            test_model(args)
            
        elif args.mode == 'trade':
            trade(args)
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
