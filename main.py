import os
import numpy as np  # Fixed syntax error: changed "import numpy np" to "import numpy as np"
import pandas as pd
import argparse
import logging
from datetime import datetime, timedelta

# Import warning suppression utility first
from src.utils.tf_warning_suppressor import suppress_tensorflow_warnings

# Set matplotlib backend to 'Agg' for non-interactive use to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

# Configure GPU memory usage (if available)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Found {len(gpus)} GPU(s)")
        
        # Use mixed precision policy for faster training on compatible GPUs
        try:
            from tensorflow.keras.mixed_precision import set_global_policy
            set_global_policy('mixed_float16')
            logging.info("Mixed precision training enabled")
        except:
            # Fall back to older TensorFlow versions
            try:
                from tensorflow.keras.mixed_precision import experimental as mixed_precision
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logging.info("Mixed precision training enabled (legacy mode)")
            except RuntimeError as e:
                logging.warning(f"Mixed precision configuration error: {e}")
                logging.warning("Running with default GPU configuration")
except ImportError:
    logging.warning("TensorFlow not imported correctly. Running in CPU mode.")

# Import custom modules
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.lstm_model import LSTMModel
from src.agent.trading_agent import TradingAgent
from src.evaluation.evaluator import ModelEvaluator
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.visualization.metrics_visualizer import MetricsVisualizer
from src.utils.performance_monitor import PerformanceMonitor

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

def close_all_figures():
    """Helper function to properly close all matplotlib figures to avoid Tkinter errors"""
    plt.close('all')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tesla Stock Trading Agent')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'trade', 'visualize-metrics'],
                        help='Mode of operation: train, test, trade, or visualize-metrics')
    
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
    # New arguments for sentiment analysis
    parser.add_argument('--use-sentiment', action='store_true', default=True,
                        help='Use news sentiment analysis in the model')
    
    parser.add_argument('--sentiment-weight', type=float, default=0.3,
                        help='Weight for sentiment influence on trading decisions (0.0-1.0)')
    
    parser.add_argument('--sentiment-days', type=int, default=30,
                        help='Number of days of news to fetch for sentiment analysis')
    
    # Add new argument for metrics visualization
    parser.add_argument('--num-iterations', type=int, default=10,
                       help='Number of previous training iterations to include in metrics visualization')
    
    # Add new arguments for enhanced functionality
    parser.add_argument('--use-ensemble', action='store_true',
                        help='Use ensemble of models for better prediction stability')
    
    parser.add_argument('--ensemble-size', type=int, default=3,
                        help='Number of models in the ensemble')
    
    parser.add_argument('--detect-regimes', action='store_true',
                        help='Detect and use market regimes for predictions')
    
    parser.add_argument('--num-regimes', type=int, default=3,
                        help='Number of market regimes to detect')
    
    parser.add_argument('--lookahead-test', action='store_true',
                       help='Test for potential look-ahead bias in the model')
    
    parser.add_argument('--confidence-intervals', action='store_true',
                       help='Generate and evaluate confidence intervals for predictions')
    
    parser.add_argument('--adaptive-horizon', action='store_true',
                       help='Use adaptive prediction horizon based on market conditions')
    
    # Add new arguments for performance optimization
    parser.add_argument('--batch-size', type=int, default=64,  # Increased from 32 to 64
                        help='Batch size for model training')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum number of epochs for model training')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping during training')
    parser.add_argument('--intelligent-sampling', action='store_true',
                        help='Enable intelligent data sampling for training')
    parser.add_argument('--simplified-architecture', action='store_true',
                        help='Use simplified model architecture for faster training')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum number of samples to use for training')
    parser.add_argument('--reduced-features', action='store_true',
                        help='Use reduced training data for faster iteration')
    parser.add_argument('--feature-count', type=int, default=10,
                        help='Number of features to use if feature reduction is enabled')
    
    return parser.parse_args()

def train_model(args):
    """Train the LSTM model"""
    logger.info("Starting model training process...")
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_timer('total_execution')
    
    # Initialize metrics visualizer
    metrics_visualizer = MetricsVisualizer()
    
    # Load previous best metrics if they exist
    previous_best_metrics = load_previous_metrics()
    if previous_best_metrics:
        logger.info("Previous best model metrics loaded")
        logger.info(f"Previous RMSE: {previous_best_metrics['rmse']:.4f}, Direction Accuracy: {previous_best_metrics['direction_accuracy']:.4f}")
        if 'f1' in previous_best_metrics:
            logger.info(f"Previous F1: {previous_best_metrics['f1']:.4f}")
    
    # Monitor data loading
    perf_monitor.start_timer('data_loading')
    
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
    
    # Fetch news data if API key is provided and sentiment analysis is enabled
    if args.newsapi_key and args.use_sentiment:
        logger.info(f"Fetching news data for sentiment analysis (past {args.sentiment_days} days)")
        data_loader.fetch_news_data(days_back=args.sentiment_days)
    
    perf_monitor.stop_timer('data_loading')
    
    # Monitor preprocessing 
    perf_monitor.start_timer('data_preprocessing')
    
    # Debug the column structure before processing
    if isinstance(stock_data.columns, pd.MultiIndex):
        logger.info(f"Stock data has MultiIndex columns: {[col for col in stock_data.columns[:5]]}")
        # If MultiIndex columns, normalize them first before preprocessing
        renamed_data = {}
        for name in ['Close', 'Open', 'High', 'Low', 'Volume']:
            col_matches = [col for col in stock_data.columns if name in col[0]]
            if col_matches:
                renamed_data[name] = stock_data[col_matches[0]]
        if renamed_data:
            logger.info("Standardizing MultiIndex columns to simple column names")
            stock_data = pd.DataFrame(renamed_data, index=stock_data.index)
    
    # Detect market regimes if requested
    if args.detect_regimes:
        logger.info(f"Detecting market regimes (n={args.num_regimes})...")
        # Create preprocessor with regime detection
        preprocessor = Preprocessor(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon,
            detect_regimes=True,
            num_regimes=args.num_regimes
        )
    else:
        # Standard preprocessor
        preprocessor = Preprocessor(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon
        )
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(stock_data)
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        if isinstance(stock_data.columns, pd.MultiIndex):
            logger.error("Trying alternative MultiIndex handling...")
            # Flatten the MultiIndex columns
            stock_data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in stock_data.columns]
            # Try again with flattened columns
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(stock_data)
        else:
            raise
    
    perf_monitor.stop_timer('data_preprocessing')
    
    # Log data shape metrics
    perf_monitor.record_metric('train_samples', len(X_train))
    perf_monitor.record_metric('test_samples', len(X_test))
    perf_monitor.record_metric('feature_count', X_train.shape[2])
    perf_monitor.record_metric('window_size', args.window_size)
    perf_monitor.record_metric('batch_size', args.batch_size)
    perf_monitor.record_metric('max_epochs', args.max_epochs)
    
    # Check for sentiment features
    sentiment_features = []
    try:
        processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
        sentiment_features_path = os.path.join(processed_dir, "sentiment_columns.npy")
        if (os.path.exists(sentiment_features_path)):
            sentiment_features = np.load(sentiment_features_path, allow_pickle=True)
            logger.info(f"Loaded {len(sentiment_features)} sentiment features: {sentiment_features}")
    except Exception as e:
        logger.warning(f"Could not load sentiment features: {e}")
    
    # Check for lookahead bias if requested
    if args.lookahead_test:
        X_train_lookahead_test = X_train.copy()
        y_train_lookahead_test = y_train.copy()
        
        # Create a small model for bias testing
        test_model = LSTMModel(
            window_size=args.window_size,
            prediction_horizon=args.prediction_horizon,
            feature_dim=X_train.shape[2]
        )
        test_model.build_model(lstm_units=64, dropout_rate=0.2)
        
        # Train for just a few epochs
        test_model.train(
            X_train_lookahead_test, y_train_lookahead_test,
            X_val=X_test[:100], y_val=y_test[:100],  # Use subset for speed
            epochs=10, batch_size=32
        )
        
        # Test for lookahead bias
        evaluator = ModelEvaluator()
        bias_score = evaluator.detect_lookahead_bias(
            test_model, X_train_lookahead_test, y_train_lookahead_test,
            X_test, y_test, n_permutations=5
        )
        
        logger.info(f"Lookahead bias test complete. Score: {bias_score:.4f}")
        
        # If bias score is high, warn the user
        if (bias_score > 0.8):
            logger.warning("WARNING: High risk of look-ahead bias detected in your data!")
            logger.warning("This may cause optimistic performance estimates that won't generalize.")
            logger.warning("Consider reviewing your feature engineering for time leakage.")
            # Offer option to abort
            if input("Would you like to continue training anyway? (y/n): ").lower() != 'y':
                logger.info("Training aborted due to look-ahead bias concerns.")
                return None, None, None, None
    
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
        # Create and train model with ensemble if requested
        if args.use_ensemble:
            feature_dim = X_train.shape[2]
            model = LSTMModel(
                window_size=args.window_size,
                prediction_horizon=args.prediction_horizon,
                feature_dim=feature_dim
            )
            logger.info(f"Training ensemble model with {args.ensemble_size} members")
            # Train with ensemble
            history = model.train(
                X_train, y_train, 
                X_val=X_test, y_val=y_test, 
                epochs=100,
                use_ensemble=True,
                n_ensemble_models=args.ensemble_size
            )
        else:
            # Create and train model with default architecture
            feature_dim = X_train.shape[2]
            model = LSTMModel(
                window_size=args.window_size,
                prediction_horizon=args.prediction_horizon,
                feature_dim=feature_dim
            )
            # After the model is created and before training, log the input shape and feature dimensions
            if hasattr(model, 'model') and model.model is not None:
                try:
                    model_input_shape = model.model.input_shape
                    if model_input_shape is not None:
                        logger.info(f"Model input shape for training: {model_input_shape}, features: {X_train.shape[2]}")
                        # Check for mismatch
                        if model_input_shape[-1] != X_train.shape[2]:
                            logger.warning(f"Feature dimension mismatch during training: model expects {model_input_shape[-1]} but data has {X_train.shape[2]}")
                except Exception as e:
                    logger.warning(f"Could not determine model input shape during training: {e}")
            
            # Monitor training
            perf_monitor.start_timer('model_training')
            perf_monitor.log_memory_usage()
            
            history = model.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)
            
            perf_monitor.stop_timer('model_training')
            perf_monitor.log_memory_usage()
    
    # Monitor evaluation
    perf_monitor.start_timer('model_evaluation')
    
    # Evaluate model
    evaluator = ModelEvaluator()
    
    # Use confidence intervals if requested
    if args.confidence_intervals:
        # Get predictions with confidence intervals - handle different model types
        try:
            # Try to use the custom predict method with use_ensemble parameter
            predictions, confidence_intervals = model.predict(
                X_test, 
                use_ensemble=args.use_ensemble,
                return_confidence=True
            )
        except (TypeError, AttributeError) as e:
            # Fall back to standard predict method for TensorFlow models
            logger.warning(f"Model doesn't support custom prediction parameters: {e}")
            logger.warning("Falling back to standard prediction without ensemble/confidence")
            
            # Standard TensorFlow predict
            predictions = model.predict(X_test)
            
            # Create default confidence intervals
            confidence_intervals = {
                'lower_95': predictions * 0.9,  # Simple 10% lower bound
                'upper_95': predictions * 1.1   # Simple 10% upper bound
            }
            logger.info("Created simple confidence intervals for standard model")
        
        # Plot predictions with confidence intervals
        plot_predictions_with_confidence(
            y_test, predictions, confidence_intervals,
            preprocessor.price_scaler
        )
        
        # Evaluate confidence intervals
        ci_metrics = evaluator.evaluate_confidence_intervals(
            y_test, predictions, confidence_intervals
        )
        # Add confidence interval metrics to main metrics
        metrics = evaluator.evaluate_price_predictions(y_test, predictions, preprocessor.price_scaler)
        metrics.update(ci_metrics)
    else:
        # Standard prediction - handle different model types
        try:
            # Try to use the custom predict method with use_ensemble parameter
            predictions = model.predict(X_test, use_ensemble=args.use_ensemble)
        except (TypeError, AttributeError) as e:
            # Fall back to standard predict method for TensorFlow models
            logger.warning(f"Model doesn't support ensemble prediction: {e}")
            predictions = model.predict(X_test)
    
        metrics = evaluator.evaluate_price_predictions(y_test, predictions, preprocessor.price_scaler)
    
    perf_monitor.stop_timer('model_evaluation')
    
    # Save metrics to history before comparing with previous best
    metrics_visualizer.save_metrics(metrics)
    
    # Compare with previous best and save if better
    is_better = compare_and_save_metrics(metrics, previous_best_metrics)
    
    # Generate performance metrics history visualization
    metrics_history_path = metrics_visualizer.plot_training_history(n_iterations=10)
    if metrics_history_path:
        logger.info(f"Performance metrics history visualization saved to {metrics_history_path}")
    
    # Print comparison with previous best
    if previous_best_metrics:
        print("\nPerformance Comparison:")
        print("-" * 50)
        print(f"{'Metric':<20} {'Previous':<15} {'Current':<15} {'Change':<15}")
        print("-" * 50)
        
        for key in ['rmse', 'mae', 'direction_accuracy', 'accuracy', 'f1']:
            if key in metrics and key in previous_best_metrics:
                change = metrics[key] - previous_best_metrics[key]
                change_symbol = "↑" if change > 0 else "↓" if change < 0 else "="
                improvement = "better" if change < 0 else "worse" if change > 0 else "same"
                
                print(f"{key.upper():<20} {previous_best_metrics[key]:.4f}      {metrics[key]:.4f}      {change:.4f} {change_symbol} ({improvement})")
        
        if is_better:
            print("\n✓ Current model SAVED as new best model")
        else:
            print("\n✗ Current model NOT saved (previous model is better)")
    
    # Use market regime detection if requested
    if args.detect_regimes and hasattr(model, 'market_regimes'):
        # Check if we have detected regimes
        if model.market_regimes and 'labels' in model.market_regimes:
            regimes = model.market_regimes['labels']
            
            # Evaluate performance across different regimes
            unique_regimes = np.unique(regimes)
            
            print("\nPerformance by Market Regime:")
            print("-" * 50)
            
            for regime in unique_regimes:
                # Get data points for this regime
                regime_indices = np.where(regimes == regime)[0]
                if len(regime_indices) > 10:  # Only evaluate if we have enough samples
                    regime_X = X_test[regime_indices]
                    regime_y = y_test[regime_indices]
                    # Get predictions for this regime
                    regime_predictions = model.predict(regime_X)
                    # Calculate basic metrics for this regime
                    regime_metrics = evaluator.evaluate_price_predictions(
                        regime_y, regime_predictions, preprocessor.price_scaler
                    )
                    # Print regime-specific metrics
                    print(f"Regime {regime} ({len(regime_indices)} samples):")
                    print(f"  RMSE: {regime_metrics['rmse']:.4f}")
                    print(f"  Direction Accuracy: {regime_metrics['direction_accuracy']:.4f}")
                    print(f"  F1 Score: {regime_metrics['f1']:.4f}")
                    
    return model, preprocessor, X_test, y_test

def load_previous_metrics():
    """Load metrics from previous best model"""
    metrics_path = os.path.join(os.path.dirname(__file__), "results", "best_model_metrics.json")
    
    if os.path.exists(metrics_path):
        try:
            # Import the JSON repair utility
            from src.utils.json_repair import repair_json_file
            
            # Try to load and potentially repair the JSON file
            success, data = repair_json_file(metrics_path)
            if success and data:
                logging.info("Successfully loaded previous metrics")
                return data
            else:
                logging.warning("Could not load or repair metrics file")
                return None
        except Exception as e:
            logging.error(f"Error loading previous metrics: {e}")
            return None
    else:
        return None

def calculate_score(metrics):
    """Calculate a composite score from metrics (higher is better)"""
    score = 0
    if 'rmse' in metrics:
        score -= 0.3 * metrics['rmse']  # Lower RMSE is better
    if 'direction_accuracy' in metrics:
        score += 0.3 * metrics['direction_accuracy']  # Higher direction accuracy is better
    if 'f1' in metrics:
        score += 0.4 * metrics['f1']  # Higher F1 is better
    return score

def compare_and_save_metrics(current_metrics, previous_metrics):
    """Compare current metrics with previous best and save if better"""
    current_score = calculate_score(current_metrics)
    
    # If no previous metrics or current is better, save current as best
    if not previous_metrics or current_score > calculate_score(previous_metrics):
        metrics_path = os.path.join(os.path.dirname(__file__), "results", "best_model_metrics.json")
        
        try:
            # Ensure results directory exists
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            
            # Convert NumPy types to Python native types before saving
            serializable_metrics = {}
            for key, value in current_metrics.items():
                # Convert NumPy types to Python native types
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                else:
                    serializable_metrics[key] = value
            
            # Save metrics using NumpyJSONEncoder
            with open(metrics_path, 'w') as f:
                import json
                from src.visualization.metrics_visualizer import NumpyJSONEncoder
                json.dump(serializable_metrics, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"New best metrics saved with score: {current_score:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    else:
        logger.info(f"Current model score ({current_score:.4f}) did not improve upon previous best")
        return False

def test_model(args, model=None, preprocessor=None):
    """Test the model on historical data"""
    logger.info("Starting model testing process...")
    
    # Initialize data_loader at the beginning regardless of path
    data_loader = DataLoader("TSLA", api_key=args.newsapi_key)
    
    model_loaded = False
    if model is None or preprocessor is None:
        # Load data if not provided
        stock_data = data_loader.fetch_stock_data()
        
        # Fetch news data for sentiment if requested
        if args.newsapi_key and args.use_sentiment:
            logger.info(f"Fetching news data for testing (past {args.sentiment_days} days)")
            data_loader.fetch_news_data(days_back=args.sentiment_days)
        
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
        # Even when model and preprocessor are provided, we still need stock_data for dates
        stock_data = data_loader.fetch_stock_data()
        
        # Get test data from preprocessor
        processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
        try:
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
    
    # Run the agent with stop-loss mechanism and new parameters
    agent = TradingAgent(
        model=model,
        price_scaler=preprocessor.price_scaler,
        initial_capital=args.initial_capital,
        transaction_fee=args.transaction_fee / 100,  # Convert from percentage to decimal
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        max_trades_per_day=args.max_trades_per_day,
        max_drawdown_pct=args.max_drawdown,
        volatility_scaling=args.volatility_scaling,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon  # Pass the prediction horizon explicitly
    )
    
    # Set sentiment influence if using sentiment
    if hasattr(agent, 'sentiment_influence') and args.use_sentiment:
        agent.sentiment_influence = args.sentiment_weight
        logger.info(f"Using sentiment in trading decisions with weight: {args.sentiment_weight}")
    
    # Load dates for test data
    stock_data = data_loader.fetch_stock_data()
    test_dates = stock_data.index[-len(X_test):]
    
    # Load sentiment data if available
    sentiment_data = None
    if args.use_sentiment:
        try:
            # Find all news sentiment columns in the stock data
            sentiment_cols = [col for col in stock_data.columns if isinstance(col, str) and col.startswith('news_')]
            if sentiment_cols:
                sentiment_data = stock_data[sentiment_cols].copy()
                logger.info(f"Loaded {len(sentiment_cols)} sentiment features for trading decisions")
        except Exception as e:
            logger.warning(f"Could not load sentiment data for trading: {e}")
    
    try:
        final_value, roi = agent.run_simulation(
            test_data=X_test,
            test_dates=test_dates,
            feature_scaler=preprocessor.feature_scaler,
            sentiment_data=sentiment_data,
            final_trade=True
        )
        
        # Ensure all plots are properly closed
        close_all_figures()
        
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
        # Make sure to close figures even if there's an error
        close_all_figures()
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
    
    # Fetch news data if API key is provided and sentiment is enabled
    sentiment_data = None
    if args.newsapi_key and args.use_sentiment:
        # Use a smaller window for news data to respect API limits
        logger.info(f"Fetching news data for trading (past {args.sentiment_days} days)")
        data_loader.fetch_news_data(days_back=args.sentiment_days)
    
    # Process data
    preprocessor = Preprocessor(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    
    preprocessor.prepare_data(stock_data)
    
    # If sentiment analysis is enabled, check for sentiment features
    if args.use_sentiment:
        sentiment_cols = [col for col in stock_data.columns if isinstance(col, str) and col.startswith('news_')]
        if sentiment_cols:
            sentiment_data = stock_data[sentiment_cols].copy()
            logger.info(f"Using {len(sentiment_cols)} sentiment features for trading recommendations")
    
    # Load model
    model = LSTMModel(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    if args.load_model:
        model_loaded = model.load_saved_model(args.load_model)
    else:
        model_loaded = model.load_saved_model()
    
    if not model_loaded:
        logger.error("Failed to load model. Cannot proceed with trading.")
        return None

    # Get most recent window of data for prediction
    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
    
    try:
        # Load training data to check feature dimensions
        X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
        
        # Extract model's expected feature dimension from loaded data
        expected_feature_dim = X_train.shape[2]
        logger.info(f"Model expects {expected_feature_dim} features based on training data")
        
        recent_window = X_train[-1:]  # Get the most recent window
        
        # Check if the feature dimensions match what the model expects
        # Get model's input shape correctly from the model itself rather than individual layers
        if model.model is not None:
            # Get input shape from model's configuration
            try:
                # Method 1: Try to get input shape from model config
                model_input_shape = model.model.input_shape
                if model_input_shape is not None:
                    model_feature_dim = model_input_shape[-1]
                    logger.info(f"Model input shape: {model_input_shape}, feature dimension: {model_feature_dim}")
                    # Adjust features if needed to match model's expectation
                    if recent_window.shape[2] != model_feature_dim:
                        logger.warning(f"Feature dimension mismatch: data has {recent_window.shape[2]} features, model expects {model_feature_dim}")
                        # If we have too many features, make an informed decision about which to truncate
                        if recent_window.shape[2] > model_feature_dim:
                            # Load feature names if available
                            feature_names = []
                            try:
                                processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
                                feature_names_path = os.path.join(processed_dir, "feature_names.npy")
                                priority_path = os.path.join(processed_dir, "priority_features.npy")
                                
                                if os.path.exists(feature_names_path):
                                    feature_names = np.load(feature_names_path, allow_pickle=True)
                                    
                                    # Check if we have priority features saved
                                    if os.path.exists(priority_path):
                                        priority_features = np.load(priority_path, allow_pickle=True)
                                        logger.info(f"Loaded priority features: {priority_features}")
                                        
                                    if len(feature_names) >= recent_window.shape[2]:
                                        # Identify candidates for removal - non-priority features at the end
                                        candidates_for_removal = []
                                        for i in range(len(feature_names)-1, -1, -1):  # Start from the end
                                            if feature_names[i] not in priority_features:
                                                candidates_for_removal.append(i)
                                                if len(candidates_for_removal) >= (recent_window.shape[2] - model_feature_dim):
                                                    break
                                        
                                        # If we found enough candidates for removal
                                        if len(candidates_for_removal) >= (recent_window.shape[2] - model_feature_dim):
                                            # Get the feature indices to keep
                                            features_to_remove = sorted(candidates_for_removal[:recent_window.shape[2] - model_feature_dim])
                                            features_to_remove_names = [feature_names[i] for i in features_to_remove]
                                            logger.warning(f"Removing non-priority features: {features_to_remove_names}")
                                            
                                            # Create a boolean mask for keeping features
                                            keep_mask = np.ones(recent_window.shape[2], dtype=bool)
                                            keep_mask[features_to_remove] = False
                                            
                                            # Use the mask to select features
                                            recent_window = recent_window[:, :, keep_mask]
                                            logger.info(f"Selectively removed {len(features_to_remove)} non-priority features")
                                        else:
                                            # Fallback: just truncate from the end
                                            truncated_features = feature_names[model_feature_dim:recent_window.shape[2]]
                                            logger.warning(f"Will truncate feature(s): {truncated_features}")
                                            logger.info(f"Last retained feature: {feature_names[model_feature_dim-1]}")
                                            recent_window = recent_window[:, :, :model_feature_dim]
                                    else:
                                        # Fallback if feature names don't match count
                                        logger.warning("Feature names length doesn't match data dimensions, truncating from end.")
                                        recent_window = recent_window[:, :, :model_feature_dim]
                                else:
                                    logger.warning("Feature names file not found. Truncating features without feature identification.")
                                    recent_window = recent_window[:, :, :model_feature_dim]
                            except Exception as e:
                                logger.warning(f"Could not load feature names: {e}")
                                # Default truncation if something goes wrong
                                recent_window = recent_window[:, :, :model_feature_dim]
                            logger.info(f"Final feature count after adjustment: {recent_window.shape[2]}")
                        # If we have too few features, we'd need to handle that case as well
                        elif recent_window.shape[2] < model_feature_dim:
                            logger.error(f"Data has fewer features than model expects: {recent_window.shape[2]} vs {model_feature_dim}")
                            raise ValueError(f"Insufficient features: model expects {model_feature_dim} but data has {recent_window.shape[2]}")
            except AttributeError:
                # Method 2: Try alternative way to get input shape
                try:
                    model_input_shape = model.model.get_config()['layers'][0]['config']['batch_input_shape']
                    if model_input_shape is not None:
                        model_feature_dim = model_input_shape[-1]
                        logger.info(f"Model input shape from config: {model_input_shape}, feature dimension: {model_feature_dim}")
                        # Handle feature dimension mismatch
                        if recent_window.shape[2] != model_feature_dim:
                            logger.warning(f"Feature dimension mismatch: data has {recent_window.shape[2]} features, model expects {model_feature_dim}")
                            # If we have too many features, make an informed decision about which to truncate
                            if recent_window.shape[2] > model_feature_dim:
                                # Load feature names if available
                                feature_names = []
                                try:
                                    processed_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
                                    feature_names_path = os.path.join(processed_dir, "feature_names.npy")
                                    if os.path.exists(feature_names_path):
                                        feature_names = np.load(feature_names_path, allow_pickle=True)
                                        if len(feature_names) >= recent_window.shape[2]:
                                            # Identify non-priority features to truncate
                                            truncated_features = feature_names[model_feature_dim:recent_window.shape[2]]
                                            logger.warning(f"Will truncate feature(s): {truncated_features}")
                                            logger.info(f"Last retained feature: {feature_names[model_feature_dim-1]}")
                                            recent_window = recent_window[:, :, :model_feature_dim]
                                        else:
                                            logger.warning("Feature names file not found. Truncating features without feature identification.")
                                            recent_window = recent_window[:, :, :model_feature_dim]
                                except Exception as e:
                                    logger.warning(f"Could not load feature names: {e}")
                                    # Default truncation if something goes wrong
                                    recent_window = recent_window[:, :, :model_feature_dim]
                            logger.info(f"Final feature count after adjustment: {recent_window.shape[2]}")
                        # If we have too few features, we'd need to handle that case as well
                        elif recent_window.shape[2] < model_feature_dim:
                            logger.error(f"Data has fewer features than model expects: {recent_window.shape[2]} vs {model_feature_dim}")
                            raise ValueError(f"Insufficient features: model expects {model_feature_dim} but data has {recent_window.shape[2]}")
                except Exception as e:
                    logger.warning(f"Could not determine model input shape from config: {e}")
                    logger.info(f"Using feature dimension from training data: {expected_feature_dim}")
                    recent_window = recent_window[:, :, :expected_feature_dim]
        else:
            logger.warning("Model input shape not found. Using feature dimension from training data.")
            recent_window = recent_window[:, :, :expected_feature_dim]
    except Exception as e:
        logger.error(f"Error adjusting feature dimensions: {e}")
        return None
    
    # Make predictions with error handling
    try:
        predictions, confidence_intervals = model.predict(recent_window, return_confidence=True)
        logger.info(f"Predicted prices for next {args.prediction_horizon} days: {predictions[0]}")
        
        # Log confidence intervals
        if args.confidence_intervals:
            lower_bounds = confidence_intervals['lower_95'][0]
            upper_bounds = confidence_intervals['upper_95'][0]
            for i in range(len(predictions[0])):
                logger.info(f"Day {i+1}: ${predictions[0][i]:.2f} (${lower_bounds[i]:.2f} - ${upper_bounds[i]:.2f})")
        
        # Scale predictions back to original prices
        if preprocessor.price_scaler is not None:
            try:
                # Reshape for inverse transform
                predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
                scaled_predictions = preprocessor.price_scaler.inverse_transform(predictions_reshaped)
                predictions = scaled_predictions.reshape(predictions.shape)
                
                # Get the current price from stock data to validate predictions
                if isinstance(stock_data.columns, pd.MultiIndex):
                    close_cols = [col for col in stock_data.columns if 'Close' in col[0]]
                    if close_cols:
                        current_price = float(stock_data[close_cols[0]].iloc[-1])
                else:
                    current_price = float(stock_data['Close'].iloc[-1])
                
                # Validate predictions are within a reasonable range (not more than 50% change)
                pred_min = np.min(predictions)
                pred_max = np.max(predictions)
                
                if pred_min < current_price * 0.5 or pred_max > current_price * 1.5:
                    logging.warning(f"Predictions may be unrealistic: range ${pred_min:.2f}-${pred_max:.2f} vs current ${current_price:.2f}")
                    logging.warning("Applying correction to predictions - using relative changes instead of absolute values")
                    
                    # Use relative changes from day to day rather than absolute values
                    # This helps when model predictions are on wrong scale but directional movement is correct
                    corrected_predictions = np.zeros_like(predictions[0])
                    
                    # Get predicted percentage changes between days
                    day_to_day_changes = []
                    for i in range(len(predictions[0])-1):
                        if predictions[0][i] != 0:  # Avoid division by zero
                            day_to_day_changes.append((predictions[0][i+1] / predictions[0][i]) - 1)
                        else:
                            day_to_day_changes.append(0)
                    day_to_day_changes.append(day_to_day_changes[-1] if day_to_day_changes else 0)  # Repeat last change for final day
                    
                    # Apply these percentage changes to the current price
                    corrected_predictions[0] = current_price * (1 + day_to_day_changes[0])
                    for i in range(1, len(corrected_predictions)):
                        corrected_predictions[i] = corrected_predictions[i-1] * (1 + day_to_day_changes[i-1])
                    
                    # Replace the original predictions with the corrected ones
                    predictions = np.array([corrected_predictions])
                    logging.info(f"Corrected predictions: {predictions[0]}")
            except Exception as e:
                logging.error(f"Error in prediction scaling: {e}")
                raise
        
        logging.info(f"Successfully predicted prices for next {args.prediction_horizon} days")
        
        # Extract the prediction array here to avoid the "referenced before assignment" error
        pred_prices = predictions[0]
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        
        # Fallback: use recent prices and add small random variation
        logger.warning("Using fallback prediction method based on recent prices")
        
        # Get last known price (convert to scalar)
        if 'Close' in stock_data.columns:
            last_price = float(stock_data['Close'].iloc[-1])
        else:
            # Try to find a Close column in a multi-index
            close_cols = [col for col in stock_data.columns if isinstance(col, tuple) and 'Close' in col[0]]
            if close_cols:
                last_price = float(stock_data[close_cols[0]].iloc[-1])
            else:
                # Last resort
                last_price = 200.0  # Default fallback value as float
        
        logger.info(f"Using last price: ${last_price:.2f} for fallback predictions")
        
        # Generate fallback predictions with small variations (±2%)
        random_variations = np.random.uniform(-0.02, 0.02, size=args.prediction_horizon)
        fallback_prices = np.array([last_price * (1 + var) for var in random_variations])
        predictions = np.array([fallback_prices])  # Make it 2D to match expected format
        logger.info(f"Using fallback predictions: {predictions[0]}")
        
        # Make sure to set pred_prices in the fallback case too
        pred_prices = predictions[0]
    
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
        if (next_day.weekday() < 5):  # Monday to Friday (0-4)
            next_days.append(next_day.strftime('%Y-%m-%d'))
            day_count += 1
            if day_count >= args.prediction_horizon:
                break
    
    # Get current sentiment if available
    current_sentiment = None
    if sentiment_data is not None and not sentiment_data.empty:
        try:
            # Get the most recent sentiment data point
            current_sentiment = sentiment_data.iloc[-1].to_dict()
            logger.info(f"Current sentiment: {current_sentiment}")
        except Exception as e:
            logger.warning(f"Could not retrieve current sentiment: {e}")
    
    # Create trading recommendations with sentiment consideration
    trading_agent = TradingAgent(
        model=model,
        price_scaler=preprocessor.price_scaler,
        initial_capital=args.initial_capital,
        transaction_fee=args.transaction_fee / 100,  # Convert percentage to decimal
        risk_factor=args.risk_factor,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        max_trades_per_day=args.max_trades_per_day,
        max_drawdown_pct=args.max_drawdown,
        volatility_scaling=args.volatility_scaling,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon  # Pass the prediction horizon explicitly
    )
    
    # Set sentiment influence if using sentiment
    if hasattr(trading_agent, 'sentiment_influence') and args.use_sentiment:
        trading_agent.sentiment_influence = args.sentiment_weight
    
    # Detect current market regime if requested
    current_regime = None
    regime_confidence = 0
    if args.detect_regimes and hasattr(model, 'current_market_regime'):
        try:
            current_regime, regime_confidence = model.current_market_regime(recent_window)
            if current_regime is not None:
                logger.info(f"Detected current market regime: {current_regime} (confidence: {regime_confidence:.2f})")
                
                # Adjust prediction horizon based on regime if adaptive horizon is enabled
                if args.adaptive_horizon and current_regime is not None:
                    # For high volatility regimes, use shorter prediction horizon
                    # For low volatility regimes, can use longer horizon
                    if current_regime == 0:  # Assuming 0 is high volatility regime
                        effective_horizon = min(3, args.prediction_horizon)
                        logger.info(f"High volatility regime detected - using shorter horizon ({effective_horizon} days)")
                    elif current_regime == 1:  # Assuming 1 is medium volatility
                        effective_horizon = args.prediction_horizon
                        logger.info(f"Medium volatility regime - using standard horizon ({effective_horizon} days)")
                    else:  # Low volatility regime
                        effective_horizon = args.prediction_horizon
                        logger.info(f"Low volatility regime - using standard horizon ({effective_horizon} days)")
                        
                    # Truncate predictions to effective horizon
                    pred_prices = predictions[0][:effective_horizon]
                    next_days = next_days[:effective_horizon]
                    logger.info(f"Adjusted prediction horizon to {effective_horizon} days based on current market regime")
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
    
    # Calculate min/max price days BEFORE the recommendation loop
    pred_prices = predictions[0]
    min_price_day = np.argmin(pred_prices)
    max_price_day = np.argmax(pred_prices)
    min_price = pred_prices[min_price_day]
    max_price = pred_prices[max_price_day]
    
    # Make sure min_price_day comes before max_price_day to follow buy-low, sell-high strategy
    valid_strategy = min_price_day < max_price_day
    potential_gain_pct = ((max_price - min_price) / min_price * 100) if valid_strategy else 0
    
    # Make recommendation logic more sophisticated
    recommendations = []
    
    # Calculate day-to-day price changes (not just from current price)
    day_to_day_changes = []
    for i in range(1, len(pred_prices)):
        if pred_prices[i-1] != 0:
            change = (pred_prices[i] - pred_prices[i-1]) / pred_prices[i-1] * 100
            day_to_day_changes.append(change)
        else:
            day_to_day_changes.append(0)
    
    # Prepend first day change (from current to first prediction)
    first_day_change = (pred_prices[0] - current_price) / current_price * 100
    day_to_day_changes = [first_day_change] + day_to_day_changes
    
    # Calculate momentum (acceleration of price changes)
    momentum = []
    for i in range(1, len(day_to_day_changes)):
        momentum.append(day_to_day_changes[i] - day_to_day_changes[i-1])
    
    # Prepend first momentum (just use first change)
    momentum = [day_to_day_changes[0]] + momentum
    
    # Get overall trend direction
    avg_change = sum(day_to_day_changes) / len(day_to_day_changes)
    trend_direction = "uptrend" if avg_change > 1 else "downtrend" if avg_change < -1 else "sideways"
    logging.info(f"Detected market trend: {trend_direction} with average daily change of {avg_change:.2f}%")

    # Simplified recommendation logic: only buy on min_price_day, only sell on max_price_day
    for i in range(args.prediction_horizon):
        pred_price = predictions[0][i]
        price_change = (pred_price - current_price) / current_price * 100
        
        # Get day-specific momentum
        daily_momentum = momentum[i]
        # Day-to-day change
        day_change = day_to_day_changes[i]
        
        # Calculate risk factor based on prediction distance and momentum
        days_ahead = i + 1
        forecast_risk = days_ahead * 0.8  # Risk increases with prediction distance
        
        # Adjust price change expectation based on sentiment if available
        sentiment_adjustment = 0
        if current_sentiment and hasattr(trading_agent, 'sentiment_influence'):
            if 'news_weighted_sentiment' in current_sentiment:
                sentiment_value = current_sentiment['news_weighted_sentiment']
                sentiment_adjustment = sentiment_value * args.sentiment_weight * 5  # Scale appropriately
                logging.info(f"Day {i+1} sentiment adjustment: {sentiment_adjustment:.2f}%")
            
        adjusted_price_change = price_change + sentiment_adjustment
        
        # Simple recommendation logic based on the optimal trading strategy
        if valid_strategy:
            if i == min_price_day:
                recommendation = "BUY"
                confidence = "High"
            elif i == max_price_day:
                recommendation = "SELL"
                confidence = "High"
            else:
                recommendation = "HOLD"
                confidence = "Medium"
        else:
            # If min price is after max price (invalid strategy), use a fallback approach 
            # based on price movement relative to current price
            if adjusted_price_change > 3 and i == 0:
                recommendation = "BUY"  # Buy immediately if first day looks good
                confidence = "Medium"
            elif adjusted_price_change < -3 and i == 0:
                recommendation = "SELL"  # Sell immediately if first day looks bad
                confidence = "Medium"
            else:
                recommendation = "HOLD"
                confidence = "Low"
            
        recommendations.append({
            'date': next_days[i],
            'predicted_price': pred_price,
            'raw_change_percent': price_change,
            'day_to_day_change': day_change,
            'momentum': daily_momentum,
            'sentiment_adjustment': sentiment_adjustment,
            'adjusted_change': adjusted_price_change,
            'recommendation': recommendation,
            'confidence': confidence
        })
    
    # Add optimal day analysis to recommendations
    optimal_strategy = {
        'date': 'Optimal Strategy',
        'min_price_day': min_price_day + 1,  # Convert to 1-indexed for display
        'min_price': min_price,
        'max_price_day': max_price_day + 1,  # Convert to 1-indexed for display
        'max_price': max_price,
        'potential_gain_pct': potential_gain_pct,
        'strategy': 'Buy on day {} and sell on day {}'.format(
            min_price_day + 1, max_price_day + 1
        ) if valid_strategy else 'No clear buy-sell opportunity'
    }
    
    # Calculate potential portfolio value after optimal trade (assuming we use all capital)
    if valid_strategy:
        transaction_fee = args.transaction_fee / 100  # Convert from percentage to decimal
        shares_bought = args.initial_capital / min_price
        sell_value = shares_bought * max_price * (1 - transaction_fee)  # Account for transaction fee when selling
        portfolio_increase = sell_value - args.initial_capital
        roi_percent = (portfolio_increase / args.initial_capital) * 100
        
        logging.info(f"Optimal trade would yield ${portfolio_increase:.2f} profit ({roi_percent:.2f}% ROI)")
    
    print("\nOptimal Trading Strategy:")
    if valid_strategy:
        print(f"Buy on day {optimal_strategy['min_price_day']} at ${optimal_strategy['min_price']:.2f}")
        print(f"Sell on day {optimal_strategy['max_price_day']} at ${optimal_strategy['max_price']:.2f}")
        print(f"Potential gain: {optimal_strategy['potential_gain_pct']:.2f}%")
        print(f"Portfolio value would increase from ${args.initial_capital:.2f} to ${sell_value:.2f} " +
              f"(${portfolio_increase:.2f} profit, {roi_percent:.2f}% ROI)")
    else:
        print("No clear buy-low, sell-high opportunity in the prediction window.")
    
    # Display recommendations
    rec_df = pd.DataFrame(recommendations)
    print("\nTrading Recommendations for Next 5 Trading Days:")
    print(rec_df[['date', 'predicted_price', 'day_to_day_change', 'recommendation', 'confidence']].to_string(index=False))
    
    # Save recommendations
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    rec_df.to_csv(os.path.join(results_dir, f'recommendations_{datetime.now().strftime("%Y%m%d")}.csv'), index=False)
    
    # Visualize predictions with confidence intervals if available
    plt.figure(figsize=(10, 6))
    
    # Plot historical data (last 30 days)
    hist_dates = stock_data.index[-30:]
    hist_prices = stock_data['Close'][-30:].values if 'Close' in stock_data.columns else None
    
    if hist_prices is not None:
        plt.plot(range(len(hist_dates)), hist_prices, 'b-', label='Historical')
    
    # Plot raw predictions
    pred_prices = predictions[0]
    plt.plot(range(len(hist_dates), len(hist_dates) + len(pred_prices)), 
             pred_prices, 'r--', label='Predicted')
    
    # Plot confidence intervals if available
    if args.confidence_intervals and 'confidence_intervals' in locals():
        lower_bounds = confidence_intervals['lower_95'][0]
        upper_bounds = confidence_intervals['upper_95'][0]
        
        x_range = range(len(hist_dates), len(hist_dates) + len(pred_prices))
        plt.fill_between(
            x_range, 
            lower_bounds, 
            upper_bounds, 
            color='gray', 
            alpha=0.3, 
            label='95% Confidence'
        )
    
    # Highlight optimal buy/sell points
    if min_price_day < max_price_day:
        plt.scatter(len(hist_dates) + min_price_day, min_price, color='green', marker='^', s=120, label='Optimal Buy')
        plt.scatter(len(hist_dates) + max_price_day, max_price, color='red', marker='v', s=120, label='Optimal Sell')
    
    # Set x-axis labels
    all_dates = np.concatenate([hist_dates.strftime('%Y-%m-%d').values if hist_dates is not None else [], next_days])
    plt.xticks(range(0, len(all_dates), 5), all_dates[::5], rotation=45)
    
    plt.title('Tesla Stock Price Prediction with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, f'prediction_chart_{datetime.now().strftime("%Y%m%d")}.png'))
    plt.close()
    
    return rec_df

def plot_predictions_with_confidence(actual, predicted, confidence_intervals, scaler=None):
    """Plot predictions with confidence intervals"""
    if confidence_intervals is None:
        return
        
    # Transform data if scaler is provided
    if scaler:
        # Reshape for inverse transform if necessary
        if len(actual.shape) > 1:
            actual_reshaped = actual.reshape(-1, actual.shape[-1])
            predicted_reshaped = predicted.reshape(-1, predicted.shape[-1])
            lower_reshaped = confidence_intervals['lower_95'].reshape(-1, confidence_intervals['lower_95'].shape[-1])
            upper_reshaped = confidence_intervals['upper_95'].reshape(-1, confidence_intervals['upper_95'].shape[-1])
        else:
            actual_reshaped = actual.reshape(-1, 1)
            predicted_reshaped = predicted.reshape(-1, 1)
            lower_reshaped = confidence_intervals['lower_95'].reshape(-1, 1)
            upper_reshaped = confidence_intervals['upper_95'].reshape(-1, 1)
            
        # Transform back to original scale
        actual_orig = scaler.inverse_transform(actual_reshaped)
        predicted_orig = scaler.inverse_transform(predicted_reshaped)
        lower_orig = scaler.inverse_transform(lower_reshaped)
        upper_orig = scaler.inverse_transform(upper_reshaped)
        
        # Reshape back if necessary
        if len(actual.shape) > 1:
            actual_values = actual_orig.reshape(actual.shape)
            predicted_values = predicted_orig.reshape(predicted.shape)
            lower_bounds = lower_orig.reshape(confidence_intervals['lower_95'].shape)
            upper_bounds = upper_orig.reshape(confidence_intervals['upper_95'].shape)
        else:
            actual_values = actual_orig.flatten()
            predicted_values = predicted_orig.flatten()
            lower_bounds = lower_orig.flatten()
            upper_bounds = upper_orig.flatten()
    else:
        actual_values = actual
        predicted_values = predicted
        lower_bounds = confidence_intervals['lower_95']
        upper_bounds = confidence_intervals['upper_95']
    
    # Create the plot
    plt.figure(figsize=(14, 7))
    
    # Plot for first day prediction with confidence intervals
    plt.subplot(1, 2, 1)
    x = np.arange(len(actual_values))
    plt.plot(x, actual_values[:, 0], 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted_values[:, 0], 'r--', label='Predicted', linewidth=2)
    
    # Add confidence intervals shading
    plt.fill_between(
        x, 
        lower_bounds[:, 0], 
        upper_bounds[:, 0], 
        color='red', 
        alpha=0.2, 
        label='95% Confidence'
    )
    
    plt.title('1-Day Ahead Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot for last day prediction with confidence intervals
    plt.subplot(1, 2, 2)
    plt.plot(x, actual_values[:, -1], 'b-', label='Actual', linewidth=2)
    plt.plot(x, predicted_values[:, -1], 'r--', label='Predicted', linewidth=2)
    
    # Add confidence intervals shading
    plt.fill_between(
        x, 
        lower_bounds[:, -1], 
        upper_bounds[:, -1], 
        color='red', 
        alpha=0.2, 
        label='95% Confidence'
    )
    
    plt.title(f'{actual_values.shape[1]}-Day Ahead Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(
        os.path.dirname(__file__), 
        "results", 
        f'predictions_with_confidence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    ))
    plt.close()

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
        elif args.mode == 'visualize-metrics':
            # Add a new mode for visualizing metrics history
            visualizer = MetricsVisualizer()
            metrics_path = visualizer.plot_training_history(n_iterations=args.num_iterations if hasattr(args, 'num_iterations') else 10)
            logger.info(f"Metrics history visualization created: {metrics_path}")
        else:
            logger.error(f"Unknown mode: {args.mode}")
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
    finally:
        # Ensure all matplotlib figures are closed properly to prevent Tkinter errors
        close_all_figures()
        
    return 0

if __name__ == "__main__":
    exit(main())
