# Tesla Stock Price Prediction & Trading Agent

**Course**: SOFE 4620U - Machine Learning and Data Mining  

**Group ML5 Members**:  
Hamzi Farhat 100831450  
Jason Stuckless 100248154  
Tahmid Chowdhury 100822671 

## Overview

This project implements a machine learning-based trading agent that analyzes historical Tesla stock data, predicts stock movements, and makes trading decisions (Buy, Sell, or Hold) for the next 5 trading days in a simulated environment, with the goal of maximizing the final account balance. The trading agent uses LSTM deep learning to analyze stock data pulled from Yahoo Finance.

## Key Features
- **LSTM Deep Learning**: Utilizes long short-term memory neural networks for time series prediction
- **Technical Indicators**: Incorporates market indicators like RSI, MACD, and Bollinger Bands
- **News Sentiment**: Optionally includes news sentiment analysis for Tesla (requires NewsAPI key)
- **Trading Simulation**: Simulates trades with realistic transaction fees and constraints
- **Comprehensive Evaluation**: Measures both prediction accuracy and trading performance
- **Market Regime Detection**: Identifies different market conditions for optimized trading strategies
- **Ensemble Models**: Option to use multiple models for more robust predictions
- **JSON Data Resilience**: Automatic repair of corrupted JSON files preserves training history
- **Enhanced Visualization**: Advanced metrics tracking with historical performance comparison

## How It Works
1. **Data Collection**:
  - Historical Tesla stock data is fetched from Yahoo Finance
  - Optional news data is collected from NewsAPI
2. **Data Preprocessing**:
  - Technical indicators are calculated from raw price data
  - Features are normalized for model training
  - Sliding window approach creates sequential data for LSTM
3. **Model Training**:
  - LSTM model is trained to predict closing prices for the next 5 trading days
  - Early stopping prevents overfitting
  - Best model is saved for inference
4. **Trading Simulation**:
  - Agent decides to buy, sell, or hold based on predicted price movements
  - Transaction fees are applied to simulate realistic trading
  - Performance metrics track profit/loss and ROI
5. **Evaluation**:
  - Price prediction accuracy is measured using RMSE and directional accuracy
  - Trading performance is evaluated based on final portfolio value and ROI

## Trading Rules

- Simulation Period: Next 5 trading days from when the program is run
- Trading Schedule:
  1. Generate trading advice
  2. Execute orders at current price
- Starting Capital: $10,000 USD
- Transaction Fee: 1% per Buy/Sell order

## Performance Metrics Used

The model is trained based on the following performance metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Trading Strategy

The trading strategy combines multiple approaches:

1. **Price Prediction**: LSTM model predicts closing prices for the next 5 days
2. **Training Analysis**: Using indicators like the F1-Score, the model is iterated and trained
3. **Risk Management**: Leverages a sliding window of the past 30 trading days to capture recent market behaviour effectively, while smoothing out some noise without losing sensitivity based on recent volatility of the stock
4. **Decision Rules**:
   - Buy when predicted price increase > 2%
   - Sell when predicted price decrease > 2%
   - Hold otherwise
   - Transaction size based on risk factor (0.1-0.5 of available capital)
5. **Market Regime Adaptation**: Adjusts strategies based on detected market conditions (high/low volatility)
6. **Confidence-Based Sizing**: Uses prediction confidence intervals to adjust position sizing

## Results

The program outputs a CSV file and several PNG images with evaluation metrics:
- **Performance Metrics**: Accuracy, precision, recall, and F1-scores for the current model iteration
- **Trading Decisions**: Visualization of buy/sell points with stock price
- **Portfolio Value**: Chart showing portfolio value over time during simulation
- **Performance History**: Tracks model improvements across training iterations

## Project Structure

```
/tesla-stock-prediction/
├── data/
│   ├── raw/         # Raw stock data
│   └── processed/   # Processed features
├── models/          # Trained ML model
├── src/
│   ├── data/        # Data preprocessing
│   ├── models/      # ML model definition
│   ├── agent/       # Trading agent logic
│   ├── evaluation/  # Backtesting framework
│   ├── utils/       # Utility functions
│   └── visualization/ # Visualization tools
├── results/         # Simulation results
├── main.py          # Main execution script
├── README.md        # Project documentation
└── requirements.txt # Project requirements
```

## Performance Analysis

The trading agent's performance is evaluated based on:
1. Final portfolio value
2. Total return
3. Direction accuracy of predictions
4. Risk-adjusted returns
5. Maximum drawdown

## Requirements

### Libraries

- Python 3.10+
- `tensorflow` & `keras` for deep learning
- `scikit-learn` for evaluation and preprocessing
- `pandas` & `numpy` for data manipulation
- `matplotlib` & `seaborn` for visualization
- `yfinance` for stock data retrieval
- `newsapi-python` for news data (optional)
- `ta` for technical indicators
- `tqdm` for progress tracking
- `keras-tuner` for hyperparameter optimization

### Input Data Format

CSV file from Yahoo Finance named `TSLA.csv` with columns:
- `Date`
- `Close`
- `High`
- `Low`
- `Open`
- `Volume`

## Usage
1. **Virtual Environment Setup**:
   ```
   # Create a virtual environment using Python 3.10
   python3.10 -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   # source venv/bin/activate
   ```

2. **Installation**: Install dependencies with `pip install -r requirements.txt`
3. **Training**: Run `python main.py --mode train` to train the model
4. **Testing**: Run `python main.py --mode test` to test on historical data
5. **Trading**: Run `python main.py --mode trade` to get recommendations for the next 5 trading days
6. **Metrics Visualization**: Run `python main.py --mode visualize-metrics` to see performance across training iterations
7. **Optional Parameters**:
   - `--window-size`: Size of sliding window in days (default: 30)
   - `--prediction-horizon`: Days to predict ahead (default: 5)
   - `--initial-capital`: Starting investment amount (default: 10000)
   - `--transaction-fee`: Fee per transaction as percentage (default: 0.01)
   - `--risk-factor`: Trading risk factor between 0.1-0.5 (default: 0.3)
   - `--newsapi-key`: API key for NewsAPI (optional)
   - `--load-model`: Path to pre-trained model file (optional)
   - `--use-complete-history`: Use the entire available historical data for training
   - `--use-ensemble`: Use ensemble of models for better prediction stability
   - `--ensemble-size`: Number of models in ensemble (default: 3)
   - `--detect-regimes`: Detect and use market regimes for predictions
   - `--confidence-intervals`: Generate confidence intervals for predictions
   - `--adaptive-horizon`: Adjust prediction horizon based on market conditions
   - `--simplified-architecture`: Use simpler model for faster training

## Additional Features

### Error Resilience
- **JSON Repair Utility**: Automatically detects and repairs corrupted JSON files, ensuring training history is preserved
- **Warning Suppression**: Cleanly handles TensorFlow and dependency warnings for a clearer console output

### Performance Tracking
- **Metrics Visualization**: Track model performance across multiple training iterations
- **Comprehensive Metrics**: Monitor not just price accuracy but trading performance indicators

### Market Adaptation
- **Regime Detection**: Identify distinct market regimes (bull/bear/sideways) to apply specialized strategies
- **Adaptive Position Sizing**: Adjust trade sizes based on market volatility and prediction confidence