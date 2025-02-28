# Tesla Stock Price Prediction & Trading Agent

This project implements a machine learning-based trading agent that predicts Tesla stock movements and makes trading decisions in a simulated environment. The simulation covers the trading days from March 24-28, 2025.

## Project Overview

The trading agent uses deep learning (LSTM) to analyze historical Tesla stock data and make daily trading decisions (Buy, Sell, or Hold) with the goal of maximizing the final account balance.

## Trading Rules

- Simulation Period: March 24-28, 2025 (5 trading days)
- Trading Schedule:
  - 9:00 AM (EST): Generate trading advice
  - 10:00 AM (EST): Execute orders at current price
- Starting Capital: $10,000 USD
- Transaction Fee: 1% per Buy/Sell order

## Technical Indicators Used

The model analyzes the following technical indicators:
- Moving Averages (5, 10, 20 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volatility
- Price Returns

## Trading Strategy

The trading strategy combines multiple approaches:

1. **Price Prediction**: LSTM model predicts next-day closing prices
2. **Technical Analysis**: Using indicators like RSI, MACD, and Bollinger Bands
3. **Risk Management**: Adjusts position size based on volatility
4. **Decision Rules**:
   - Buy when predicted price increase > 2%
   - Sell when predicted price decrease > 2%
   - Hold otherwise
   - Transaction size based on risk factor (0.1-0.5 of available capital)

## Project Structure

```
/tesla-stock-prediction/
├── data/
│   ├── raw/        # Raw stock data
│   └── processed/  # Processed features
├── models/         # Trained ML models
├── src/
│   ├── data/       # Data preprocessing
│   ├── models/     # ML model definition
│   ├── agent/      # Trading agent logic
│   └── evaluation/ # Backtesting framework
├── results/        # Simulation results
├── main.py         # Main execution script
└── README.md       # Project documentation
```

## Model Architecture

The prediction model uses a stacked LSTM neural network:
- Input: Sequence of 10 days of technical indicators
- Architecture: 2 LSTM layers (50 units each) with dropout
- Output: Next-day predicted price

## Performance Analysis

The trading agent's performance is evaluated based on:
1. Final portfolio value
2. Total return