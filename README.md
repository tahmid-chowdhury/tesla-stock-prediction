# Tesla Stock Prediction

This project aims to predict Tesla's stock movements using an LSTM (Long Short-Term Memory) machine learning model. The model is trained on historical stock price data and can be used to forecast future price movements.

## Project Structure

```
tesla-stock-prediction
├── data
│   ├── raw                # Directory for raw stock data (CSV files)
│   └── processed          # Directory for processed data ready for modeling
├── src
│   ├── model.py           # Defines the LSTM model architecture
│   ├── train.py           # Responsible for training the LSTM model
│   └── predict.py         # Used for making predictions with the trained model
├── utils
│   └── data_preprocessing.py # Contains utility functions for data preprocessing
├── requirements.txt       # Lists the required Python dependencies
└── README.md              # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd tesla-stock-prediction
   ```

2. **Create a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

- Place the raw stock data files in the `data/raw` directory.
- Run the data preprocessing script to clean and prepare the data for training:
  ```
  python -m utils.data_preprocessing
  ```

## Training the Model

- To train the LSTM model, execute the following command:
  ```
  python src/train.py
  ```

## Making Predictions

- After training, you can make predictions using the trained model:
  ```
  python src/predict.py
  ```

## Trading Strategy

To execute the trading strategy using the model's predictions:
```
python src/trading_strategy.py
```

Ensure that you have placed the latest raw market data (with a 'close' column) in the `data/raw/latest_data.csv` file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.