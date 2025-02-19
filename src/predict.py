from keras.models import load_model
import numpy as np
import pandas as pd
from utils.data_preprocessing import preprocess_data

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def make_prediction(model, input_data):
    input_data = np.array(input_data).reshape((1, input_data.shape[0], input_data.shape[1]))
    prediction = model.predict(input_data)
    return prediction

def main():
    model_path = 'path/to/your/trained/model.h5'  # Update with the actual path to your model
    model = load_trained_model(model_path)

    # Load and preprocess new input data
    new_data = pd.read_csv('path/to/your/new/data.csv')  # Update with the actual path to your new data
    processed_data = preprocess_data(new_data)

    # Make predictions
    predictions = make_prediction(model, processed_data)
    print(predictions)

if __name__ == "__main__":
    main()