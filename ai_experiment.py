import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

class Brain:
    def __init__(self):
        self.model = None
        self.data = None
        self.scaler = None

    def load_data(self):
        if not os.path.exists('data.csv'):
            print('Error: Data file not found.')
            return False
        self.data = pd.read_csv('data.csv')
        return True

    def preprocess_data(self):
        if self.data is None:
            print('Error: Data not loaded.')
            return False
        self.scaler = MinMaxScaler()
        self.data[['feature1', 'feature2', 'feature3']] = self.scaler.fit_transform(self.data[['feature1', 'feature2', 'feature3']])
        return True

    def build_model(self):
        if self.data is None or self.scaler is None:
            print('Error: Data or scaler not loaded.')
            return False
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.data.shape[0], 3)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return True

    def train_model(self):
        if self.model is None or self.data is None:
            print('Error: Model or data not built.')
            return False
        self.model.fit(self.data[['feature1', 'feature2', 'feature3']], self.data['target'], epochs=100, batch_size=32, verbose=0)
        return True

    def make_prediction(self):
        if self.model is None:
            print('Error: Model not built.')
            return False
        prediction = self.model.predict(self.data[['feature1', 'feature2', 'feature3']])
        return prediction