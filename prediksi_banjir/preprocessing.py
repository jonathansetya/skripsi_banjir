from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess(df):
    df['weather'] = df['weather'].astype('category').cat.codes

    features = df[['temp', 'humidity', 'wind', 'weather', 'rain']]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    return scaled

def create_sequence(data, window=3):
    X, y = [], []

    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window][1])  # humidity

    return np.array(X), np.array(y)