from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Kembali menggunakan MSE (Permintaanmu)
    # Ditahan dengan Learning Rate 0.00005 agar penurunan loss stabil dan melandai halus
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='mse'
    )
    return model
