from bmkg_fetch import fetch_bmkg
from preprocessing import preprocess
from lstm_model import build_lstm
from cnn_model import build_cnn
from utils import label_banjir
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. ambil data
df = fetch_bmkg()

# 2. preprocessing
scaled, scaler = preprocess(df)

# 3. windowing
from preprocessing import create_sequence
X, y = create_sequence(scaled, 3)

# 4. LSTM
lstm = build_lstm((3,4))
lstm.fit(X, y, epochs=10)

# 5. output LSTM
lstm_output = lstm.predict(X)

# 6. reshape ke CNN
X_cnn = lstm_output.reshape((lstm_output.shape[0],1,1,1))

# 7. label banjir
y_cnn = np.array([label_banjir(v) for v in lstm_output.flatten()])
y_cnn = to_categorical(y_cnn)

# 8. CNN
cnn = build_cnn()
cnn.fit(X_cnn, y_cnn, epochs=10)

# 9. prediksi akhir
pred = cnn.predict(X_cnn)

print("Status prediksi:", pred[-1])