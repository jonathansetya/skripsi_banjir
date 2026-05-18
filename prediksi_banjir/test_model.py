import tensorflow as tf
import numpy as np
import random
import os

# =========================
# FIX RANDOM RESULT
# =========================
os.environ['PYTHONHASHSEED'] = '42'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# IMPORT LIBRARY
# =========================
from tensorflow.keras.utils import to_categorical

from bmkg_fetch import fetch_bmkg
from preprocessing import preprocess, create_sequence
from lstm_model import build_lstm
from cnn_model import build_cnn
from utils import label_banjir

# =========================
# AMBIL DATA BMKG
# =========================
df = fetch_bmkg()

print("\n=== DATA BMKG ===")
print(df.head())

# =========================
# PREPROCESSING
# =========================
scaled = preprocess(df)

X, y = create_sequence(scaled, 3)

print("\nShape X:", X.shape)
print("Shape y:", y.shape)

# =========================
# LSTM
# =========================
lstm = build_lstm((3,5))

print("\n=== TRAINING LSTM ===")

lstm.fit(X, y, epochs=10, verbose=1)

lstm_out = lstm.predict(X)

print("\nShape LSTM Output:", lstm_out.shape)

# =========================
# CNN INPUT
# =========================
X_cnn = lstm_out.reshape((lstm_out.shape[0],1,1,1))

# =========================
# LABEL CNN
# =========================
y_cnn = []

for i in range(len(X)):
    rain = float(X[i][-1][-1])

    label = label_banjir(rain)

    y_cnn.append(label)

y_cnn = np.array(y_cnn)

print("\nShape Label Sebelum Encoding:", y_cnn.shape)

y_cnn = to_categorical(y_cnn, num_classes=3)

print("Shape Label Sesudah Encoding:", y_cnn.shape)

# =========================
# CNN
# =========================
cnn = build_cnn()

print("\n=== TRAINING CNN ===")

cnn.fit(X_cnn, y_cnn, epochs=10, verbose=1)

# =========================
# PREDIKSI
# =========================
pred = cnn.predict(X_cnn)

print("\n=== HASIL PREDIKSI ===")

hasil = pred[-1]

aman = hasil[0] * 100
waspada = hasil[1] * 100
bahaya = hasil[2] * 100

print(f"Aman     : {aman:.2f}%")
print(f"Waspada  : {waspada:.2f}%")
print(f"Bahaya   : {bahaya:.2f}%")

status = np.argmax(hasil)

if status == 0:
    print("\nSTATUS BANJIR: AMAN 🟢")

elif status == 1:
    print("\nSTATUS BANJIR: WASPADA 🟡")

else:
    print("\nSTATUS BANJIR: BAHAYA 🔴")