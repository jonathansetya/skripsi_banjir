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
# Data langsung diambil, teks 'Status API: 200' dan tabel
# akan otomatis dicetak 1 kali oleh fungsi ini di awal.
df = fetch_bmkg()

# Catatan: Baris print(df.head()) di sini sudah dihapus
# agar tidak terjadi duplikasi cetak di terminal.

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

# gunakan data rain asli (BELUM scaling)
rain_data = df["rain"].values

# looping sesuai jumlah sequence
for i in range(3, len(rain_data)):

    rain = float(rain_data[i])

    print(f"Rain Data Index {i}: {rain}")

    label = label_banjir(rain)

    print(f"Label: {label}")

    y_cnn.append(label)

y_cnn = np.array(y_cnn)

print("\nShape Label Sebelum Encoding:", y_cnn.shape)
print("Isi Label:", y_cnn)

# cek distribusi label
unique, counts = np.unique(y_cnn, return_counts=True)

print("\nDistribusi Label:")
print(dict(zip(unique, counts)))

# one-hot encoding
y_cnn = to_categorical(y_cnn, num_classes=3)

print("\nShape Label Sesudah Encoding:", y_cnn.shape)

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

# =========================
# PREDIKSI CLASS
# =========================
y_pred = np.argmax(pred, axis=1)

# =========================
# LABEL ASLI
# =========================
y_true = np.argmax(y_cnn, axis=1)

# =========================
# EVALUATION
# =========================
from evaluation import evaluate_model

evaluate_model(y_true, y_pred)
