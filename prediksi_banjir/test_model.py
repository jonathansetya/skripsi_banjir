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
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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
print(df.to_string())

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
lstm = build_lstm((3, 5))

print("\n=== TRAINING LSTM ===")

history_lstm = lstm.fit(
    X,
    y,
    epochs=10,
    verbose=1
)

# =========================
# OUTPUT LSTM
# =========================
lstm_out = lstm.predict(X)

print("\nShape LSTM Output:", lstm_out.shape)

# =========================
# GRAFIK LOSS LSTM
# =========================
plt.figure(figsize=(8, 4))

plt.plot(
    history_lstm.history['loss'],
    marker='o',
    color='blue',
    label='Training Loss'
)

plt.title('Grafik Loss Pelatihan Model LSTM')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.xticks(range(10), range(1, 11))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('grafik_loss_lstm.png', dpi=150)
plt.show()

print("Grafik loss LSTM tersimpan.")

# =========================
# RMSE DAN MAE LSTM
# =========================
y_pred_lstm = lstm.predict(X)

rmse_lstm = np.sqrt(
    mean_squared_error(
        y.flatten(),
        y_pred_lstm.flatten()
    )
)

mae_lstm = mean_absolute_error(
    y.flatten(),
    y_pred_lstm.flatten()
)

print("\n=== METRIK EVALUASI LSTM ===")
print(f"RMSE : {rmse_lstm:.4f}")
print(f"MAE  : {mae_lstm:.4f}")

# =========================
# CNN INPUT
# =========================
X_cnn = lstm_out.reshape(
    (lstm_out.shape[0], 1, 1, 1)
)

# =========================
# LABEL CNN
# =========================
y_cnn = []

rain_data = df["rain"].values

for i in range(3, len(rain_data)):

    rain = float(rain_data[i])

    print(f"Rain Data Index {i}: {rain}")

    label = label_banjir(rain)

    print(f"Label: {label}")

    y_cnn.append(label)

y_cnn = np.array(y_cnn)

print("\nShape Label Sebelum Encoding:", y_cnn.shape)
print("Isi Label:", y_cnn)

# =========================
# DISTRIBUSI LABEL
# =========================
unique, counts = np.unique(
    y_cnn,
    return_counts=True
)

print("\nDistribusi Label:")
print(dict(zip(unique, counts)))

# =========================
# SIMPAN LABEL ASLI
# =========================
y_true = y_cnn.copy()

# =========================
# ONE HOT ENCODING
# =========================
y_cnn = to_categorical(
    y_cnn,
    num_classes=3
)

print("\nShape Label Sesudah Encoding:", y_cnn.shape)

# =========================
# CNN
# =========================
cnn = build_cnn()

print("\n=== TRAINING CNN ===")

history_cnn = cnn.fit(
    X_cnn,
    y_cnn,
    epochs=10,
    verbose=1
)

# =========================
# GRAFIK ACCURACY CNN
# =========================
plt.figure(figsize=(8, 4))

plt.plot(
    history_cnn.history['accuracy'],
    marker='s',
    color='green',
    label='Training Accuracy'
)

plt.title('Grafik Accuracy Pelatihan Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(10), range(1, 11))
plt.ylim(0, 1.1)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    'grafik_accuracy_cnn.png',
    dpi=150
)

plt.show()

# =========================
# GRAFIK LOSS CNN
# =========================
plt.figure(figsize=(8, 4))

plt.plot(
    history_cnn.history['loss'],
    marker='s',
    color='red',
    label='Training Loss'
)

plt.title('Grafik Loss Pelatihan Model CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(10), range(1, 11))

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    'grafik_loss_cnn.png',
    dpi=150
)

plt.show()

print("Grafik CNN tersimpan.")

# =========================
# PREDIKSI CNN
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
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n=== CONFUSION MATRIX ===")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[
        'AMAN',
        'WASPADA',
        'BAHAYA'
    ],
    yticklabels=[
        'AMAN',
        'WASPADA',
        'BAHAYA'
    ]
)

plt.title(
    'Confusion Matrix Model Hybrid LSTM-CNN'
)

plt.ylabel('Kelas Aktual')
plt.xlabel('Kelas Prediksi')

plt.tight_layout()

plt.savefig(
    'confusion_matrix.png',
    dpi=150
)

plt.show()

# =========================
# METRIK CNN
# =========================
acc = accuracy_score(
    y_true,
    y_pred
)

prec = precision_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0
)

rec = recall_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0
)

print("\n=== METRIK EVALUASI CNN ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            'AMAN',
            'WASPADA',
            'BAHAYA'
        ],
        zero_division=0
    )
)

# =========================
# RMSE DAN MAE CNN
# =========================
rmse_cnn = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mae_cnn = mean_absolute_error(
    y_true,
    y_pred
)

print("\n=== RMSE & MAE CNN ===")
print(f"RMSE : {rmse_cnn:.4f}")
print(f"MAE  : {mae_cnn:.4f}")

# =========================
# EVALUATION LAMA
# =========================
from evaluation import evaluate_model

evaluate_model(
    y_true,
    y_pred
)