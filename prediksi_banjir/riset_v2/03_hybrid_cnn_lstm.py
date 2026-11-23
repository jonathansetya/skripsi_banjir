"""
Langkah 5: dua model hybrid pada deret HARIAN hujan (30 hari sebelum akhir tiap dasarian).
  A. LSTM-CNN : LSTM dulu, keluarannya (per langkah waktu) diproses Conv1D. Urutan ini sesuai proposal dan judul.
  B. CNN-LSTM : Conv1D dulu, lalu LSTM. Varian sebagai pembanding.
Keduanya dilaporkan, tidak ada yang dipilih berdasarkan hasil.
Dievaluasi dengan skema yang SAMA dengan baseline (leave-one-year-out).
Jalankan setelah 02_pipeline_baseline.py:   python 03_hybrid_cnn_lstm.py

ATURAN: hyperparameter ditetapkan SEKALI di bawah, tidak diubah-ubah setelah melihat hasil.
"""
import os, calendar, random
import numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             average_precision_score, roc_auc_score)

DATA = os.environ.get("DATA_DIR", "data"); OUT = os.environ.get("OUT_DIR", "hasil")
os.makedirs(OUT, exist_ok=True)
SEED, WINDOW, EPOCHS, BATCH = 42, 30, 40, 32     # <- ditetapkan di awal, jangan diubah demi hasil
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

das = pd.read_csv(f"{OUT}/dataset_dasarian.csv", parse_dates=["mulai"])
ch = pd.read_csv(f"{DATA}/chirps_sutojayan.csv", parse_dates=["date"]).rename(columns={"mean": "hujan"})
harian = ch.pivot_table(index="date", columns="st", values="hujan").mean(axis=1).clip(lower=0)

def akhir(r):
    n = 10 if r.dasarian < 3 else calendar.monthrange(int(r.tahun), int(r.bulan))[1] - 20
    return r.mulai + pd.Timedelta(days=n - 1)

Xs = []
for _, r in das.iterrows():
    e = akhir(r)
    Xs.append(harian.loc[e - pd.Timedelta(days=WINDOW - 1): e].values)
X = np.log1p(np.array(Xs))[..., None].astype("float32")   # log1p meredam hujan ekstrem; (n, 30, 1)
y, thn = das.banjir.values.astype("float32"), das.tahun.values

def bangun(arsitektur):
    if arsitektur == "LSTM-CNN":
        m = models.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.LSTM(16, return_sequences=True),               # LSTM: dinamika temporal
            layers.Conv1D(8, 3, activation="relu", padding="same"),  # CNN: pola lokal pada keluaran LSTM
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")])
    else:  # CNN-LSTM
        m = models.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.Conv1D(8, 3, activation="relu", padding="same"),
            layers.MaxPooling1D(2),
            layers.LSTM(16),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

def evaluasi(nama, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(model=nama, TP=tp, FP=fp, FN=fn, TN=tn,
        precision=round(precision_score(y, pred, zero_division=0), 3),
        recall=round(recall_score(y, pred, zero_division=0), 3),
        F1=round(f1_score(y, pred, zero_division=0), 3),
        PR_AUC=round(average_precision_score(y, prob), 3), ROC_AUC=round(roc_auc_score(y, prob), 3))

hasil, oof = [], dict(mulai=das.mulai, banjir=y)
for arsitektur, kolom in [("LSTM-CNN", "p_lstm_cnn"), ("CNN-LSTM", "p_cnn_lstm")]:
    prob = np.zeros(len(y))
    for t in np.unique(thn):
        te, tr = thn == t, thn != t
        if y[tr].sum() == 0: continue
        pos = y[tr].sum(); w = {0: 1.0, 1: float((len(y[tr]) - pos) / pos)}   # class weight
        tf.keras.backend.clear_session(); tf.random.set_seed(SEED)
        m = bangun(arsitektur); m.fit(X[tr], y[tr], epochs=EPOCHS, batch_size=BATCH, class_weight=w, verbose=0)
        prob[te] = m.predict(X[te], verbose=0).ravel()
    oof[kolom] = prob
    hasil.append(evaluasi(f"Hybrid {arsitektur}", prob))
    print("selesai:", arsitektur)

res = pd.DataFrame(hasil)
print(res.to_string(index=False)); res.to_csv(f"{OUT}/hasil_hybrid.csv", index=False)
pd.DataFrame(oof).to_csv(f"{OUT}/prediksi_oof_hybrid.csv", index=False)
