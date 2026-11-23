"""
Langkah 7: gabungkan hasil penelitian menjadi satu file untuk tab "Uji Ulang Historis" di aplikasi.
Jalankan setelah 02, 03, 04:  python 05_ekspor_replay.py
Lalu salin hasil/replay_dasarian.csv ke  prediksi_banjir_v2/data/replay_dasarian.csv
Probabilitas di file ini adalah prediksi out-of-fold: tahun yang diprediksi tidak ikut dilatih.
"""
import os
import pandas as pd
OUT = os.environ.get("OUT_DIR", "hasil")
d = pd.read_csv(f"{OUT}/dataset_dasarian.csv", parse_dates=["mulai"])
b = pd.read_csv(f"{OUT}/prediksi_oof_baseline.csv", parse_dates=["mulai"])
h = pd.read_csv(f"{OUT}/prediksi_oof_hybrid.csv", parse_dates=["mulai"])
r = d[["mulai", "tahun", "total", "banjir"]].rename(columns={"total": "total_mm"})
r = r.merge(b[["mulai", "p_lr", "p_rf"]], on="mulai").merge(h.drop(columns=["banjir"]), on="mulai")
r.to_csv(f"{OUT}/replay_dasarian.csv", index=False)
print("Ditulis:", f"{OUT}/replay_dasarian.csv", "|", len(r), "baris |", int(r.banjir.sum()), "dasarian banjir")
