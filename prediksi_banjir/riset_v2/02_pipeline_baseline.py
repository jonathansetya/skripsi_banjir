"""
Langkah 2-4: validasi CHIRPS vs BMKG, pembuatan dataset dasarian, evaluasi baseline.
Jalankan dari folder riset_v2:   python 02_pipeline_baseline.py
Butuh: data/chirps_sutojayan.csv (hasil 01_ekstrak_chirps.js)

ATURAN: tidak ada tuning-sampai-bagus. Semua angka yang keluar dilaporkan apa adanya.
"""
import os, calendar
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             average_precision_score, roc_auc_score)

DATA = os.environ.get("DATA_DIR", "data")
OUT = os.environ.get("OUT_DIR", "hasil")
os.makedirs(OUT, exist_ok=True)
SEED = 42
AMBANG_BMKG = 150.0   # mm/dasarian, peringatan dini hujan tinggi BMKG (sumber: wawancara BMKG; cek ke peraturan BMKG sebelum dikutip)

# ---------- 1. Muat CHIRPS ----------
ch = pd.read_csv(f"{DATA}/chirps_sutojayan.csv", parse_dates=["date"])
ch = ch.rename(columns={"mean": "hujan"})
ch["hujan"] = ch["hujan"].clip(lower=0)
wide = ch.pivot_table(index="date", columns="st", values="hujan").sort_index()
wide["rata2"] = wide.mean(axis=1)
print(f"CHIRPS: {wide.index.min().date()} s.d. {wide.index.max().date()}, {len(wide)} hari")
print("Korelasi harian Judeg vs Bacem:", round(wide["Judeg"].corr(wide["Bacem"]), 3))

# ---------- 2. Validasi vs BMKG bulanan ----------
bm = pd.read_csv(f"{DATA}/bmkg_curah_hujan_bulanan.csv")
bm["st"] = bm["stasiun"].map({"Sukorejo_Judeg": "Judeg", "Gedok_Bacem": "Bacem"})
rows = []
for st in ["Judeg", "Bacem"]:
    m = wide[st].resample("MS").sum().to_frame("chirps")
    m["tahun"], m["bulan"] = m.index.year, m.index.month
    j = bm[bm.st == st].merge(m, on=["tahun", "bulan"])
    r = np.corrcoef(j.hujan_mm, j.chirps)[0, 1]
    rmse = float(np.sqrt(np.mean((j.chirps - j.hujan_mm) ** 2)))
    bias = float((j.chirps - j.hujan_mm).mean())
    rows.append(dict(stasiun=st, n_bulan=len(j), korelasi=round(r, 3), RMSE_mm=round(rmse, 1), bias_mm=round(bias, 1)))
val = pd.DataFrame(rows)
print("\n=== Validasi CHIRPS vs BMKG bulanan ===\n", val.to_string(index=False))
val.to_csv(f"{OUT}/validasi_chirps_vs_bmkg.csv", index=False)

# ---------- 3. Dataset dasarian ----------
d = wide[["rata2"]].copy()
d["tahun"], d["bulan"], d["hari"] = d.index.year, d.index.month, d.index.day
d["dasarian"] = np.where(d.hari <= 10, 1, np.where(d.hari <= 20, 2, 3))
g = d.groupby(["tahun", "bulan", "dasarian"])
das = g["rata2"].agg(total="sum", maks="max", n_hari="count").reset_index()
def hari_seharusnya(r):
    if r.dasarian == 1: return 10
    if r.dasarian == 2: return 10
    return calendar.monthrange(int(r.tahun), int(r.bulan))[1] - 20
das["lengkap"] = das.apply(lambda r: r.n_hari == hari_seharusnya(r), axis=1)
das = das[das.lengkap].drop(columns="lengkap").reset_index(drop=True)
das["mulai"] = pd.to_datetime(dict(year=das.tahun, month=das.bulan, day=(das.dasarian - 1) * 10 + 1))
das["lag1"] = das["total"].shift(1)
das["lag2"] = das["total"].shift(2)
das["bulan_sin"] = np.sin(2 * np.pi * das.bulan / 12)
das["bulan_cos"] = np.cos(2 * np.pi * das.bulan / 12)
das = das.dropna()
das = das[das.mulai >= "2016-01-01"].reset_index(drop=True)   # 2015 hanya untuk pemanasan lag

# ---------- 4. Label dari SmartPB ----------
ev = pd.read_csv(f"{DATA}/kejadian_banjir_sutojayan.csv", parse_dates=["tanggal"])
ev = ev[ev.include == "ya"].drop_duplicates("tanggal")
def label(row):
    mulai = row.mulai
    akhir = mulai + pd.Timedelta(days=hari_seharusnya(row) - 1)
    return int(((ev.tanggal >= mulai) & (ev.tanggal <= akhir)).any())
das["banjir"] = das.apply(label, axis=1)
# kejadian yang jatuh di luar rentang data tidak akan terhitung -> beri tahu
tercakup = ev[(ev.tanggal >= das.mulai.min()) & (ev.tanggal <= das.mulai.max() + pd.Timedelta(days=10))]
print(f"\nDasarian dipakai: {len(das)} | positif (ada banjir): {das.banjir.sum()} | kejadian tercakup: {len(tercakup)} dari {len(ev)}")
das.to_csv(f"{OUT}/dataset_dasarian.csv", index=False)

# ---------- 5. Evaluasi: leave-one-year-out ----------
FITUR = ["total", "maks", "lag1", "lag2", "bulan_sin", "bulan_cos"]
X, y, thn = das[FITUR].values, das.banjir.values, das.tahun.values

def cv(model_fn):
    prob = np.zeros(len(y))
    for t in np.unique(thn):
        te, tr = thn == t, thn != t
        if y[tr].sum() == 0:
            continue
        m = model_fn(); m.fit(X[tr], y[tr]); prob[te] = m.predict_proba(X[te])[:, 1]
    return prob

def ringkas(nama, prob, thr=0.5, skor=None):
    # skor: skor kontinu untuk AUC (bila None, pakai prob)
    pred = (prob >= thr).astype(int)
    sk = prob if skor is None else skor
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ap = average_precision_score(y, sk) if sk.std() > 0 else float("nan")
    auc = roc_auc_score(y, sk) if sk.std() > 0 else float("nan")
    return dict(model=nama, TP=tp, FP=fp, FN=fn, TN=tn,
                precision=round(precision_score(y, pred, zero_division=0), 3),
                recall=round(recall_score(y, pred, zero_division=0), 3),
                F1=round(f1_score(y, pred, zero_division=0), 3),
                PR_AUC=round(ap, 3), ROC_AUC=round(auc, 3))

hasil = []
# baseline 0: aturan BMKG 150 mm/dasarian (tanpa pelatihan)
p_rule = (das.total.values >= AMBANG_BMKG).astype(float)
hasil.append(ringkas("Aturan BMKG >=150 mm/dasarian", p_rule, skor=das.total.values))   # P/R/F1 dari ambang 150; AUC dari curah hujan kontinu
# baseline 1: regresi logistik
p_lr = cv(lambda: make_pipeline(StandardScaler(),
             LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)))
hasil.append(ringkas("Regresi logistik (class_weight)", p_lr))
# baseline 2: random forest
p_rf = cv(lambda: RandomForestClassifier(
             n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=3, random_state=SEED))
hasil.append(ringkas("Random forest (class_weight)", p_rf))
res = pd.DataFrame(hasil)
print("\n=== Hasil leave-one-year-out (apa adanya) ===\n", res.to_string(index=False))
res.to_csv(f"{OUT}/hasil_baseline.csv", index=False)
pd.DataFrame(dict(mulai=das.mulai, tahun=das.tahun, banjir=y, skor_hujan=das.total.values,
    p_rule=p_rule, p_lr=p_lr, p_rf=p_rf)).to_csv(f"{OUT}/prediksi_oof_baseline.csv", index=False)
print(f"\nCatatan: hanya {int(y.sum())} dasarian positif. Interpretasi harus hati-hati; laporkan sebagai keterbatasan.")
