"""
Langkah 6: interval kepercayaan (bootstrap per tahun), perbandingan antar model, dan grafik untuk naskah.
Jalankan setelah 02 dan 03 (keduanya sudah menyimpan prediksi out-of-fold):  python 04_analisis_dan_grafik.py
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

OUT = os.environ.get("OUT_DIR", "hasil"); DATA = os.environ.get("DATA_DIR", "data")
B, SEED = 2000, 42
rng = np.random.default_rng(SEED)

das = pd.read_csv(f"{OUT}/dataset_dasarian.csv", parse_dates=["mulai"])
oof = pd.read_csv(f"{OUT}/prediksi_oof_baseline.csv", parse_dates=["mulai"])
skor = {"Curah hujan dasarian (aturan BMKG)": oof.skor_hujan.values,
        "Regresi logistik": oof.p_lr.values, "Random forest": oof.p_rf.values}
if os.path.exists(f"{OUT}/prediksi_oof_hybrid.csv"):
    hy = pd.read_csv(f"{OUT}/prediksi_oof_hybrid.csv", parse_dates=["mulai"])
    assert (hy.mulai.values == oof.mulai.values).all(), "urutan data hybrid dan baseline berbeda"
    if "p_lstm_cnn" in hy.columns:
        skor["Hybrid LSTM-CNN"] = hy.p_lstm_cnn.values
    if "p_cnn_lstm" in hy.columns:
        skor["Hybrid CNN-LSTM (varian)"] = hy.p_cnn_lstm.values
    if "p_hybrid" in hy.columns:      # kompatibel dengan keluaran 03 versi lama
        skor["Hybrid CNN-LSTM (varian)"] = hy.p_hybrid.values
y, thn = oof.banjir.values, oof.tahun.values
tahun = np.unique(thn)
idx_t = {t: np.where(thn == t)[0] for t in tahun}

def metrik(sc, ii):
    yy = y[ii]
    if yy.sum() == 0 or yy.sum() == len(yy): return np.nan, np.nan
    return roc_auc_score(yy, sc[ii]), average_precision_score(yy, sc[ii])

# ---- bootstrap per tahun (blok), menjaga ketergantungan antar dasarian dalam satu tahun ----
boot = {k: [] for k in skor}
for _ in range(B):
    pilih = rng.choice(tahun, size=len(tahun), replace=True)
    ii = np.concatenate([idx_t[t] for t in pilih])
    for k, sc in skor.items():
        boot[k].append(metrik(sc, ii))
rows = []
for k, sc in skor.items():
    a = np.array(boot[k]); a = a[~np.isnan(a).any(axis=1)]
    auc0, ap0 = metrik(sc, np.arange(len(y)))
    rows.append(dict(model=k, ROC_AUC=round(auc0, 3),
                     ROC_AUC_CI95=f"{np.percentile(a[:,0],2.5):.2f}-{np.percentile(a[:,0],97.5):.2f}",
                     PR_AUC=round(ap0, 3),
                     PR_AUC_CI95=f"{np.percentile(a[:,1],2.5):.2f}-{np.percentile(a[:,1],97.5):.2f}"))
tab = pd.DataFrame(rows); tab["PR_AUC_acak(prevalensi)"] = round(y.mean(), 3)
print(tab.to_string(index=False)); tab.to_csv(f"{OUT}/hasil_dengan_CI.csv", index=False)

# ---- selisih berpasangan terhadap model acuan (aturan BMKG) ----
ref = "Curah hujan dasarian (aturan BMKG)"
dif = []
for k in skor:
    if k == ref: continue
    d = []
    for _ in range(B):
        pilih = rng.choice(tahun, size=len(tahun), replace=True)
        ii = np.concatenate([idx_t[t] for t in pilih])
        a1, p1 = metrik(skor[k], ii); a0, p0 = metrik(skor[ref], ii)
        if not np.isnan(a1): d.append((a1 - a0, p1 - p0))
    d = np.array(d)
    dif.append(dict(model=k, dROC_AUC_vs_acuan=round(np.mean(d[:,0]), 3),
                    CI95_ROC=f"{np.percentile(d[:,0],2.5):.2f}..{np.percentile(d[:,0],97.5):.2f}",
                    dPR_AUC_vs_acuan=round(np.mean(d[:,1]), 3),
                    CI95_PR=f"{np.percentile(d[:,1],2.5):.2f}..{np.percentile(d[:,1],97.5):.2f}"))
dif = pd.DataFrame(dif); print("\nSelisih terhadap aturan BMKG (CI mencakup 0 = tidak berbeda nyata):\n", dif.to_string(index=False))
dif.to_csv(f"{OUT}/selisih_vs_acuan.csv", index=False)

# ---- grafik ----
# 1) deret dasarian + kejadian banjir
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(das.mulai, das.total, width=9, color="#9ecae1", label="Curah hujan dasarian (CHIRPS)")
ax.axhline(150, color="gray", ls="--", lw=1, label="Ambang BMKG 150 mm/dasarian")
b = das[das.banjir == 1]
ax.scatter(b.mulai, b.total, color="red", zorder=3, s=28, label="Dasarian dengan kejadian banjir (SmartPB)")
ax.set_ylabel("mm/dasarian"); ax.set_ylim(0, 360); ax.legend(fontsize=8, loc="upper center", ncol=3); plt.tight_layout()
plt.savefig(f"{OUT}/grafik_deret_dasarian.png", dpi=200); plt.close()

# 2) CHIRPS vs BMKG bulanan
bm = pd.read_csv(f"{DATA}/bmkg_curah_hujan_bulanan.csv")
bb = bm.groupby(["tahun", "bulan"]).hujan_mm.mean().reset_index()
m = das.groupby(["tahun", "bulan"]).total.sum().reset_index().rename(columns={"total": "chirps"})
j = bb.merge(m, on=["tahun", "bulan"])
r = np.corrcoef(j.hujan_mm, j.chirps)[0, 1]
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(j.hujan_mm, j.chirps, s=18, alpha=.8); lim = max(j.hujan_mm.max(), j.chirps.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("BMKG bulanan, rata-rata 2 stasiun (mm)"); ax.set_ylabel("CHIRPS bulanan (mm)")
ax.set_title(f"r = {r:.2f} (n = {len(j)} bulan)", fontsize=10); plt.tight_layout()
plt.savefig(f"{OUT}/grafik_chirps_vs_bmkg.png", dpi=200); plt.close()

# 3) kurva ROC dan PR
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for k, sc in skor.items():
    f, t, _ = roc_curve(y, sc); axs[0].plot(f, t, label=k)
    p, rc, _ = precision_recall_curve(y, sc); axs[1].plot(rc, p, label=k)
axs[0].plot([0, 1], [0, 1], "k--", lw=1); axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].set_title("ROC")
axs[1].axhline(y.mean(), color="k", ls="--", lw=1); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision-Recall")
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/grafik_roc_pr.png", dpi=200); plt.close()
print("\nGrafik disimpan di", OUT)
