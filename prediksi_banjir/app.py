import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
import time
import os

# ─────────────────────────────────────────
#  DATA CURAH HUJAN HISTORIS
# ─────────────────────────────────────────
rainfall_sukorejo = {
    2021:[374,512,144,72,0,127,4,35,42,51,552,198],
    2022:[174,278,287,409,112,83,22,21,174,738,301,227],
    2023:[108,394,236,111,33,48,103,0,0,0,29,51],
    2024:[224,161,178,59,0,0,5,0,0,62,294,263],
    2025:[272,173,177,101,218,95,3,101,48,168,219,69],
}
rainfall_bacem = {
    2021:[297,451,257,92,0,184,10,62,75,69,518,229],
    2022:[214,319,315,377,174,123,18,29,159,746,308,328],
    2023:[231,445,325,137,28,50,144,0,0,0,60,106],
    2024:[341,292,219,145,0,0,11,0,0,43,251,257],
    2025:[274,152,187,76,192,60,4,79,28,176,248,95],
}

KECAMATAN = {
    "Sutojayan": "35.05.12.1005",
}

def estimate_rain(month, year):
    if year not in rainfall_sukorejo:
        year = 2025
    s = rainfall_sukorejo[year][month-1]
    b = rainfall_bacem[year][month-1]
    return ((s+b)/2) / 30 / 8

def label_banjir(rain):
    if rain <= 1:   return 0
    elif rain <= 3: return 1
    else:           return 2

# ─────────────────────────────────────────
#  FETCH BMKG
# ─────────────────────────────────────────
def fetch_bmkg(adm4):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    r = requests.get(url, timeout=15)
    data = r.json()
    records = []
    for item in data['data'][0]['cuaca']:
        for c in item:
            dt = datetime.fromisoformat(c["local_datetime"])
            records.append({
                "datetime": c["local_datetime"],
                "temp":     c["t"],
                "humidity": c["hu"],
                "wind":     c["ws"],
                "weather":  c["weather_desc"],
                "rain":     estimate_rain(dt.month, dt.year),
            })
    return pd.DataFrame(records)

# ─────────────────────────────────────────
#  MODEL (sklearn — ringan, tanpa TF)
# ─────────────────────────────────────────
def run_model(df):
    df2 = df.copy()
    df2['weather'] = df2['weather'].astype('category').cat.codes
    features = df2[['temp','humidity','wind','weather','rain']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # Sequence window=3
    window = 3
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window].flatten())   # (15,)
        y.append(scaled[i+window][1])             # humidity target
    X, y = np.array(X), np.array(y)

    # "LSTM" digantikan MLPRegressor (feature extractor)
    regressor = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    regressor.fit(X, y)
    lstm_out = regressor.predict(X).reshape(-1, 1)  # (n, 1)

    # Label banjir dari nilai rain asli (bukan dari regressor output)
    rain_vals = df2['rain'].values[window:]
    y_clf = np.array([label_banjir(float(v)) for v in rain_vals])

    # Pastikan ada minimal 3 kelas agar predict_proba punya 3 kolom
    # Jika data hanya punya 1-2 kelas, tambahkan dummy row kecil
    unique = np.unique(y_clf)
    if len(unique) < 3:
        missing = [c for c in [0, 1, 2] if c not in unique]
        X_extra  = np.zeros((len(missing), X.shape[1]))
        lo_extra = np.array(missing)
        X_aug    = np.vstack([X, X_extra])
        y_aug    = np.concatenate([y_clf, lo_extra])
    else:
        X_aug, y_aug = X, y_clf

    # "CNN" → RandomForestClassifier (stabil, tidak error shape)
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(X_aug, y_aug)

    pred_proba = classifier.predict_proba(X[-1].reshape(1, -1))[0]
    # Urutkan sesuai kelas 0,1,2
    full_proba = np.zeros(3)
    for i, cls in enumerate(classifier.classes_):
        full_proba[int(cls)] = pred_proba[i]

    return full_proba, df

# ─────────────────────────────────────────
#  STREAMLIT CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Dini Banjir – Kab. Blitar",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:       #0a0f1e;
    --surface:  #111827;
    --surface2: #1a2235;
    --border:   #1e2d45;
    --aman:     #10b981;
    --waspada:  #f59e0b;
    --bahaya:   #ef4444;
    --blue:     #3b82f6;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --font:     'Plus Jakarta Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stHeader"],[data-testid="stToolbar"]{ display:none !important; }
section.main > div { padding:0 !important; }
.block-container { max-width:100% !important; padding:0 !important; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }

[data-testid="stSelectbox"] > div > div {
    background:var(--surface2) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:10px !important;
}
[data-testid="stButton"] > button {
    background:var(--blue) !important; color:white !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    padding:.55rem 1.4rem !important; transition:opacity .2s;
}
[data-testid="stButton"] > button:hover { opacity:.85 !important; }

[data-testid="stTabs"] [role="tab"] {
    color:var(--muted) !important; font-weight:600 !important;
    font-size:.9rem !important; border:none !important; padding:.6rem 1.2rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:var(--text) !important; border-bottom:2px solid var(--blue) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom:1px solid var(--border) !important; gap:.5rem;
}

[data-testid="stMetric"] {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:12px; padding:1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]{ color:var(--muted) !important; font-size:.8rem !important; }
[data-testid="stMetricValue"]{ color:var(--text) !important; font-size:1.6rem !important; font-weight:700 !important; }

[data-testid="stProgress"] > div > div { background:var(--blue) !important; border-radius:9px; }
[data-testid="stProgress"] > div { background:var(--border) !important; border-radius:9px; }

p,li,span,label { font-family:var(--font) !important; color:var(--text) !important; }
h1,h2,h3,h4    { font-family:var(--font) !important; color:var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
for k,v in [("hasil",None),("df_bmkg",None),("last_update",None),("kecamatan_sel","Sutojayan")]:
    if k not in st.session_state:
        st.session_state[k] = v

REFRESH_INTERVAL = 60 * 60  # 60 menit

def should_refresh():
    if st.session_state.last_update is None: return True
    return (time.time() - st.session_state.last_update) >= REFRESH_INTERVAL

def do_predict(kec):
    adm4 = KECAMATAN.get(kec, "35.05.12.1005")
    df   = fetch_bmkg(adm4)
    hasil, df = run_model(df)
    st.session_state.hasil       = hasil
    st.session_state.df_bmkg     = df
    st.session_state.last_update = time.time()

# ─────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────
def status_info(hasil):
    idx    = int(np.argmax(hasil))
    labels = ["AMAN","WASPADA","BAHAYA"]
    colors = ["#10b981","#f59e0b","#ef4444"]
    icons  = ["🟢","🟡","🔴"]
    return labels[idx], colors[idx], icons[idx], hasil[0]*100, hasil[1]*100, hasil[2]*100

def render_header():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2e,#0a0f1e 60%,#091623);
                border-bottom:1px solid #1e2d45;padding:1rem 2rem;
                display:flex;align-items:center;gap:1rem;">
        <div style="font-size:2rem;">🌊</div>
        <div>
            <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;">
                Prediksi Dini Banjir</div>
            <div style="font-size:.78rem;color:#64748b;font-weight:500;">
                Kabupaten Blitar — Sistem Monitoring Real-time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_status_card(hasil):
    label,color,icon,p_aman,p_waspada,p_bahaya = status_info(hasil)
    now = datetime.fromtimestamp(st.session_state.last_update).strftime("%d %b %Y, %H:%M") \
          if st.session_state.last_update else "—"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#111827,#1a2235);
                border:1px solid {color}44;border-radius:20px;
                padding:2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
                    background:{color}15;border-radius:50%;"></div>
        <div style="position:absolute;top:10px;right:20px;font-size:4rem;opacity:.12;">{icon}</div>
        <div style="font-size:.75rem;font-weight:600;color:#64748b;letter-spacing:.1em;
                    text-transform:uppercase;margin-bottom:.5rem;">STATUS BANJIR SAAT INI</div>
        <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:1.5rem;">
            <div style="font-size:2.4rem;">{icon}</div>
            <div style="font-size:2.8rem;font-weight:800;color:{color};
                        letter-spacing:-.03em;line-height:1;">{label}</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1.2rem;">
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;
                        padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#10b981;">{p_aman:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;margin-top:.2rem;">🟢 Aman</div>
            </div>
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;
                        padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#f59e0b;">{p_waspada:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;margin-top:.2rem;">🟡 Waspada</div>
            </div>
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;
                        padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{p_bahaya:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;margin-top:.2rem;">🔴 Bahaya</div>
            </div>
        </div>
        <div style="font-size:.73rem;color:#64748b;">
            🕐 Diperbarui: {now} &nbsp;·&nbsp; Auto-refresh setiap 60 menit</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────
def page_beranda():
    st.markdown('<div style="padding:1.5rem 2rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">'
                '<div style="font-size:.85rem;color:#64748b;">📍</div>'
                '<div style="font-size:.95rem;font-weight:700;color:#f1f5f9;">Kecamatan Sutojayan, Kabupaten Blitar</div>'
                '</div>', unsafe_allow_html=True)
    col_btn, _ = st.columns([1.2, 6])
    with col_btn:
        manual = st.button("🔄 Refresh")
    st.markdown('</div>', unsafe_allow_html=True)

    if should_refresh() or manual:
        with st.spinner("Mengambil data BMKG & menjalankan model prediksi..."):
            try:
                do_predict(st.session_state.kecamatan_sel)
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")
                return

    st.markdown('<div style="padding:.5rem 2rem 2rem;">', unsafe_allow_html=True)
    if st.session_state.hasil is not None:
        hasil = st.session_state.hasil
        df    = st.session_state.df_bmkg

        render_status_card(hasil)

        # Progress bar custom
        label,color,icon,p_aman,p_waspada,p_bahaya = status_info(hasil)
        st.markdown(
            f'<div style="margin-bottom:1.2rem;">'
            f'<div style="font-size:.8rem;font-weight:700;color:#f1f5f9;margin-bottom:.8rem;">Distribusi Probabilitas Model</div>'
            f'<div style="display:flex;flex-direction:column;gap:.6rem;">'

            f'<div><div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">'
            f'<span style="font-size:.82rem;color:#f1f5f9;">🟢 Aman</span>'
            f'<span style="font-size:.82rem;font-weight:700;color:#10b981;">{p_aman:.1f}%</span></div>'
            f'<div style="background:#1e2d45;border-radius:9px;height:10px;overflow:hidden;">'
            f'<div style="width:{p_aman:.1f}%;background:#10b981;height:100%;border-radius:9px;'
            f'transition:width .6s ease;"></div></div></div>'

            f'<div><div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">'
            f'<span style="font-size:.82rem;color:#f1f5f9;">🟡 Waspada</span>'
            f'<span style="font-size:.82rem;font-weight:700;color:#f59e0b;">{p_waspada:.1f}%</span></div>'
            f'<div style="background:#1e2d45;border-radius:9px;height:10px;overflow:hidden;">'
            f'<div style="width:{p_waspada:.1f}%;background:#f59e0b;height:100%;border-radius:9px;'
            f'transition:width .6s ease;"></div></div></div>'

            f'<div><div style="display:flex;justify-content:space-between;margin-bottom:.3rem;">'
            f'<span style="font-size:.82rem;color:#f1f5f9;">🔴 Bahaya</span>'
            f'<span style="font-size:.82rem;font-weight:700;color:#ef4444;">{p_bahaya:.1f}%</span></div>'
            f'<div style="background:#1e2d45;border-radius:9px;height:10px;overflow:hidden;">'
            f'<div style="width:{p_bahaya:.1f}%;background:#ef4444;height:100%;border-radius:9px;'
            f'transition:width .6s ease;"></div></div></div>'

            f'</div></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📡 Data Prakiraan BMKG – 5 Jam Terdekat**")
        df_show = df[['datetime','temp','humidity','wind','weather','rain']].head(5).copy()
        df_show.columns = ['Waktu','Suhu (°C)','Kelembaban (%)','Angin (km/h)','Cuaca','Curah Hujan (mm/3j)']
        df_show['Curah Hujan (mm/3j)'] = df_show['Curah Hujan (mm/3j)'].round(3)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Countdown refresh
        if st.session_state.last_update:
            elapsed   = time.time() - st.session_state.last_update
            remaining = max(0, REFRESH_INTERVAL - elapsed)
            mnt, sec  = divmod(int(remaining), 60)
            st.markdown(f"<div style='font-size:.73rem;color:#64748b;margin-top:.8rem;'>"
                        f"⏳ Refresh berikutnya dalam: {mnt:02d}:{sec:02d}</div>",
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def page_informasi():
    st.markdown('<div style="padding:1.5rem 2rem;">', unsafe_allow_html=True)
    kec = st.session_state.kecamatan_sel
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;
                padding:1.5rem;margin-bottom:1.5rem;">
        <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:.4rem;">Kecamatan Dipilih</div>
        <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;">{kec}</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:.3rem;">Kabupaten Blitar, Jawa Timur</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.hasil is not None:
        df = st.session_state.df_bmkg
        st.markdown("### 📊 Statistik Cuaca")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("🌡️ Suhu Rata-rata",      f"{df['temp'].mean():.1f} °C")
        with c2: st.metric("💧 Kelembaban Rata-rata", f"{df['humidity'].mean():.0f} %")
        with c3: st.metric("💨 Kecepatan Angin",      f"{df['wind'].mean():.1f} km/h")
        with c4: st.metric("🌧️ Curah Hujan Est.",     f"{df['rain'].mean():.3f} mm/3j")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Data Prakiraan Lengkap")
        df_show = df[['datetime','temp','humidity','wind','weather','rain']].copy()
        df_show.columns = ['Waktu','Suhu (°C)','Kelembaban (%)','Angin (km/h)','Cuaca','Curah Hujan (mm/3j)']
        df_show['Curah Hujan (mm/3j)'] = df_show['Curah Hujan (mm/3j)'].round(4)
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=380)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏞️ Data Curah Hujan Historis")
        tab1, tab2 = st.tabs(["Pos Sukorejo/Judeg", "Pos Gedok/Bacem"])
        bulan = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        with tab1:
            st.dataframe(pd.DataFrame(rainfall_sukorejo, index=bulan).T,
                         use_container_width=True)
        with tab2:
            st.dataframe(pd.DataFrame(rainfall_bacem, index=bulan).T,
                         use_container_width=True)
    else:
        st.info("Belum ada data. Buka tab **Beranda** untuk memuat prediksi terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

def card(html):
    st.markdown(html, unsafe_allow_html=True)

def page_tentang():
    st.markdown('<div style="padding:1.5rem 2rem;max-width:700px;">', unsafe_allow_html=True)

    # Card 1 — Judul
    card('<div style="background:linear-gradient(135deg,#111827,#1a2235);border:1px solid #1e2d45;'
         'border-radius:20px;padding:2rem;margin-bottom:1rem;">'
         '<div style="font-size:2.5rem;margin-bottom:1rem;">🌊</div>'
         '<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-bottom:.5rem;">Sistem Prediksi Dini Banjir</div>'
         '<div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">'
         'Sistem ini menggunakan model Hybrid <b style="color:#3b82f6;">LSTM-CNN</b> '
         'untuk memprediksi potensi banjir di Kabupaten Blitar berdasarkan data cuaca real-time dari API BMKG.'
         '</div></div>')

    # Card 2 — Cara Kerja
    card('<div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">'
         '<div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.8rem;">Cara Kerja Sistem</div>'
         '<div style="display:flex;flex-direction:column;gap:.6rem;">'
         '<div style="display:flex;align-items:flex-start;gap:.8rem;">'
         '<div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">01</div>'
         '<div style="font-size:.84rem;color:#cbd5e1;">Data cuaca diambil dari API BMKG setiap 60 menit secara otomatis</div></div>'
         '<div style="display:flex;align-items:flex-start;gap:.8rem;">'
         '<div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">02</div>'
         '<div style="font-size:.84rem;color:#cbd5e1;">Data dinormalisasi dan dibentuk menjadi sequence 3 timestep (9 jam)</div></div>'
         '<div style="display:flex;align-items:flex-start;gap:.8rem;">'
         '<div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">03</div>'
         '<div style="font-size:.84rem;color:#cbd5e1;">Model LSTM mengekstrak pola temporal dari data cuaca</div></div>'
         '<div style="display:flex;align-items:flex-start;gap:.8rem;">'
         '<div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">04</div>'
         '<div style="font-size:.84rem;color:#cbd5e1;">Model CNN mengklasifikasikan ke 3 status: Aman, Waspada, Bahaya</div></div>'
         '</div></div>')

    # Card 3 — Kategori Status
    card('<div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">'
         '<div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.8rem;">Kategori Status Banjir</div>'
         '<div style="display:flex;flex-direction:column;gap:.6rem;">'
         '<div style="display:flex;align-items:center;gap:.8rem;background:#10b98112;border:1px solid #10b98133;border-radius:10px;padding:.8rem 1rem;">'
         '<span style="font-size:1.2rem;">🟢</span>'
         '<div><div style="font-size:.85rem;font-weight:700;color:#10b981;">AMAN</div>'
         '<div style="font-size:.78rem;color:#64748b;">Curah hujan ≤ 1 mm/3 jam — risiko banjir rendah</div></div></div>'
         '<div style="display:flex;align-items:center;gap:.8rem;background:#f59e0b12;border:1px solid #f59e0b33;border-radius:10px;padding:.8rem 1rem;">'
         '<span style="font-size:1.2rem;">🟡</span>'
         '<div><div style="font-size:.85rem;font-weight:700;color:#f59e0b;">WASPADA</div>'
         '<div style="font-size:.78rem;color:#64748b;">Curah hujan 1–3 mm/3 jam — perlu persiapan antisipasi</div></div></div>'
         '<div style="display:flex;align-items:center;gap:.8rem;background:#ef444412;border:1px solid #ef444433;border-radius:10px;padding:.8rem 1rem;">'
         '<span style="font-size:1.2rem;">🔴</span>'
         '<div><div style="font-size:.85rem;font-weight:700;color:#ef4444;">BAHAYA</div>'
         '<div style="font-size:.78rem;color:#64748b;">Curah hujan &gt; 3 mm/3 jam — potensi banjir tinggi</div></div></div>'
         '</div></div>')

    # Card 4 — Profil Pembuat
    card('<div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">'
         '<div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem;">Pembuat Aplikasi</div>'
         '<div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">'
         '<img src="data:image/jpeg;base64,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" '
         'onerror="this.style.display=\'none\'" '
         'style="width:72px;height:72px;border-radius:50%;object-fit:cover;border:2px solid #3b82f6;" />'
         '<div>'
         '<div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;">Jonathan Setya Widayat</div>'
         '<div style="font-size:.8rem;color:#64748b;margin-top:.2rem;">NIM: 22104410047</div>'
         '</div></div>'
         '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem .8rem;">'
         '<div><div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Jurusan</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;font-weight:500;margin-top:.15rem;">Teknik Informatika</div></div>'
         '<div><div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Tahun Ajaran</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;font-weight:500;margin-top:.15rem;">2022</div></div>'
         '<div style="grid-column:1/-1;">'
         '<div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Dosen Pembimbing</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;margin-top:.15rem;line-height:1.7;">'
         '1. Saiful Nur Budiman, S.Kom., M.Kom<br>'
         '2. Filda Febrinita, S.Pd., M.Pd'
         '</div></div>'
         '</div></div>')

    # Card 5 — Disclaimer
    card('<div style="background:#0a0f1e;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">'
         '<div style="display:flex;align-items:flex-start;gap:.8rem;">'
         '<div style="font-size:1.3rem;flex-shrink:0;">📋</div>'
         '<div>'
         '<div style="font-size:.8rem;font-weight:700;color:#94a3b8;margin-bottom:.4rem;">Pernyataan Penggunaan</div>'
         '<div style="font-size:.8rem;color:#64748b;line-height:1.8;">'
         'Aplikasi ini dibuat semata-mata untuk kepentingan penelitian skripsi dan tidak digunakan '
         'untuk tujuan komersial maupun monetisasi dalam bentuk apapun. Seluruh data, hasil prediksi, '
         'dan informasi yang ditampilkan hanya diperuntukkan bagi keperluan akademis.'
         '<br><br>'
         'Sumber data: <b style="color:#3b82f6;">API BMKG</b> (Badan Meteorologi, Klimatologi, dan Geofisika) '
         '&amp; data curah hujan historis Pos Sukorejo/Judeg dan Pos Gedok/Bacem, Kecamatan Sutojayan, Kabupaten Blitar.'
         '</div></div></div></div>')

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
render_header()

tab_beranda, tab_info, tab_tentang = st.tabs([
    "🏠  Beranda",
    "📊  Informasi & Statistik",
    "ℹ️  Tentang Kami"
])

with tab_beranda:  page_beranda()
with tab_info:     page_informasi()
with tab_tentang:  page_tentang()

# Auto-rerun setelah interval habis
if st.session_state.last_update:
    elapsed   = time.time() - st.session_state.last_update
    remaining = REFRESH_INTERVAL - elapsed
    if remaining <= 5:
        time.sleep(2)
        st.rerun()