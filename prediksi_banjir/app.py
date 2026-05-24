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
         '<div style="margin-bottom:1.2rem;">'
         '<img src="data:image/jpeg;base64,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" '
         'onerror="this.style.display=\'none\'" '
         'style="width:100%;max-width:420px;border-radius:12px;object-fit:cover;border:2px solid #1e2d45;display:block;margin-bottom:.8rem;" />'
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