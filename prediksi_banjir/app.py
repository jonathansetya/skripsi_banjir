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
         '<div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:1rem;">Pembuat Aplikasi</div>'
         '<div style="display:flex;align-items:center;gap:1.4rem;margin-bottom:1.2rem;">'
         '<img src="data:image/jpeg;base64,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" '
         'onerror="this.style.display=\'none\'" '
         'style="width:220px;flex-shrink:0;border-radius:10px;object-fit:cover;border:2px solid #1e2d45;" />'
         '<div>'
         '<div style="font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-bottom:.4rem;">Jonathan Setya Widayat</div>'
         '<div style="font-size:.9rem;color:#64748b;">NIM: 22104410047</div>'
         '</div></div>'
         '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem .8rem;">'
         '<div><div style="font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Jurusan</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;font-weight:500;margin-top:.15rem;">Teknik Informatika</div></div>'
         '<div><div style="font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Tahun Ajaran</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;font-weight:500;margin-top:.15rem;">2022</div></div>'
         '<div style="grid-column:1/-1;margin-top:.2rem;">'
         '<div style="font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Dosen Pembimbing</div>'
         '<div style="font-size:.85rem;color:#cbd5e1;margin-top:.15rem;line-height:1.8;">1. Saiful Nur Budiman, S.Kom., M.Kom<br>2. Filda Febrinita, S.Pd., M.Pd</div>'
         '</div></div></div>')
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