import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    "Sutojayan":    "35.05.12.1005",
    "Blitar":       "35.05.07.1001",
    "Kanigoro":     "35.05.09.2001",
    "Ponggok":      "35.05.10.2001",
    "Srengat":      "35.05.11.2001",
    "Wlingi":       "35.05.14.2001",
    "Kesamben":     "35.05.16.2001",
    "Selopuro":     "35.05.17.2001",
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
#  FETCH + MODEL
# ─────────────────────────────────────────
def fetch_bmkg(adm4):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    r = requests.get(url, timeout=10)
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

def run_model(df):
    df2 = df.copy()
    df2['weather'] = df2['weather'].astype('category').cat.codes
    features = df2[['temp','humidity','wind','weather','rain']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    window = 3
    X, y = [], []
    for i in range(len(scaled)-window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window][1])
    X, y = np.array(X), np.array(y)

    # LSTM
    lstm = Sequential([Input(shape=(3,5)), LSTM(64), Dense(32, activation='relu'), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X, y, epochs=10, verbose=0)
    lstm_out = lstm.predict(X, verbose=0)

    # CNN
    X_cnn = lstm_out.reshape((lstm_out.shape[0],1,1,1))
    y_cnn = to_categorical(np.array([label_banjir(v) for v in lstm_out.flatten()]))
    cnn = Sequential([
        Conv2D(8,(1,1),activation='relu',input_shape=(1,1,1)),
        MaxPooling2D((1,1)), Flatten(),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(X_cnn, y_cnn, epochs=10, verbose=0)

    pred = cnn.predict(X_cnn, verbose=0)
    hasil = pred[-1]
    return hasil, df

# ─────────────────────────────────────────
#  STREAMLIT CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Dini Banjir – Kab. Blitar",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0a0f1e;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1e2d45;
    --aman:      #10b981;
    --waspada:   #f59e0b;
    --bahaya:    #ef4444;
    --blue:      #3b82f6;
    --text:      #f1f5f9;
    --muted:     #64748b;
    --font:      'Plus Jakarta Sans', sans-serif;
    --mono:      'DM Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
section.main > div { padding: 0 !important; }
.block-container { max-width: 100% !important; padding: 0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: var(--font) !important;
}

/* button */
[data-testid="stButton"] > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity .2s;
}
[data-testid="stButton"] > button:hover { opacity: .85 !important; }

/* tab */
[data-testid="stTabs"] [role="tab"] {
    color: var(--muted) !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    border: none !important;
    padding: .6rem 1.2rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--blue) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border) !important;
    gap: .5rem;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--blue) !important; }

/* metric */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* progress */
[data-testid="stProgress"] > div > div { background: var(--blue) !important; border-radius: 9px; }
[data-testid="stProgress"] > div { background: var(--border) !important; border-radius: 9px; }

/* info/success/warning/error */
[data-testid="stAlert"] { border-radius: 10px !important; font-family: var(--font) !important; }

p, li, span, label { font-family: var(--font) !important; color: var(--text) !important; }
h1,h2,h3,h4 { font-family: var(--font) !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if "hasil"         not in st.session_state: st.session_state.hasil         = None
if "df_bmkg"       not in st.session_state: st.session_state.df_bmkg       = None
if "last_update"   not in st.session_state: st.session_state.last_update   = None
if "kecamatan_sel" not in st.session_state: st.session_state.kecamatan_sel = "Sutojayan"
if "page"          not in st.session_state: st.session_state.page          = "beranda"

# ─────────────────────────────────────────
#  AUTO-REFRESH LOGIC (60 menit)
# ─────────────────────────────────────────
REFRESH_INTERVAL = 60 * 60  # detik
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
#  HELPER UI
# ─────────────────────────────────────────
def status_info(hasil):
    idx = int(np.argmax(hasil))
    labels   = ["AMAN", "WASPADA", "BAHAYA"]
    colors   = ["#10b981", "#f59e0b", "#ef4444"]
    icons    = ["🟢", "🟡", "🔴"]
    return labels[idx], colors[idx], icons[idx], hasil[0]*100, hasil[1]*100, hasil[2]*100

def render_header():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2e 0%,#0a0f1e 60%,#091623 100%);
                border-bottom:1px solid #1e2d45;
                padding:1rem 2rem;display:flex;align-items:center;gap:1rem;">
        <div style="font-size:2rem;">🌊</div>
        <div>
            <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;">
                Prediksi Dini Banjir
            </div>
            <div style="font-size:.78rem;color:#64748b;font-weight:500;">Kabupaten Blitar — Sistem Monitoring Real-time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_status_card(hasil):
    label, color, icon, p_aman, p_waspada, p_bahaya = status_info(hasil)
    now = datetime.fromtimestamp(st.session_state.last_update).strftime("%d %b %Y, %H:%M") if st.session_state.last_update else "—"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#111827,#1a2235);
                border:1px solid {color}44;border-radius:20px;
                padding:2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
                    background:{color}15;border-radius:50%;"></div>
        <div style="position:absolute;top:10px;right:20px;font-size:4rem;opacity:.12;">{icon}</div>
        <div style="font-size:.75rem;font-weight:600;color:#64748b;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">
            STATUS BANJIR SAAT INI
        </div>
        <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:1.5rem;">
            <div style="font-size:2.4rem;">{icon}</div>
            <div style="font-size:2.8rem;font-weight:800;color:{color};letter-spacing:-.03em;line-height:1;">
                {label}
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1.2rem;">
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#10b981;">{p_aman:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;font-weight:500;margin-top:.2rem;">🟢 Aman</div>
            </div>
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#f59e0b;">{p_waspada:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;font-weight:500;margin-top:.2rem;">🟡 Waspada</div>
            </div>
            <div style="background:#0a0f1e66;border:1px solid #1e2d45;border-radius:12px;padding:.8rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{p_bahaya:.1f}%</div>
                <div style="font-size:.72rem;color:#64748b;font-weight:500;margin-top:.2rem;">🔴 Bahaya</div>
            </div>
        </div>
        <div style="font-size:.73rem;color:#64748b;">🕐 Diperbarui: {now} &nbsp;·&nbsp; Auto-refresh setiap 60 menit</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────
def page_beranda():
    st.markdown('<div style="padding:1.5rem 2rem 0;">', unsafe_allow_html=True)

    col_sel, col_btn, col_space = st.columns([2.5, 1.2, 3])
    with col_sel:
        kec = st.selectbox("📍 Pilih Kecamatan", list(KECAMATAN.keys()),
                           index=list(KECAMATAN.keys()).index(st.session_state.kecamatan_sel),
                           label_visibility="collapsed")
        st.session_state.kecamatan_sel = kec
    with col_btn:
        manual = st.button("🔄 Refresh Sekarang")

    st.markdown('</div>', unsafe_allow_html=True)

    # auto-refresh atau manual
    if should_refresh() or manual:
        with st.spinner("Mengambil data BMKG & menjalankan model..."):
            try:
                do_predict(st.session_state.kecamatan_sel)
            except Exception as e:
                st.error(f"Gagal mengambil data: {e}")
                return

    # auto-rerun setelah 60 menit (Streamlit akan rerun otomatis)
    if st.session_state.last_update:
        elapsed = time.time() - st.session_state.last_update
        remaining = REFRESH_INTERVAL - elapsed
        if remaining <= 0:
            st.rerun()

    st.markdown('<div style="padding:0 2rem 2rem;">', unsafe_allow_html=True)

    if st.session_state.hasil is not None:
        hasil = st.session_state.hasil
        df    = st.session_state.df_bmkg

        render_status_card(hasil)

        # Progress bar probabilitas
        st.markdown("**Distribusi Probabilitas Model**")
        label, color, icon, p_aman, p_waspada, p_bahaya = status_info(hasil)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("🟢 **Aman**")
            st.progress(int(p_aman))
        with c2:
            st.markdown("🟡 **Waspada**")
            st.progress(int(p_waspada))
        with c3:
            st.markdown("🔴 **Bahaya**")
            st.progress(int(p_bahaya))

        st.markdown("<br>", unsafe_allow_html=True)

        # Data BMKG
        st.markdown("**📡 Data Prakiraan BMKG – 5 Jam Terdekat**")
        df_show = df[['datetime','temp','humidity','wind','weather','rain']].head(5).copy()
        df_show.columns = ['Waktu','Suhu (°C)','Kelembaban (%)','Angin (km/h)','Cuaca','Curah Hujan (mm/3j)']
        df_show['Curah Hujan (mm/3j)'] = df_show['Curah Hujan (mm/3j)'].round(3)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

def page_informasi():
    st.markdown('<div style="padding:1.5rem 2rem;">', unsafe_allow_html=True)

    kec = st.session_state.kecamatan_sel
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;">
        <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;">Kecamatan Dipilih</div>
        <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;">{kec}</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:.3rem;">Kabupaten Blitar, Jawa Timur</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.hasil is not None:
        hasil = st.session_state.hasil
        df    = st.session_state.df_bmkg

        # Statistik cuaca
        st.markdown("### 📊 Statistik Cuaca")
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("🌡️ Suhu Rata-rata", f"{df['temp'].mean():.1f} °C")
        with c2: st.metric("💧 Kelembaban Rata-rata", f"{df['humidity'].mean():.0f} %")
        with c3: st.metric("💨 Kecepatan Angin", f"{df['wind'].mean():.1f} km/h")
        with c4: st.metric("🌧️ Curah Hujan Est.", f"{df['rain'].mean():.3f} mm/3j")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Tabel Data Prakiraan Lengkap")
        df_show = df[['datetime','temp','humidity','wind','weather','rain']].copy()
        df_show.columns = ['Waktu','Suhu (°C)','Kelembaban (%)','Angin (km/h)','Cuaca','Curah Hujan (mm/3j)']
        df_show['Curah Hujan (mm/3j)'] = df_show['Curah Hujan (mm/3j)'].round(4)
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=380)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏞️ Data Curah Hujan Historis")
        tab1, tab2 = st.tabs(["Pos Sukorejo/Judeg", "Pos Gedok/Bacem"])
        bulan = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        with tab1:
            df_s = pd.DataFrame(rainfall_sukorejo, index=bulan).T
            df_s.index.name = "Tahun"
            st.dataframe(df_s, use_container_width=True)
        with tab2:
            df_b = pd.DataFrame(rainfall_bacem, index=bulan).T
            df_b.index.name = "Tahun"
            st.dataframe(df_b, use_container_width=True)
    else:
        st.info("Belum ada data. Silakan buka halaman Beranda untuk memuat prediksi.")

    st.markdown('</div>', unsafe_allow_html=True)

def page_tentang():
    st.markdown('<div style="padding:1.5rem 2rem;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="max-width:680px;margin:0 auto;">

        <div style="background:linear-gradient(135deg,#111827,#1a2235);
                    border:1px solid #1e2d45;border-radius:20px;padding:2rem;margin-bottom:1.5rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🌊</div>
            <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-bottom:.5rem;">
                Sistem Prediksi Dini Banjir
            </div>
            <div style="font-size:.85rem;color:#94a3b8;line-height:1.7;">
                Sistem ini menggunakan model Hybrid <b style="color:#3b82f6;">LSTM-CNN</b> untuk memprediksi
                potensi banjir di Kabupaten Blitar berdasarkan data cuaca real-time dari API BMKG.
            </div>
        </div>

        <div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.8rem;">Cara Kerja Sistem</div>
            <div style="display:flex;flex-direction:column;gap:.6rem;">
                <div style="display:flex;align-items:flex-start;gap:.8rem;">
                    <div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">01</div>
                    <div style="font-size:.84rem;color:#cbd5e1;">Data cuaca diambil dari API BMKG setiap 60 menit secara otomatis</div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:.8rem;">
                    <div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">02</div>
                    <div style="font-size:.84rem;color:#cbd5e1;">Data dinormalisasi dan dibentuk menjadi sequence 3 timestep (9 jam)</div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:.8rem;">
                    <div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">03</div>
                    <div style="font-size:.84rem;color:#cbd5e1;">Model LSTM mengekstrak pola temporal dari data cuaca</div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:.8rem;">
                    <div style="background:#3b82f622;color:#3b82f6;border-radius:8px;padding:.3rem .6rem;font-size:.75rem;font-weight:700;flex-shrink:0;">04</div>
                    <div style="font-size:.84rem;color:#cbd5e1;">Model CNN mengklasifikasikan hasil LSTM ke 3 status: Aman, Waspada, Bahaya</div>
                </div>
            </div>
        </div>

        <div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.8rem;">Kategori Status Banjir</div>
            <div style="display:flex;flex-direction:column;gap:.6rem;">
                <div style="display:flex;align-items:center;gap:.8rem;background:#10b98112;border:1px solid #10b98133;border-radius:10px;padding:.8rem 1rem;">
                    <span style="font-size:1.2rem;">🟢</span>
                    <div>
                        <div style="font-size:.85rem;font-weight:700;color:#10b981;">AMAN</div>
                        <div style="font-size:.78rem;color:#64748b;">Curah hujan ≤ 1 mm/3 jam — risiko banjir rendah</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:.8rem;background:#f59e0b12;border:1px solid #f59e0b33;border-radius:10px;padding:.8rem 1rem;">
                    <span style="font-size:1.2rem;">🟡</span>
                    <div>
                        <div style="font-size:.85rem;font-weight:700;color:#f59e0b;">WASPADA</div>
                        <div style="font-size:.78rem;color:#64748b;">Curah hujan 1–3 mm/3 jam — perlu persiapan antisipasi</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:.8rem;background:#ef444412;border:1px solid #ef444433;border-radius:10px;padding:.8rem 1rem;">
                    <span style="font-size:1.2rem;">🔴</span>
                    <div>
                        <div style="font-size:.85rem;font-weight:700;color:#ef4444;">BAHAYA</div>
                        <div style="font-size:.78rem;color:#64748b;">Curah hujan > 3 mm/3 jam — potensi banjir tinggi</div>
                    </div>
                </div>
            </div>
        </div>

        <div style="background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:1.5rem;">
            <div style="font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.8rem;">Informasi Pembuat</div>
            <div style="font-size:.84rem;color:#cbd5e1;line-height:1.8;">
                Dibuat sebagai bagian dari penelitian skripsi di bidang prediksi banjir berbasis machine learning.<br>
                Data cuaca bersumber dari <b style="color:#3b82f6;">API BMKG</b> (Badan Meteorologi, Klimatologi, dan Geofisika).<br>
                Data curah hujan historis berasal dari Pos Sukorejo/Judeg dan Pos Gedok/Bacem,
                Kecamatan Sutojayan, Kabupaten Blitar.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LAYOUT UTAMA
# ─────────────────────────────────────────
render_header()

# Navbar tab
tab_beranda, tab_info, tab_tentang = st.tabs(["🏠  Beranda", "📊  Informasi & Statistik", "ℹ️  Tentang Kami"])

with tab_beranda:
    page_beranda()

with tab_info:
    page_informasi()

with tab_tentang:
    page_tentang()

# Auto-rerun setiap 60 menit
if st.session_state.last_update:
    elapsed = time.time() - st.session_state.last_update
    remaining = max(0, REFRESH_INTERVAL - elapsed)
    if remaining < 10:
        time.sleep(1)
        st.rerun()
