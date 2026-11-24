"""
risk_engine.py - logika status risiko untuk prototipe aplikasi (tanpa Streamlit, supaya bisa diuji otomatis).

Dasar ambang:
- BAHAYA : 150 mm/dasarian = ambang peringatan dini curah hujan tinggi BMKG
           (disampaikan narasumber BMKG; verifikasi ke peraturan BMKG sebelum dikutip di naskah).
- WASPADA: 102 mm/dasarian = kuartil atas (persentil ke-75) total hujan dasarian CHIRPS Sutojayan 2016-2026.
           Ditetapkan dari sebaran data (bukan disetel terhadap kejadian banjir); sifatnya eksploratif.

CATATAN KETERBATASAN (harus ditulis di naskah):
Ambang di atas divalidasi untuk total hujan SATU DASARIAN (10 hari, data CHIRPS). Prototipe ini menerapkannya pada
prakiraan BMKG ~3 hari ke depan dengan menyetarakan INTENSITAS RATA-RATA (mm/hari). Penyetaraan ini belum divalidasi.
"""
import json
import math
from dataclasses import dataclass

AMBANG_BAHAYA_DASARIAN_MM = 150.0
AMBANG_WASPADA_DASARIAN_MM = 102.0
HARI_DASARIAN = 10.0
INTENSITAS_BAHAYA = AMBANG_BAHAYA_DASARIAN_MM / HARI_DASARIAN      # 15.0 mm/hari
INTENSITAS_WASPADA = AMBANG_WASPADA_DASARIAN_MM / HARI_DASARIAN    # 10.2 mm/hari
JAM_PER_RECORD = 3
MIN_JAM_CAKUPAN = 24
LEVELS = ["AMAN", "WASPADA", "BAHAYA"]


@dataclass
class Assessment:
    status: str
    status_idx: int
    total_mm: float
    jam_cakupan: int
    intensitas_mm_hari: float
    n_record: int
    simulasi: bool = False   # True bila data hujan diganti nilai uji (hanya untuk pengujian)


def _bersih(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or v < 0:
        return 0.0
    return v


def classify_intensity(intensitas_mm_hari):
    if intensitas_mm_hari >= INTENSITAS_BAHAYA:
        return 2
    if intensitas_mm_hari >= INTENSITAS_WASPADA:
        return 1
    return 0


def classify_dasarian_total(total_mm):
    """Status dari TOTAL hujan satu dasarian (dipakai tab uji ulang historis)."""
    if total_mm >= AMBANG_BAHAYA_DASARIAN_MM:
        return LEVELS[2]
    if total_mm >= AMBANG_WASPADA_DASARIAN_MM:
        return LEVELS[1]
    return LEVELS[0]


def assess_forecast(rain_mm_per_record, jam_per_record=JAM_PER_RECORD):
    """rain_mm_per_record: curah hujan (mm) tiap langkah prakiraan 3-jaman (field 'tp' API BMKG)."""
    vals = [_bersih(v) for v in rain_mm_per_record]
    jam = len(vals) * jam_per_record
    if jam < MIN_JAM_CAKUPAN:
        raise ValueError(f"Data prakiraan hanya mencakup {jam} jam (minimal {MIN_JAM_CAKUPAN} jam).")
    total = sum(vals)
    intensitas = total / (jam / 24.0)
    idx = classify_intensity(intensitas)
    return Assessment(LEVELS[idx], idx, total, jam, intensitas, len(vals))


def next_state(prev_status, current_status):
    """Kembalikan (kirim_notifikasi, status_tersimpan_baru). Notifikasi hanya saat status NAIK."""
    p = LEVELS.index(prev_status) if prev_status in LEVELS else 0
    c = LEVELS.index(current_status)
    return (c > p), current_status


def build_message(a):
    if a.status == "BAHAYA":
        judul = "🚨 STATUS BAHAYA"
    elif a.status == "WASPADA":
        judul = "⚠️ STATUS WASPADA"
    else:
        judul = "ℹ️ STATUS AMAN"
    if getattr(a, "simulasi", False):
        judul = "[SIMULASI UJI] " + judul
    isi = (f"Prakiraan hujan {a.total_mm:.0f} mm dalam {a.jam_cakupan} jam ke depan "
           f"(rata-rata {a.intensitas_mm_hari:.1f} mm/hari). "
           "Prototipe penelitian, bukan peringatan resmi BMKG/BPBD.")
    return judul, isi


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f).get("last_notified")
    except Exception:
        return None


def save_state(path, status):
    try:
        with open(path, "w") as f:
            json.dump({"last_notified": status}, f)
    except Exception:
        pass
