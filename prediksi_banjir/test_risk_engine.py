import pytest
from risk_engine import (classify_dasarian_total, assess_forecast, classify_intensity, next_state, build_message,
                         INTENSITAS_BAHAYA, INTENSITAS_WASPADA, load_state, save_state)

def lagi(total_mm, n=24):
    return [total_mm / n] * n          # n langkah 3-jaman -> 72 jam

def test_ambang_setara_dasarian():
    assert INTENSITAS_BAHAYA == 15.0
    assert INTENSITAS_WASPADA == pytest.approx(10.2)

def test_kelas_batas():
    assert classify_intensity(0) == 0
    assert classify_intensity(10.19) == 0
    assert classify_intensity(10.2) == 1
    assert classify_intensity(14.99) == 1
    assert classify_intensity(15.0) == 2

def test_akumulasi_72_jam():
    a = assess_forecast(lagi(45.0))     # 45 mm/3 hari = 15 mm/hari
    assert a.status == "BAHAYA" and a.jam_cakupan == 72 and a.total_mm == pytest.approx(45.0)
    assert assess_forecast(lagi(30.6)).status == "WASPADA"   # 10.2 mm/hari
    assert assess_forecast(lagi(10.0)).status == "AMAN"

def test_data_kurang_dari_24_jam():
    with pytest.raises(ValueError):
        assess_forecast([1.0] * 7)

def test_nilai_rusak_dianggap_nol():
    a = assess_forecast([None, "x", float("nan"), -5] + [0] * 20)
    assert a.total_mm == 0 and a.status == "AMAN"

def test_notifikasi_hanya_saat_naik():
    assert next_state(None, "AMAN") == (False, "AMAN")
    assert next_state("AMAN", "WASPADA") == (True, "WASPADA")
    assert next_state("WASPADA", "WASPADA") == (False, "WASPADA")
    assert next_state("WASPADA", "BAHAYA") == (True, "BAHAYA")
    assert next_state("BAHAYA", "WASPADA") == (False, "WASPADA")   # turun: tidak kirim
    assert next_state("WASPADA", "AMAN") == (False, "AMAN")
    assert next_state("AMAN", "BAHAYA") == (True, "BAHAYA")        # lompat langsung

def test_pesan_tidak_mengklaim_waktu_kejadian():
    judul, isi = build_message(assess_forecast(lagi(60.0)))
    assert "BAHAYA" in judul
    assert "dalam 24 jam" not in isi and "dalam 2 jam" not in isi and "dalam 1 jam" not in isi
    assert "bukan peringatan resmi" in isi

def test_state_persisten(tmp_path):
    p = str(tmp_path / "s.json")
    assert load_state(p) is None
    save_state(p, "WASPADA")
    assert load_state(p) == "WASPADA"

def test_klasifikasi_total_dasarian():
    assert classify_dasarian_total(101.9) == "AMAN"
    assert classify_dasarian_total(102.0) == "WASPADA"
    assert classify_dasarian_total(149.9) == "WASPADA"
    assert classify_dasarian_total(150.0) == "BAHAYA"
