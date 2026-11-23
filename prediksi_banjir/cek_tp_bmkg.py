"""
Cek cepat respons API BMKG: apakah ada field 'tp' (curah hujan) dan berapa nilainya.
Jalankan:  python cek_tp_bmkg.py      (hanya butuh paket requests)
"""
import requests

ADM4 = "35.05.12.1005"   # Kec. Sutojayan
URL = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={ADM4}"

try:
    r = requests.get(URL, timeout=20)
    print("HTTP status:", r.status_code)
    data = r.json()
except Exception as e:
    raise SystemExit(f"Gagal mengambil data: {e}")

rec = [c for hari in data["data"][0]["cuaca"] for c in hari]
print("Jumlah record (tiap 3 jam):", len(rec), "=> cakupan", len(rec) * 3, "jam")
print("Key pada satu record:", sorted(rec[0].keys()))
if "tp" not in rec[0]:
    print("PERINGATAN: field 'tp' TIDAK ada. Kirim daftar key di atas ke Claude.")
else:
    print("\nlocal_datetime        tp (mm)")
    for c in rec[:10]:
        print(c["local_datetime"], "  ", c["tp"])
    total = sum(float(c.get("tp", 0) or 0) for c in rec)
    print(f"\nTotal tp seluruh record: {total:.1f} mm, rata-rata {total / (len(rec) * 3 / 24):.2f} mm/hari")
