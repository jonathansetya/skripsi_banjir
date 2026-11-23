"""
Uji asap (smoke test) aplikasi dengan data BMKG TIRUAN dan OneSignal TIRUAN (tidak ada notifikasi sungguhan).
Butuh .streamlit/secrets.toml berisi nilai apa saja, mis. ONESIGNAL_APP_ID="x" dan REST_API_KEY="x".
Jalankan dari folder ini:  python uji_asap_app.py
"""
import os, json
from unittest import mock
from streamlit.testing.v1 import AppTest

def fake_bmkg(total_mm):
    steps = 24
    cuaca = [[{"local_datetime": f"2026-11-{22+i//8:02d}T{(i%8)*3:02d}:00:00", "t": 27, "hu": 80, "ws": 5,
               "weather_desc": "Hujan Ringan", "tp": total_mm/steps} for i in range(steps)]]
    return {"data": [{"cuaca": cuaca}]}

class R:
    def __init__(s, j=None, code=200): s._j=j; s.status_code=code; s.text="ok"
    def json(s): return s._j

def run(total_mm, reset=True):
    for f in (["/tmp/banjir_last_update.json", "/tmp/banjir_last_notified.json"] if reset else ["/tmp/banjir_last_update.json"]):
        if os.path.exists(f): os.remove(f)
    posts = []
    def fake_post(url, **kw): posts.append(json.loads(kw["data"])); return R()
    with mock.patch("requests.get", lambda url, **kw: R(fake_bmkg(total_mm))), mock.patch("requests.post", fake_post):
        at = AppTest.from_file("app.py", default_timeout=60).run()
    kartu = " ".join(m.value for m in at.markdown if "STATUS BANJIR SAAT INI" in m.value)
    status = next((s for s in ["BAHAYA","WASPADA","AMAN"] if f"line-height:1;\">{s}</div>" in kartu), None)
    return at, status, posts

if __name__ == "__main__":
    for total, harapan in [(20, "AMAN"), (40, "WASPADA"), (60, "BAHAYA")]:
        at, status, posts = run(total)
        print(f"tp total {total} mm/3 hari -> status UI: {status} (harapan {harapan}); exception: {len(at.exception)}; notifikasi: {[p['headings']['en'] for p in posts]}")
        assert status == harapan and not at.exception
    at, status, posts = run(60); assert len(posts) == 1
    at, status, posts = run(60, reset=False); print("run ulang status sama -> notifikasi tambahan:", len(posts)); assert len(posts) == 0
    print("OK")
