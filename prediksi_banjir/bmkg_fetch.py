import requests
import pandas as pd
from rainfall import estimate_rain
from datetime import datetime

def fetch_bmkg():

    url = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=35.05.12.1005"

    response = requests.get(url)

    print("Status API:", response.status_code)

    data = response.json()

    records = []

    for item in data['data'][0]['cuaca']:

        for cuaca in item:

            dt = datetime.fromisoformat(cuaca["local_datetime"])

            # =========================
            # AMBIL RAIN DARI BMKG
            # =========================
            rain = cuaca.get("tp", 0)

            # kalau kosong/null → fallback rainfall.py
            if rain is None:
                rain = estimate_rain(dt.month, dt.year)

            records.append({

                "datetime": dt,
                "temp": cuaca["t"],
                "humidity": cuaca["hu"],
                "wind": cuaca["ws"],
                "weather": cuaca["weather_desc"],
                "rain": rain
            })

    df = pd.DataFrame(records)

    print("\n=== DATA BMKG ===")
    print(df.head())

    return df