import requests
import pandas as pd

from rainfall import estimate_rain
from datetime import datetime

def fetch_bmkg():

    url = "https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4=35.05.12.1005"

    response = requests.get(url)

    # cek status API
    print("Status API:", response.status_code)

    data = response.json()

    records = []

    for item in data['data'][0]['cuaca']:

        for cuaca in item:

            dt = datetime.fromisoformat(cuaca["local_datetime"])

            rain = estimate_rain(dt.month, dt.year)

            records.append({
                "datetime": cuaca["local_datetime"],
                "temp": cuaca["t"],
                "humidity": cuaca["hu"],
                "wind": cuaca["ws"],
                "weather": cuaca["weather_desc"],
                "rain": rain
            })

    df = pd.DataFrame(records)

    return df