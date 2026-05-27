import requests
import json

ONESIGNAL_APP_ID = "APP_ID"
REST_API_KEY = "REST_API_KEY"

def send_notification(title, message):

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Key {REST_API_KEY}"
    }

    payload = {
        "app_id": ONESIGNAL_APP_ID,

        "included_segments": ["All"],

        "headings": {
            "en": title
        },

        "contents": {
            "en": message
        },

        "priority": 10
    }

    response = requests.post(
        "https://onesignal.com/api/v1/notifications",
        headers=headers,
        data=json.dumps(payload)
    )

    print("Status:", response.status_code)
    print(response.text)

    if response.status_code == 200:
        return True
    else:
        return False