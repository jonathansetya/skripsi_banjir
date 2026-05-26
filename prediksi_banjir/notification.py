import requests
import json

ONESIGNAL_APP_ID = "1720c94d-33da-4109-9423-0af124b42a5e"
REST_API_KEY = "os_v2_app_c4qmstjt3jaqtfbdblysjnbklyywkfrtbkwu6vebriplfrdl2ybz3tql75alurpdxpv5oyx53qwdxuqkli2cfuw2fmevnfclzvjdxra"

def send_notification(title, message):

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Basic {REST_API_KEY}"
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