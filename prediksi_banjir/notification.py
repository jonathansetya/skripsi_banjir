import json
import requests
import streamlit as st

ONESIGNAL_APP_ID = st.secrets["ONESIGNAL_APP_ID"]
REST_API_KEY = st.secrets["REST_API_KEY"]
# Selama pengembangan buat segmen khusus (mis. "Test Users") di OneSignal dan isi di secrets:
# ONESIGNAL_SEGMENT = "Test Users"
SEGMENT = st.secrets.get("ONESIGNAL_SEGMENT", "All")


def _auth():
    """Key baru OneSignal (diawali 'os_') memakai skema 'Key'; key lama memakai 'Basic'."""
    if REST_API_KEY.startswith("os_"):
        return "https://api.onesignal.com/notifications?c=push", f"Key {REST_API_KEY}"
    return "https://onesignal.com/api/v1/notifications", f"Basic {REST_API_KEY}"


def send_notification(title, message):
    url, authorization = _auth()
    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": authorization}
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "included_segments": [SEGMENT],
        "headings": {"en": title},
        "contents": {"en": message},
        "priority": 10,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
    except requests.RequestException as e:
        print("Gagal mengirim notifikasi:", e)
        return False
    print("OneSignal status:", response.status_code, response.text[:200])
    return response.status_code in (200, 201)
