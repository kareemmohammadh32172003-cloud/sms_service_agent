"""
Quick test script to send a sample SMS to your local webhook server.
Run this in a SECOND terminal while webhook_server.py is running in the first.

Usage:
    python test_webhook.py
"""

import requests

url = "http://localhost:5000/sms-webhook"
payload = {"text": "تم خصم 150 جنيه من حسابك لصالح أوبر"}

response = requests.post(url, json=payload)

print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")