"""
=====================================================================
Monthly summary job - run this once a month (e.g. via Railway's
Cron Job feature, or GitHub Actions scheduled workflow) on the 1st
of every month. Loops over every registered user, builds a natural
-language summary of last month, and sends it via Telegram.

Run manually to test:
    python monthly_summary.py
"""

import os
import requests
from dotenv import load_dotenv
import finance_core as core

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


def send_telegram_message(chat_id: int, text: str):
    resp = requests.post(TELEGRAM_API, json={"chat_id": chat_id, "text": text})
    if not resp.ok:
        print(f"Failed to send to {chat_id}: {resp.text}")


def run():
    users = core.list_all_users()
    print(f"Building monthly summaries for {len(users)} user(s)...")

    for user in users:
        summary = core.build_monthly_summary_text(user["id"])
        if summary is None:
            print(f"  user {user['id']}: no transactions last month, skipping")
            continue

        header = "ملخص مصاريفك للشهر اللي فات:\n\n"
        send_telegram_message(user["telegram_chat_id"], header + summary)
        print(f"  user {user['id']}: summary sent")


if __name__ == "__main__":
    run()