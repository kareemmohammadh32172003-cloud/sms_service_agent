"""
=====================================================================
Webhook server - receives forwarded SMS from a phone and feeds them
into the finance agent, scoped to whichever user owns the token in
the URL.

Each user gets their OWN webhook URL:
    https://<your-domain>/sms-webhook/<their-personal-token>

They get this URL from the Telegram bot after /start - see
telegram_bot.py. They paste it into their SMS Forwarder app.

Run locally for testing:
    python webhook_server.py
"""

import threading
from datetime import date
from collections import defaultdict
from flask import Flask, request, jsonify
import finance_core as core

app = Flask(__name__)

# --------------------------------------------------------------
# Simple in-memory daily rate limit per user, to stop a leaked or
# guessed webhook URL from being spammed (each processed message
# costs a Groq API call). This resets if the service restarts and
# only works per-instance - fine at this project's scale (1 Railway
# replica); a shared store (e.g. a Supabase table) would be needed
# if this ever ran across multiple instances.
# --------------------------------------------------------------
DAILY_MESSAGE_LIMIT = 200
_request_counts = defaultdict(lambda: {"date": None, "count": 0})
_lock = threading.Lock()


def _under_rate_limit(user_id: str) -> bool:
    today = date.today().isoformat()
    with _lock:
        entry = _request_counts[user_id]
        if entry["date"] != today:
            entry["date"] = today
            entry["count"] = 0
        entry["count"] += 1
        return entry["count"] <= DAILY_MESSAGE_LIMIT


@app.route("/sms-webhook/<token>", methods=["POST"])
def sms_webhook(token):
    user = core.get_user_by_token(token)
    if not user:
        return jsonify({"status": "error", "reason": "invalid token"}), 404

    if not _under_rate_limit(user["id"]):
        print(f"Rate limit exceeded for user {user['id']}")
        return jsonify({"status": "error", "reason": "daily message limit reached"}), 429

    data = request.get_json(force=True, silent=True) or {}
    raw_text = data.get("text") or data.get("message") or data.get("body") or ""

    if not raw_text:
        return jsonify({"status": "error", "reason": "no message text found in payload"}), 400

    print(f"SMS for user {user['id']}: {raw_text[:100]}")

    try:
        result = core.process_incoming_sms(
            raw_text, user_id=user["id"], telegram_chat_id=user.get("telegram_chat_id")
        )
        print(f"Processed: {result}")
        return jsonify({"status": "ok", "result": result}), 200
    except Exception as e:
        print(f"Error processing SMS: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)