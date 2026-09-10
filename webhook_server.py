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

import hmac
import hashlib
import os
import threading
from datetime import date
from collections import defaultdict
from flask import Flask, request, jsonify
import finance_core as core

app = Flask(__name__)

# =================================================================
# Per-user HMAC secret: this is the real, unforgeable proof that a
# request actually came from YOUR phone's SMS Forwarder app, not from
# someone who found/guessed the webhook URL and is POSTing fake data
# directly (e.g. via curl). The app signs the exact raw request body
# with a secret you set once in its "Sign with HMAC-SHA-256" option,
# and sends the signature in an X-Signature header. We recompute the
# same signature here and compare - only someone who knows the secret
# can produce a match, and the secret itself never travels in the URL
# or the body, so it can't leak the way a URL token could.
# =================================================================

def _verify_hmac_signature(secret: str, raw_body: bytes, signature_header: str) -> bool:
    if not signature_header:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.lower())

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

    hmac_secret = user.get("hmac_secret")
    if hmac_secret:
        raw_body = request.get_data()
        signature = request.headers.get("X-Signature", "")
        if not _verify_hmac_signature(hmac_secret, raw_body, signature):
            print(f"Rejected request for user {user['id']}: invalid or missing HMAC signature")
            return jsonify({"status": "error", "reason": "invalid signature"}), 401

    if not _under_rate_limit(user["id"]):
        print(f"Rate limit exceeded for user {user['id']}")
        return jsonify({"status": "error", "reason": "daily message limit reached"}), 429

    data = request.get_json(force=True, silent=True) or {}
    raw_text = data.get("text") or data.get("message") or data.get("body") or ""
    sender = data.get("sender") or data.get("from") or data.get("number") or ""

    if not raw_text:
        return jsonify({"status": "error", "reason": "no message text found in payload"}), 400

    print(f"SMS for user {user['id']} from '{sender}': {raw_text[:100]}")

    try:
        result = core.process_incoming_sms(
            raw_text, user_id=user["id"], telegram_chat_id=user.get("telegram_chat_id"), sender=sender
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