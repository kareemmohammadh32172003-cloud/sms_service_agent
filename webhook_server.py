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

from flask import Flask, request, jsonify
import finance_core as core

app = Flask(__name__)


@app.route("/sms-webhook/<token>", methods=["POST"])
def sms_webhook(token):
    user = core.get_user_by_token(token)
    if not user:
        # Unknown token - don't process, don't reveal why (avoid leaking
        # which tokens are valid to someone probing the endpoint).
        return jsonify({"status": "error", "reason": "invalid token"}), 404

    data = request.get_json(force=True, silent=True) or {}
    raw_text = data.get("text") or data.get("message") or data.get("body") or ""

    if not raw_text:
        return jsonify({"status": "error", "reason": "no message text found in payload"}), 400

    print(f"SMS for user {user['id']}: {raw_text[:100]}")

    try:
        result = core.process_incoming_sms(raw_text, user_id=user["id"])
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