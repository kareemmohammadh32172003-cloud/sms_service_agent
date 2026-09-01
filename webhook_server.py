"""
=====================================================================
Webhook server - receives forwarded SMS from your phone and feeds
them into the finance agent automatically.
=====================================================================

This is what the "SMS Forwarder" app on your phone will send its
POST requests to.

Install:
    pip install flask

Run locally for testing:
    python webhook_server.py
    (it starts on http://localhost:5000)
"""

from flask import Flask, request, jsonify
import finance_core as core

app = Flask(__name__)


@app.route("/sms-webhook", methods=["POST"])
def sms_webhook():
    data = request.get_json(force=True, silent=True) or {}

    # Different SMS forwarder apps use different field names for the
    # message body - we check the common ones so this works with
    # whichever app you end up installing.
    raw_text = data.get("text") or data.get("message") or data.get("body") or ""

    if not raw_text:
        return jsonify({"status": "error", "reason": "no message text found in payload"}), 400

    print(f"📩 Received SMS: {raw_text[:100]}")

    try:
        result = core.process_incoming_sms(raw_text)
        print(f"✅ Processed: {result}")
        return jsonify({"status": "ok", "result": result}), 200
    except Exception as e:
        print(f"❌ Error processing SMS: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    # Simple endpoint to check the server is alive - useful once
    # deployed, to confirm it's reachable before configuring the app
    return jsonify({"status": "alive"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)